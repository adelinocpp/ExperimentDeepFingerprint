"""
Módulo de treinamento para DeepPrint com suporte a debug e prod
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from tqdm import tqdm
from datetime import datetime
import psutil
import gc

from models_base import create_model
from config import TRAINING_CONFIG, OPTIMIZER_CONFIG, LOSS_CONFIG, LOGGING_CONFIG


class CenterLoss(nn.Module):
    """Center Loss para melhorar discriminação de embeddings.
    
    Baseado na implementação original do DeepPrint:
    - Centros são inicializados com valores normalizados
    - Centros são atualizados dinamicamente durante o forward (não são parâmetros treináveis)
    - Usa alpha como "learning rate" para atualização dos centros
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """
    
    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        # Registrar como buffer (não é parâmetro treinável, mas é salvo no state_dict)
        self.register_buffer(
            "centers",
            F.normalize(torch.randn(num_classes, feat_dim), dim=1),
            persistent=True,
        )
        
    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (batch_size, feat_dim) embeddings normalizados
            labels: (batch_size,) labels inteiros
        
        Returns:
            loss: soma das distâncias L2 ao quadrado
        """
        # Garantir que labels estão no range válido
        labels = labels.clamp(0, self.num_classes - 1).long()
        
        # Obter os centros correspondentes a cada label do batch
        with torch.no_grad():
            batch_centers = torch.index_select(self.centers, 0, labels)
        
        # Calcular diferença entre embeddings e seus centros
        diff = feat - batch_centers
        
        # Atualizar centros dinamicamente (como no original)
        with torch.no_grad():
            self.centers.index_add_(0, labels, diff, alpha=self.alpha)
        
        # Retornar soma das distâncias L2 ao quadrado (como no original)
        return torch.sum(diff ** 2)


class TripletLoss(nn.Module):
    """Triplet Loss com mineração automática de triplets (batch all strategy)"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,)
        
        Returns:
            triplet_loss: loss média sobre todos os triplets válidos
        """
        # Calcular distâncias pareadas entre todos os embeddings
        # dist[i, j] = ||embeddings[i] - embeddings[j]||_2
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Criar máscaras para pares positivos e negativos
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (batch, batch)
        
        # Para cada anchor, encontrar:
        # - Positivo mais difícil (hard positive): mesmo label, maior distância
        # - Negativo mais difícil (hard negative): label diferente, menor distância
        
        batch_size = embeddings.size(0)
        losses = []
        
        for i in range(batch_size):
            # Positivos: mesma classe, exceto ele mesmo
            pos_mask = labels_equal[i].clone()
            pos_mask[i] = False
            
            if not pos_mask.any():
                continue  # Não há positivos neste batch para este anchor
            
            # Negativos: classe diferente
            neg_mask = ~labels_equal[i]
            
            if not neg_mask.any():
                continue  # Não há negativos neste batch
            
            # Pegar todas as distâncias positivas e negativas
            pos_dists = pairwise_dist[i][pos_mask]
            neg_dists = pairwise_dist[i][neg_mask]
            
            # Criar todos os triplets (batch all)
            # Para cada positivo e cada negativo, calcular loss
            for pos_dist in pos_dists:
                for neg_dist in neg_dists:
                    loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                    losses.append(loss)
        
        if len(losses) == 0:
            # Nenhum triplet válido no batch
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return torch.stack(losses).mean()


class DeepPrintTrainer:
    """Trainer para DeepPrint com suporte a debug e prod"""
    
    def __init__(
        self,
        experiment_dir: Path,
        model_type: str = "baseline",
        mode: str = "debug",
        texture_embedding_dims: int = 96,
        minutia_embedding_dims: int = 96,
    ):
        self.experiment_dir = experiment_dir
        self.mode = mode
        self.texture_embedding_dims = texture_embedding_dims
        self.minutia_embedding_dims = minutia_embedding_dims
        
        # Configuração
        self.config = TRAINING_CONFIG[mode]
        
        # Logging (deve ser configurado antes de _setup_device)
        self.logger = self._setup_logging()
        self.logger.info(f"Iniciando treinamento em modo {mode}")
        
        # Dispositivo
        self.device = self._setup_device()
        self.logger.info(f"Dispositivo: {self.device}")
        
        # Criar diretórios
        self._create_directories()
        
        # Modelo - todos baseados no baseline (STN + 2 branches)
        if model_type == "baseline":
            from models_base import DeepPrintBaseline
            self.model = DeepPrintBaseline(
                texture_embedding_dims=texture_embedding_dims,
                minutia_embedding_dims=minutia_embedding_dims
            ).to(self.device)
        elif model_type == "enhanced_representation":
            from models_base import DeepPrintEnhancedRepresentation
            self.model = DeepPrintEnhancedRepresentation(
                texture_embedding_dims=texture_embedding_dims,
                minutia_embedding_dims=minutia_embedding_dims
            ).to(self.device)
        elif model_type == "spatial_attention":
            from models_base import DeepPrintSpatialAttention
            self.model = DeepPrintSpatialAttention(
                texture_embedding_dims=texture_embedding_dims,
                minutia_embedding_dims=minutia_embedding_dims
            ).to(self.device)
        elif model_type == "reranking":
            from models_base import DeepPrintWithReranking
            self.model = DeepPrintWithReranking(
                texture_embedding_dims=texture_embedding_dims,
                minutia_embedding_dims=minutia_embedding_dims
            ).to(self.device)
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
        # Otimizador
        self.optimizer = optim.Adam(
            self.model.parameters(),
            **OPTIMIZER_CONFIG["adam"]
        )
        
        # Scheduler (melhorias para datasets grandes)
        self.scheduler = None
        self.warmup_epochs = 0
        if OPTIMIZER_CONFIG.get("use_lr_scheduler", False):
            self.warmup_epochs = OPTIMIZER_CONFIG.get("warmup_epochs", 0)
            # Cosine annealing após warmup
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"] - self.warmup_epochs,
                eta_min=OPTIMIZER_CONFIG.get("min_lr", 1e-6)
            )
            self.logger.info(f"Scheduler ativado: Cosine Annealing com {self.warmup_epochs} épocas de warmup")
        
        # Funções de perda
        self.criterion_center_texture = None  # Para texture embedding
        self.criterion_center_minutia = None  # Para minutiae embedding
        self.criterion_triplet = TripletLoss(margin=1.0)
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if self.config["use_gpu"] and self.config["mixed_precision"] else None
        
        # Histórico
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
        
    def _setup_device(self) -> torch.device:
        """Configurar dispositivo (CPU ou GPU)"""
        if self.config["use_gpu"] and torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU disponível: {gpu_name}")
        else:
            device = torch.device("cpu")
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging"""
        logger = logging.getLogger(f"DeepPrint_{self.mode}")
        logger.setLevel(LOGGING_CONFIG["log_level"])
        
        # Handler para arquivo
        log_file = self.experiment_dir / "logs" / f"training_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(LOGGING_CONFIG["log_level"])
        
        # Handler para console
        ch = logging.StreamHandler()
        ch.setLevel(LOGGING_CONFIG["log_level"])
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _create_directories(self):
        """Criar diretórios necessários"""
        for subdir in ["models", "logs", "results", "checkpoints"]:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Obter uso de memória"""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        result = {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
        }
        
        if torch.cuda.is_available():
            result["gpu_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return result
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Treinar uma epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=self.mode != "debug")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass com mixed precision
            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(images)
                    embedding = outputs["embedding"]
                    logits = outputs.get("logits", None)
                    
                    # Calcular perda
                    loss = self._compute_loss(embedding, labels, logits)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass simples (igual ao original)
                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels)
                
                # Backward pass simples (igual ao original - sem gradient clipping)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log periódico
            if (batch_idx + 1) % LOGGING_CONFIG["log_interval"] == 0:
                avg_loss = total_loss / num_batches
                mem = self._get_memory_usage()
                self.logger.info(
                    f"Epoch {epoch} Batch {batch_idx+1}/{len(train_loader)} - "
                    f"Loss: {avg_loss:.4f} - "
                    f"Mem: {mem['rss_mb']:.1f}MB"
                )
                
                if self.scaler:
                    pbar.set_postfix({"loss": avg_loss, "scale": self.scaler.get_scale()})
                else:
                    pbar.set_postfix({"loss": avg_loss})
            
            # Limpeza de memória
            if self.config["use_gpu"]:
                torch.cuda.empty_cache()
            gc.collect()
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validar modelo
        
        Calcula CrossEntropy + Center Loss (IGUAL ao treino, fiel ao DeepPrint original).
        Com remapeamento global de labels, podemos usar as mesmas losses.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=self.mode != "debug")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                batch_loss = self._compute_loss(outputs, labels)
                
                total_loss += batch_loss.item()
                num_batches += 1
                
                pbar.set_postfix({"loss": total_loss / num_batches})
                
                # Limpeza agressiva de memória GPU
                if self.config["use_gpu"] and num_batches % 5 == 0:
                    torch.cuda.empty_cache()
        
        # Limpeza final
        if self.config["use_gpu"]:
            torch.cuda.empty_cache()
        gc.collect()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def _compute_loss(self, outputs: Dict, labels: torch.Tensor) -> torch.Tensor:
        """Computar perda total para DeepPrint (LocTexMinu)
        
        Baseado na implementação original do DeepPrint:
        - Texture: CrossEntropy + Center Loss
        - Minutiae: CrossEntropy + Center Loss
        - Pesos: W_CROSS_ENTROPY = 1.0, W_CENTER_LOSS = 0.125
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Texture branch loss
        if "texture_logits" in outputs and "texture_embedding" in outputs:
            texture_logits = outputs["texture_logits"]
            texture_embedding = outputs["texture_embedding"]
            
            # CrossEntropy
            ce_loss_texture = F.cross_entropy(texture_logits, labels)
            total_loss = total_loss + LOSS_CONFIG["softmax_loss_weight"] * ce_loss_texture
            
            # Center Loss
            if self.criterion_center_texture is not None:
                center_loss_texture = self.criterion_center_texture(texture_embedding, labels)
                total_loss = total_loss + LOSS_CONFIG["center_loss_weight"] * center_loss_texture
        
        # Minutiae branch loss
        if "minutia_logits" in outputs and "minutia_embedding" in outputs:
            minutia_logits = outputs["minutia_logits"]
            minutia_embedding = outputs["minutia_embedding"]
            
            # CrossEntropy
            ce_loss_minutia = F.cross_entropy(minutia_logits, labels)
            total_loss = total_loss + LOSS_CONFIG["softmax_loss_weight"] * ce_loss_minutia
            
            # Center Loss
            if self.criterion_center_minutia is not None:
                center_loss_minutia = self.criterion_center_minutia(minutia_embedding, labels)
                total_loss = total_loss + LOSS_CONFIG["center_loss_weight"] * center_loss_minutia
        
        # Fallback para modelos texture-only (exp1-exp3)
        if "logits" in outputs and "embedding" in outputs:
            if "texture_logits" not in outputs:  # Só se não for LocTexMinu
                logits = outputs["logits"]
                embedding = outputs["embedding"]
                
                ce_loss = F.cross_entropy(logits, labels)
                total_loss = total_loss + LOSS_CONFIG["softmax_loss_weight"] * ce_loss
                
                if self.criterion_center_texture is not None:
                    center_loss = self.criterion_center_texture(embedding, labels)
                    total_loss = total_loss + LOSS_CONFIG["center_loss_weight"] * center_loss
        
        return total_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        num_classes: Optional[int] = None,
        resume: bool = False,
    ):
        """Treinar modelo
        
        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação
            num_epochs: número de epochs (usa config se None)
            num_classes: número total de classes (origens únicas) no dataset
            resume: se True, tenta retomar do último checkpoint
        """
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]
        
        start_epoch = 1
        best_val_loss = float("inf")
        
        # Tentar retomar de checkpoint se solicitado
        if resume:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path:
                start_epoch = self.load_checkpoint(checkpoint_path)
                # Restaurar best_val_loss do histórico
                if self.history.get("val_loss"):
                    best_val_loss = min(self.history["val_loss"])
                self.logger.info(f"Retomando treinamento da época {start_epoch}")
            else:
                self.logger.info("Nenhum checkpoint encontrado, iniciando do zero")
        
        # Obter número de classes do dataset ou calcular (apenas se não retomou)
        if self.criterion_center_texture is None:
            if num_classes is None:
                if hasattr(train_loader.dataset, 'num_classes'):
                    num_classes = train_loader.dataset.num_classes
                else:
                    all_labels = set()
                    for _, labels in train_loader:
                        all_labels.update(labels.tolist())
                    num_classes = len(all_labels)
            
            self.logger.info(f"Número total de classes (origens únicas): {num_classes}")
            
            # Configurar classificador no modelo
            if hasattr(self.model, 'set_num_classes'):
                self.model.set_num_classes(num_classes)
                # Adicionar parâmetros dos classificadores ao optimizer
                if hasattr(self.model, 'texture_classifier') and self.model.texture_classifier is not None:
                    self.optimizer.add_param_group({'params': self.model.texture_classifier.parameters()})
                if hasattr(self.model, 'minutia_classifier') and self.model.minutia_classifier is not None:
                    self.optimizer.add_param_group({'params': self.model.minutia_classifier.parameters()})
                if hasattr(self.model, 'classifier') and self.model.classifier is not None:
                    self.optimizer.add_param_group({'params': self.model.classifier.parameters()})
                self.logger.info(f"Classificador configurado com {num_classes} classes")
            
            # Inicializar Center Loss para texture e minutiae branches
            if self.criterion_center_texture is None:
                self.logger.info(f"Inicializando Center Loss (Texture) com {num_classes} classes e {self.texture_embedding_dims} dims")
                self.criterion_center_texture = CenterLoss(
                    num_classes=num_classes,
                    feat_dim=self.texture_embedding_dims,
                    alpha=0.01
                ).to(self.device)
        
            if self.criterion_center_minutia is None and hasattr(self.model, 'minutia_embedding_dims'):
                self.logger.info(f"Inicializando Center Loss (Minutiae) com {num_classes} classes e {self.minutia_embedding_dims} dims")
                self.criterion_center_minutia = CenterLoss(
                    num_classes=num_classes,
                    feat_dim=self.minutia_embedding_dims,
                    alpha=0.01
                ).to(self.device)
                self.logger.info(f"Center Loss inicializado com {num_classes} classes")
        
        patience_counter = 0
        max_patience = 20  # Aumentado de 10 para permitir mais exploração
        
        # LR Scheduler - reduz LR quando val_loss estagnar
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        for epoch in range(start_epoch, num_epochs + 1):
            # Treinar
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validar
            val_metrics = self.validate(val_loader, epoch)
            
            # Atualizar histórico
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            
            # LR Scheduler - reduz LR quando val_loss estagnar
            scheduler.step(val_metrics["loss"])
            
            # Checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics["loss"], is_best=True)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                self.logger.info(f"Early stopping em epoch {epoch}")
                break
            
            # Log
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
        
        self.logger.info("Treinamento concluído")
        self._save_history()
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Salvar checkpoint: apenas latest (retomada) e best (melhor modelo)"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "center_loss_texture_state": self.criterion_center_texture.state_dict() if self.criterion_center_texture else None,
            "center_loss_minutia_state": self.criterion_center_minutia.state_dict() if self.criterion_center_minutia else None,
            "num_classes": self.criterion_center_texture.num_classes if self.criterion_center_texture else None,
        }
        
        # Sempre salvar checkpoint_latest (retomada)
        latest_path = self.experiment_dir / "checkpoints" / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Salvar best_model se for o melhor
        if is_best:
            best_path = self.experiment_dir / "models" / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Melhor modelo salvo: época {epoch}, val_loss={val_loss:.4f}")
    
    def _save_history(self):
        """Salvar histórico de treinamento"""
        history_file = self.experiment_dir / "results" / f"training_history_{self.mode}.json"
        
        # Converter para formato JSON-serializable
        history_json = {
            "train_loss": self.history["train_loss"],
            "val_loss": self.history["val_loss"],
        }
        
        with open(history_file, "w") as f:
            json.dump(history_json, f, indent=2)
        
        self.logger.info(f"Histórico salvo em {history_file}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Carregar checkpoint e retornar a época para retomada
        
        Returns:
            start_epoch: época para iniciar (epoch + 1 do checkpoint)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []})
        
        # Restaurar CenterLoss para texture branch
        if checkpoint.get("center_loss_texture_state") and checkpoint.get("num_classes"):
            self.criterion_center_texture = CenterLoss(
                num_classes=checkpoint["num_classes"],
                feat_dim=self.texture_embedding_dims,
                alpha=0.01
            ).to(self.device)
            self.criterion_center_texture.load_state_dict(checkpoint["center_loss_texture_state"])
        
        # Restaurar CenterLoss para minutiae branch
        if checkpoint.get("center_loss_minutia_state") and checkpoint.get("num_classes"):
            self.criterion_center_minutia = CenterLoss(
                num_classes=checkpoint["num_classes"],
                feat_dim=self.minutia_embedding_dims,
                alpha=0.01
            ).to(self.device)
            self.criterion_center_minutia.load_state_dict(checkpoint["center_loss_minutia_state"])
        
        start_epoch = checkpoint.get("epoch", 0) + 1
        self.logger.info(f"Checkpoint carregado de {checkpoint_path}, retomando da época {start_epoch}")
        return start_epoch
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Encontrar o checkpoint mais recente para retomada
        
        Returns:
            Path do checkpoint mais recente ou None se não existir
        """
        latest_path = self.experiment_dir / "checkpoints" / "checkpoint_latest.pt"
        if latest_path.exists():
            return latest_path
        
        # Fallback: procurar o checkpoint com maior época
        checkpoints_dir = self.experiment_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None
        
        checkpoints = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None
        
        # Ordenar por época (extrair número do nome)
        def get_epoch(p):
            try:
                return int(p.stem.split("_")[-1])
            except:
                return 0
        
        checkpoints.sort(key=get_epoch, reverse=True)
        return checkpoints[0]
