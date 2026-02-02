"""
Módulo de treinamento para DeepPrint com suporte a debug e prod
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast, GradScaler
from minutia_map_generator import batch_load_minutia_maps
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
from config import TRAINING_CONFIG, OPTIMIZER_CONFIG, LOSS_CONFIG, LOGGING_CONFIG, get_center_loss_weight
from losses import get_loss_function, CenterLoss as CenterLossNew


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
        
        # Otimizador - CORRIGIDO: Paper usa RMSprop, não Adam!
        # STN (Localization Network) precisa de LR menor (3.5% do base LR)
        self.optimizer = self._create_optimizer()
        
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
        self.loss_type = LOSS_CONFIG.get("loss_type", "center").lower()  # "center" ou "arcface"
        self.criterion_center_texture = None  # Para texture embedding (center loss)
        self.criterion_center_minutia = None  # Para minutiae embedding (center loss)
        self.arcface_criterion = None  # Para ArcFace loss (se configurado)
        self.criterion_triplet = TripletLoss(margin=1.0)
        self.center_loss_weight = LOSS_CONFIG["center_loss_base_weight"]  # Será atualizado em train()
        
        # Mixed precision
        self.scaler = GradScaler(device='cuda') if self.config["use_gpu"] and self.config["mixed_precision"] else None
        
        # Histórico
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
        
    def _create_optimizer(self):
        """
        Cria otimizador (Adam ou RMSprop) com learning rate diferenciado para STN.
        """
        optimizer_type = OPTIMIZER_CONFIG.get("optimizer", "adam")
        opt_config = OPTIMIZER_CONFIG[optimizer_type]
        base_lr = opt_config["lr"]
        loc_lr_scale = OPTIMIZER_CONFIG.get("localization_network_lr_scale", 0.035)

        # Separar parâmetros do Localization Network (STN)
        loc_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "localization" in name.lower() or "stn" in name.lower() or "_loc" in name.lower():
                loc_params.append(param)
                self.logger.debug(f"  STN param: {name}")
            else:
                other_params.append(param)

        # Criar grupos de parâmetros com LRs diferentes
        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": loc_params, "lr": base_lr * loc_lr_scale},
        ]

        if optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(
                param_groups,
                lr=base_lr,
                alpha=opt_config["alpha"],
                eps=opt_config["eps"],
                weight_decay=opt_config["weight_decay"],
                momentum=opt_config["momentum"],
            )
        else:  # adam
            optimizer = optim.Adam(
                param_groups,
                lr=base_lr,
                betas=(opt_config["beta1"], opt_config["beta2"]),
                eps=opt_config["eps"],
                weight_decay=opt_config["weight_decay"],
            )

        self.logger.info(f"Otimizador: {optimizer_type.upper()}")
        self.logger.info(f"  Base LR: {base_lr}")
        self.logger.info(f"  STN LR: {base_lr * loc_lr_scale:.6f} ({loc_lr_scale*100:.1f}% do base)")
        self.logger.info(f"  Weight decay: {opt_config['weight_decay']}")
        self.logger.info(f"  STN params: {len(loc_params)}, Other params: {len(other_params)}")

        return optimizer

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
        """Treinar uma época"""
        # CRÍTICO: Forçar training mode em TUDO no início de cada época
        self.model.train()
        if self.criterion_center_texture is not None:
            self.criterion_center_texture.train()
        if self.criterion_center_minutia is not None:
            self.criterion_center_minutia.train()
        
        # Verificar requires_grad novamente (pode ter sido alterado)
        for param in self.model.parameters():
            if not param.requires_grad:
                param.requires_grad = True
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=self.mode != "debug")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Desempacotar batch (pode ter paths se for dataset customizado)
            if len(batch_data) == 3:
                images, labels, image_paths = batch_data
            else:
                images, labels = batch_data
                image_paths = None
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Carregar minutia maps se disponíveis
            minutia_maps = None
            minutia_map_weights = None
            if image_paths is not None:
                try:
                    minutia_maps, minutia_map_weights = batch_load_minutia_maps(image_paths)
                    minutia_maps = minutia_maps.to(self.device)
                    minutia_map_weights = minutia_map_weights.to(self.device)
                    
                    # Log no primeiro batch
                    if batch_idx == 0 and epoch == 1:
                        num_with_minutiae = (minutia_map_weights > 0).sum().item()
                        self.logger.info(f"Minutia maps carregados: {num_with_minutiae}/{len(image_paths)} amostras com minutiae")
                except Exception as e:
                    # Log de erro para diagnóstico
                    if batch_idx == 0:
                        self.logger.warning(f"Erro ao carregar minutia maps: {e}")
                    pass
            
            self.optimizer.zero_grad()
            
            # Forward pass com mixed precision
            if self.scaler:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, labels, minutia_maps, minutia_map_weights)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # REMOVIDO: Gradient clipping NÃO é usado no paper original
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass simples
                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels, minutia_maps, minutia_map_weights)
                
                # Backward pass
                loss.backward()

                # REMOVIDO: Gradient clipping NÃO é usado no paper original
                # Era adicionado para estabilizar STN, mas paper não usa
                # STN agora usa LR 3.5% menor, o que deve estabilizar
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

        CORRIGIDO: Calcula loss completa (CrossEntropy + Minutia Map Loss).
        Center Loss desabilitado (training_mode=False) para open-set.
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=self.mode != "debug")

            for batch_data in pbar:
                # Desempacotar batch (pode ter paths se for dataset customizado)
                if len(batch_data) == 3:
                    images, labels, image_paths = batch_data
                else:
                    images, labels = batch_data
                    image_paths = None

                images = images.to(self.device)
                labels = labels.to(self.device)

                # CRÍTICO: Carregar minutia maps para calcular loss corretamente!
                minutia_maps = None
                minutia_map_weights = None
                if image_paths is not None:
                    try:
                        minutia_maps, minutia_map_weights = batch_load_minutia_maps(image_paths)
                        minutia_maps = minutia_maps.to(self.device)
                        minutia_map_weights = minutia_map_weights.to(self.device)
                    except Exception as e:
                        self.logger.warning(f"Erro ao carregar minutia maps na validação: {e}")

                outputs = self.model(images)

                # training_mode=False desabilita Center Loss (open-set)
                # MAS calcula CrossEntropy + Minutia Map Loss!
                batch_loss = self._compute_loss(
                    outputs,
                    labels,
                    minutia_maps=minutia_maps,
                    minutia_map_weights=minutia_map_weights,
                    training_mode=False
                )

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
    
    def _compute_loss(
        self,
        outputs: Dict,
        labels: torch.Tensor,
        minutia_maps: torch.Tensor = None,
        minutia_map_weights: torch.Tensor = None,
        training_mode: bool = True
    ) -> torch.Tensor:
        """Computar perda total para DeepPrint (LocTexMinu)

        Args:
            training_mode: Se False, desabilita Center/ArcFace Loss (para validação open-set)

        Suporta dois modos:
        1. CENTER LOSS (original): Softmax + Center Loss por branch
        2. ARCFACE LOSS: ArcFace no embedding combinado final
        """
        total_loss = torch.tensor(0.0, device=self.device)

        # Obter loss type (center ou arcface)
        loss_type = getattr(self, 'loss_type', 'center')

        if loss_type == "arcface" and training_mode:
            # ARCFACE MODE: Aplicar ArcFace no embedding combinado final
            # ArcFace já inclui classificação (softmax interno), então não precisa de CrossEntropy separado

            if "embedding" in outputs:
                # Embedding final combinado (texture + minutiae = 192-dim para baseline)
                embedding = outputs["embedding"]

                # ArcFace loss (inclui softmax interno)
                arcface_loss_dict = self.arcface_criterion(
                    features=embedding,
                    logits=None,  # Não usado (ArcFace calcula internamente)
                    labels=labels
                )
                total_loss = total_loss + arcface_loss_dict['total_loss']

            # Minutia Map Loss (ainda supervisionado, mesmo com ArcFace)
            if "minutia_maps" in outputs and minutia_maps is not None:
                predicted_maps = outputs["minutia_maps"]
                mm_squared_diff = (predicted_maps - minutia_maps) ** 2
                mm_mse = mm_squared_diff.reshape(minutia_map_weights.shape[0], -1).mean(dim=1)
                mm_loss = (mm_mse * minutia_map_weights).mean()
                total_loss = total_loss + LOSS_CONFIG["minutia_map_loss_weight"] * mm_loss

        else:
            # CENTER LOSS MODE (original DeepPrint)
            # Texture branch loss
            if "texture_logits" in outputs and "texture_embedding" in outputs:
                texture_logits = outputs["texture_logits"]
                texture_embedding = outputs["texture_embedding"]

                # CrossEntropy
                ce_loss_texture = F.cross_entropy(texture_logits, labels)
                total_loss = total_loss + LOSS_CONFIG["softmax_loss_weight"] * ce_loss_texture

                # Center Loss (apenas em treino, não em validação open-set)
                if training_mode and self.criterion_center_texture is not None:
                    center_loss_texture = self.criterion_center_texture(texture_embedding, labels)
                    total_loss = total_loss + self.center_loss_weight * center_loss_texture

            # Minutiae branch loss
            if "minutia_logits" in outputs and "minutia_embedding" in outputs:
                minutia_logits = outputs["minutia_logits"]
                minutia_embedding = outputs["minutia_embedding"]

                # CrossEntropy
                ce_loss_minutia = F.cross_entropy(minutia_logits, labels)
                total_loss = total_loss + LOSS_CONFIG["softmax_loss_weight"] * ce_loss_minutia

                # Center Loss (apenas em treino, não em validação open-set)
                if training_mode and self.criterion_center_minutia is not None:
                    center_loss_minutia = self.criterion_center_minutia(minutia_embedding, labels)
                    total_loss = total_loss + self.center_loss_weight * center_loss_minutia

                # Minutia Map Loss (supervisionado)
                if "minutia_maps" in outputs and minutia_maps is not None:
                    predicted_maps = outputs["minutia_maps"]
                    mm_squared_diff = (predicted_maps - minutia_maps) ** 2
                    mm_mse = mm_squared_diff.reshape(minutia_map_weights.shape[0], -1).mean(dim=1)
                    mm_loss = (mm_mse * minutia_map_weights).mean()
                    total_loss = total_loss + LOSS_CONFIG["minutia_map_loss_weight"] * mm_loss

            # Fallback para modelos texture-only (exp1-exp3)
            if "logits" in outputs and "embedding" in outputs:
                if "texture_logits" not in outputs:  # Só se não for LocTexMinu
                    logits = outputs["logits"]
                    embedding = outputs["embedding"]

                    ce_loss = F.cross_entropy(logits, labels)
                    total_loss = total_loss + LOSS_CONFIG["softmax_loss_weight"] * ce_loss

                    if training_mode and self.criterion_center_texture is not None:
                        center_loss = self.criterion_center_texture(embedding, labels)
                        total_loss = total_loss + self.center_loss_weight * center_loss

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
        need_recreate_optimizer = False
        if resume:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path:
                start_epoch, optimizer_loaded = self.load_checkpoint(checkpoint_path)
                # Restaurar best_val_loss do histórico
                if self.history.get("val_loss"):
                    best_val_loss = min(self.history["val_loss"])
                self.logger.info(f"Retomando treinamento da época {start_epoch}")
                
                # Se optimizer não foi carregado, marcar para recriar após configurar classes
                if not optimizer_loaded:
                    need_recreate_optimizer = True
            else:
                self.logger.info("Nenhum checkpoint encontrado, iniciando do zero")
        
        # Obter número de classes do dataset
        # CRÍTICO: Sempre configurar classificadores, mesmo no resume (podem estar None após load_checkpoint)
        if num_classes is None:
            if hasattr(train_loader.dataset, 'num_classes'):
                num_classes = train_loader.dataset.num_classes
            else:
                all_labels = set()
                for _, labels in train_loader:
                    all_labels.update(labels.tolist())
                num_classes = len(all_labels)
        
        self.logger.info(f"Número total de classes (origens únicas): {num_classes}")
        
        # Configurar classificador no modelo (sempre, mesmo no resume)
        if hasattr(self.model, 'set_num_classes'):
            self.model.set_num_classes(num_classes)
            
            # Se precisar recriar optimizer (após resume com falha), recriar completamente
            if need_recreate_optimizer:
                self.logger.info("Recriando optimizer com todos os parâmetros do modelo")
                self.optimizer = self._create_optimizer()
                need_recreate_optimizer = False
            else:
                # Adicionar parâmetros dos classificadores ao optimizer (treino do zero)
                if hasattr(self.model, 'texture_classifier') and self.model.texture_classifier is not None:
                    self.optimizer.add_param_group({'params': self.model.texture_classifier.parameters()})
                if hasattr(self.model, 'minutia_classifier') and self.model.minutia_classifier is not None:
                    self.optimizer.add_param_group({'params': self.model.minutia_classifier.parameters()})
                if hasattr(self.model, 'classifier') and self.model.classifier is not None:
                    self.optimizer.add_param_group({'params': self.model.classifier.parameters()})
            
            self.logger.info(f"Classificador configurado com {num_classes} classes")
        
        # Inicializar loss function baseada no tipo configurado
        loss_type = LOSS_CONFIG.get("loss_type", "center").lower()
        self.loss_type = loss_type

        if loss_type == "center":
            # CENTER LOSS (original DeepPrint): separado por branch
            if self.criterion_center_texture is None:
                self.logger.info(f"Inicializando Center Loss (Texture) com {num_classes} classes e {self.texture_embedding_dims} dims")
                self.criterion_center_texture = CenterLoss(
                    num_classes=num_classes,
                    feat_dim=self.texture_embedding_dims,
                    alpha=0.01
                ).to(self.device)

                # Adicionar ao optimizer se necessário
                if need_recreate_optimizer or resume:
                    self.optimizer.add_param_group({'params': self.criterion_center_texture.parameters()})

            if self.criterion_center_minutia is None and hasattr(self.model, 'minutia_embedding_dims'):
                self.logger.info(f"Inicializando Center Loss (Minutiae) com {num_classes} classes e {self.minutia_embedding_dims} dims")
                self.criterion_center_minutia = CenterLoss(
                    num_classes=num_classes,
                    feat_dim=self.minutia_embedding_dims,
                    alpha=0.01
                ).to(self.device)
                self.logger.info(f"Center Loss inicializado com {num_classes} classes")

                # Adicionar ao optimizer se necessário
                if need_recreate_optimizer or resume:
                    self.optimizer.add_param_group({'params': self.criterion_center_minutia.parameters()})

        elif loss_type == "arcface":
            # ARCFACE LOSS: aplicada no embedding combinado final
            # Para DeepPrint LocTexMinu: texture (96) + minutiae (96) = 192-dim
            total_embedding_dim = self.texture_embedding_dims + self.minutia_embedding_dims

            self.logger.info("=" * 80)
            self.logger.info("ARCFACE LOSS CONFIGURADA")
            self.logger.info(f"  Número de classes: {num_classes}")
            self.logger.info(f"  Dimensão embedding: {total_embedding_dim} (texture {self.texture_embedding_dims} + minutiae {self.minutia_embedding_dims})")
            self.logger.info(f"  Angular margin (m): {LOSS_CONFIG['arcface_margin']:.3f} rad (~{LOSS_CONFIG['arcface_margin'] * 57.3:.1f}°)")
            self.logger.info(f"  Feature scale (s): {LOSS_CONFIG['arcface_scale']:.1f}")
            self.logger.info(f"  Easy margin: {LOSS_CONFIG['arcface_easy_margin']}")
            self.logger.info("=" * 80)

            # Criar loss combinada usando factory function
            self.arcface_criterion = get_loss_function(
                loss_type="arcface",
                num_classes=num_classes,
                feat_dim=total_embedding_dim,
                arcface_margin=LOSS_CONFIG["arcface_margin"],
                arcface_scale=LOSS_CONFIG["arcface_scale"],
                device=self.device
            )

            # Adicionar parâmetros da ArcFace ao optimizer (weight matrix)
            if need_recreate_optimizer or resume:
                self.optimizer.add_param_group({'params': self.arcface_criterion.parameters()})

        # Calcular peso ADAPTATIVO do Center Loss baseado no número de classes
        self.center_loss_weight = get_center_loss_weight(num_classes)
        if LOSS_CONFIG["center_loss_use_adaptive"]:
            self.logger.info("=" * 80)
            self.logger.info("CENTER LOSS ADAPTATIVO")
            self.logger.info(f"  Número de classes: {num_classes}")
            self.logger.info(f"  Peso base (paper, 6000 classes): {LOSS_CONFIG['center_loss_base_weight']:.6f}")
            self.logger.info(f"  Peso adaptativo: {self.center_loss_weight:.8f}")
            self.logger.info(f"  Fator de escala: {self.center_loss_weight / LOSS_CONFIG['center_loss_base_weight']:.4f}x")
            self.logger.info("=" * 80)
        else:
            self.logger.info(f"Center Loss peso fixo: {self.center_loss_weight:.6f}")
        
        # CRÍTICO: Garantir que modelo esteja em training mode após todas as modificações
        self.model.train()
        if self.criterion_center_texture is not None:
            self.criterion_center_texture.train()
        if self.criterion_center_minutia is not None:
            self.criterion_center_minutia.train()
        
        # Verificar que todos os parâmetros têm requires_grad
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                self.logger.warning(f"Parâmetro {name} NÃO tem requires_grad=True! Corrigindo...")
                param.requires_grad = True
        
        self.logger.info("Modelo configurado e pronto para treinamento")
        
        # CORREÇÃO: Salvar melhor modelo baseado em EER, não val_loss
        # Val_loss otimiza classificação, mas EER mede embeddings discriminativos
        best_eer = float('inf')

        for epoch in range(start_epoch, num_epochs + 1):
            # Treinar
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validar
            val_metrics = self.validate(val_loader, epoch)

            # Calcular EER TODA época (não só cada 5)
            eer_result = self._compute_quick_eer(val_loader)
            current_eer = eer_result['eer'] if eer_result is not None else float('inf')

            # Retornar modelo para modo de treinamento após EER
            self.model.train()

            # Limpar memória após cálculo de EER
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Atualizar histórico
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            if "val_eer" not in self.history:
                self.history["val_eer"] = []
            self.history["val_eer"].append(current_eer)

            # Salvar melhor modelo baseado em EER (não loss!)
            if current_eer < best_eer:
                best_eer = current_eer
                self._save_checkpoint(epoch, val_metrics["loss"], is_best=True)
                self.logger.info(f"✅ Melhor modelo salvo: época {epoch}, EER={current_eer:.4f}")

            # Log
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - EER: {current_eer:.4f}"
            )

            # Salvar checkpoint periódico (a cada 10 épocas)
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_metrics["loss"], is_best=False)
                self.logger.info(f"Checkpoint salvo: época {epoch}")
        
        self.logger.info("Treinamento concluído")
        self._save_history()
    
    def _compute_quick_eer(self, val_loader: DataLoader, max_samples: int = 500) -> Optional[Dict]:
        """Calcular EER rapidamente com amostra do val_loader
        
        Estratégia: coletar múltiplas amostras por classe para ter pares genuínos
        
        Args:
            val_loader: DataLoader de validação
            max_samples: Número máximo de amostras para usar
        
        Returns:
            Dict com EER e outras métricas, ou None se erro
        """
        try:
            self.model.eval()
            embeddings_by_label = {}  # Agrupar por label para ter genuínos
            
            with torch.no_grad():
                total_samples = 0
                for batch_idx, batch_data in enumerate(val_loader):
                    if total_samples >= max_samples:
                        break
                    
                    # Desempacotar batch
                    if len(batch_data) == 3:
                        images, batch_labels, _ = batch_data
                    else:
                        images, batch_labels = batch_data
                    
                    images = images.to(self.device)
                    outputs = self.model(images)
                    embedding = outputs["embedding"]
                    
                    batch_embeddings = embedding.cpu().numpy()
                    batch_labels_np = batch_labels.numpy()
                    
                    # Agrupar por label
                    for emb, lbl in zip(batch_embeddings, batch_labels_np):
                        if lbl not in embeddings_by_label:
                            embeddings_by_label[lbl] = []
                        embeddings_by_label[lbl].append(emb)
                        total_samples += 1
                        if total_samples >= max_samples:
                            break
                    
                    # Liberar memória GPU a cada batch
                    del images, outputs, embedding
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            if len(embeddings_by_label) < 2:
                return None
            
            # Criar pares genuínos e impostores balanceados
            genuine_scores = []
            impostor_scores = []
            
            labels_list = list(embeddings_by_label.keys())
            
            # Pares genuínos: mesma classe
            for label, embs in embeddings_by_label.items():
                embs = np.array(embs)
                if len(embs) < 2:
                    continue
                
                # Normalizar embeddings
                embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
                
                # Criar até 10 pares genuínos por classe
                n_pairs = min(10, len(embs) * (len(embs) - 1) // 2)
                for _ in range(n_pairs):
                    i, j = np.random.choice(len(embs), size=2, replace=False)
                    score = np.dot(embs[i], embs[j])
                    genuine_scores.append(score)
            
            # Pares impostores: classes diferentes
            max_impostor_pairs = len(genuine_scores) * 3  # 3x mais impostores
            for _ in range(max_impostor_pairs):
                # Escolher 2 classes diferentes
                if len(labels_list) < 2:
                    break
                lbl1, lbl2 = np.random.choice(labels_list, size=2, replace=False)
                
                embs1 = np.array(embeddings_by_label[lbl1])
                embs2 = np.array(embeddings_by_label[lbl2])
                
                # Normalizar
                embs1 = embs1 / (np.linalg.norm(embs1, axis=1, keepdims=True) + 1e-8)
                embs2 = embs2 / (np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-8)
                
                # Escolher 1 amostra de cada
                i = np.random.choice(len(embs1))
                j = np.random.choice(len(embs2))
                
                score = np.dot(embs1[i], embs2[j])
                impostor_scores.append(score)
            
            if len(genuine_scores) == 0 or len(impostor_scores) == 0:
                self.logger.warning(f"EER: Pares insuficientes (genuínos={len(genuine_scores)}, impostores={len(impostor_scores)})")
                return None
            
            genuine_scores = np.array(genuine_scores)
            impostor_scores = np.array(impostor_scores)
            
            # DEBUG: Estatísticas dos scores
            self.logger.info(f"  DEBUG - Pares genuínos: {len(genuine_scores)}, impostores: {len(impostor_scores)}")
            self.logger.info(f"  DEBUG - Genuínos: min={genuine_scores.min():.4f}, max={genuine_scores.max():.4f}, mean={genuine_scores.mean():.4f}, std={genuine_scores.std():.4f}")
            self.logger.info(f"  DEBUG - Impostores: min={impostor_scores.min():.4f}, max={impostor_scores.max():.4f}, mean={impostor_scores.mean():.4f}, std={impostor_scores.std():.4f}")
            
            # Calcular EER: ponto onde FAR = FRR
            all_scores = np.concatenate([genuine_scores, impostor_scores])
            thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)
            
            best_eer = 1.0
            best_threshold = 0.0
            best_diff = 1.0
            far_at_frr_01 = 1.0
            
            # DEBUG: Mostrar alguns pontos do threshold
            debug_thresholds = [thresholds[0], thresholds[len(thresholds)//2], thresholds[-1]]
            
            for idx, threshold in enumerate(thresholds):
                # False Accept Rate (impostor aceito como genuíno)
                far = np.sum(impostor_scores >= threshold) / len(impostor_scores)
                # False Reject Rate (genuíno rejeitado como impostor)
                frr = np.sum(genuine_scores < threshold) / len(genuine_scores)
                
                # DEBUG: Mostrar alguns valores
                if threshold in debug_thresholds or idx == 0:
                    self.logger.info(f"  DEBUG - Threshold={threshold:.4f}: FAR={far:.4f}, FRR={frr:.4f}, diff={abs(far-frr):.4f}")
                
                # EER é onde FAR ≈ FRR (mínima diferença)
                diff = abs(far - frr)
                if diff < best_diff:
                    best_diff = diff
                    best_eer = (far + frr) / 2
                    best_threshold = threshold
                
                # FAR quando FRR = 0.1
                if abs(frr - 0.1) < 0.02:
                    far_at_frr_01 = far
            
            # DEBUG: Resultado final
            self.logger.info(f"  DEBUG - RESULTADO: EER={best_eer:.4f}, threshold={best_threshold:.4f}, diff={best_diff:.4f}")

            result = {
                "eer": best_eer,
                "threshold": best_threshold,
                "far_at_frr_01": far_at_frr_01,
                "num_genuine": len(genuine_scores),
                "num_impostor": len(impostor_scores),
                "num_classes": len(embeddings_by_label),
            }

            # Limpar memória antes de retornar
            del embeddings_by_label, genuine_scores, impostor_scores, all_scores, thresholds
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            self.logger.warning(f"Erro ao calcular EER: {e}")
            # Limpar memória em caso de erro também
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return None
    
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
    
    def load_checkpoint(self, checkpoint_path: Path) -> tuple[int, bool]:
        """Carregar checkpoint e retornar a época para retomada
        
        Returns:
            (start_epoch, optimizer_loaded): época para iniciar e se optimizer foi carregado
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # strict=False: ignora classificadores (são recriados via set_num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Tentar carregar optimizer state (pode falhar se parameter groups mudaram)
        optimizer_loaded = False
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_loaded = True
        except ValueError as e:
            self.logger.warning(f"Não foi possível carregar optimizer state (parâmetros diferentes): {e}")
            self.logger.warning("Optimizer será recriado após configurar classes para incluir todos os parâmetros")
        
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []})
        
        # Restaurar CenterLoss para texture branch
        if checkpoint.get("center_loss_texture_state") and checkpoint.get("num_classes"):
            self.criterion_center_texture = CenterLoss(
                num_classes=checkpoint["num_classes"],
                feat_dim=self.texture_embedding_dims,
                alpha=0.01
            ).to(self.device)
            self.criterion_center_texture.load_state_dict(checkpoint["center_loss_texture_state"])
            # CRÍTICO: Garantir requires_grad após load_state_dict
            self.criterion_center_texture.train()
            for param in self.criterion_center_texture.parameters():
                param.requires_grad = True
        
        # Restaurar CenterLoss para minutiae branch
        if checkpoint.get("center_loss_minutia_state") and checkpoint.get("num_classes"):
            self.criterion_center_minutia = CenterLoss(
                num_classes=checkpoint["num_classes"],
                feat_dim=self.minutia_embedding_dims,
                alpha=0.01
            ).to(self.device)
            self.criterion_center_minutia.load_state_dict(checkpoint["center_loss_minutia_state"])
            # CRÍTICO: Garantir requires_grad após load_state_dict
            self.criterion_center_minutia.train()
            for param in self.criterion_center_minutia.parameters():
                param.requires_grad = True
        
        # CRÍTICO: Garantir que TODOS os parâmetros do modelo tenham requires_grad após load
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        start_epoch = checkpoint.get("epoch", 0) + 1
        self.logger.info(f"Checkpoint carregado de {checkpoint_path}, retomando da época {start_epoch}")
        return start_epoch, optimizer_loaded
    
    def load_best_model(self):
        """Carregar melhor modelo para avaliação (sem treinar)
        
        Configura o modelo com o best_model.pt e inicializa Center Loss.
        """
        best_model_path = self.experiment_dir / "models" / "best_model.pt"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model não encontrado em {best_model_path}")
        
        self.logger.info(f"Carregando best_model de {best_model_path} para avaliação")
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        
        # Carregar model state
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Configurar número de classes e Center Loss
        if checkpoint.get("num_classes"):
            num_classes = checkpoint["num_classes"]
            self.logger.info(f"Configurando modelo com {num_classes} classes")
            
            if hasattr(self.model, 'set_num_classes'):
                self.model.set_num_classes(num_classes)
            
            # Restaurar Center Loss
            if checkpoint.get("center_loss_texture_state"):
                self.criterion_center_texture = CenterLoss(
                    num_classes=num_classes,
                    feat_dim=self.texture_embedding_dims,
                    alpha=0.01
                ).to(self.device)
                self.criterion_center_texture.load_state_dict(checkpoint["center_loss_texture_state"])
            
            if checkpoint.get("center_loss_minutia_state"):
                self.criterion_center_minutia = CenterLoss(
                    num_classes=num_classes,
                    feat_dim=self.minutia_embedding_dims,
                    alpha=0.01
                ).to(self.device)
                self.criterion_center_minutia.load_state_dict(checkpoint["center_loss_minutia_state"])
        
        self.model.eval()
        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("val_loss", "?")
        self.logger.info(f"Best model carregado (época {epoch}, val_loss={val_loss})")
    
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
