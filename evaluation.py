"""
Módulo de avaliação para comparação de impressões digitais.

Realiza comparações todas-contra-todas, monitora recursos do sistema,
e gera CSV com resultados incluindo qualidade NFIQ2.
"""

import os
import time
import psutil
import threading
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import logging


@dataclass
class ResourceMonitor:
    """Monitora uso de recursos durante a execução."""
    
    interval: float = 0.5  # segundos entre medições
    _running: bool = False
    _thread: Optional[threading.Thread] = None
    _measurements: List[Dict] = field(default_factory=list)
    _start_time: float = 0.0
    _end_time: float = 0.0
    
    def start(self):
        """Inicia o monitoramento."""
        self._running = True
        self._measurements = []
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict:
        """Para o monitoramento e retorna estatísticas."""
        self._running = False
        self._end_time = time.time()
        if self._thread:
            self._thread.join(timeout=2.0)
        
        return self.get_statistics()
    
    def _monitor_loop(self):
        """Loop de monitoramento em thread separada."""
        process = psutil.Process()
        
        while self._running:
            try:
                mem_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                # GPU memory se disponível
                gpu_memory = 0
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                
                self._measurements.append({
                    "timestamp": time.time() - self._start_time,
                    "memory_rss_mb": mem_info.rss / (1024**2),
                    "memory_vms_mb": mem_info.vms / (1024**2),
                    "cpu_percent": cpu_percent,
                    "gpu_memory_mb": gpu_memory,
                })
            except Exception:
                pass
            
            time.sleep(self.interval)
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas agregadas do monitoramento."""
        if not self._measurements:
            return {
                "total_time_seconds": self._end_time - self._start_time,
                "memory_peak_mb": 0,
                "memory_avg_mb": 0,
                "cpu_avg_percent": 0,
                "gpu_memory_peak_mb": 0,
            }
        
        memories = [m["memory_rss_mb"] for m in self._measurements]
        cpus = [m["cpu_percent"] for m in self._measurements]
        gpu_memories = [m["gpu_memory_mb"] for m in self._measurements]
        
        return {
            "total_time_seconds": self._end_time - self._start_time,
            "memory_peak_mb": max(memories),
            "memory_avg_mb": np.mean(memories),
            "cpu_avg_percent": np.mean(cpus),
            "gpu_memory_peak_mb": max(gpu_memories) if gpu_memories else 0,
            "num_measurements": len(self._measurements),
        }


class QualityLoader:
    """Carrega e gerencia valores de qualidade NFIQ2 das imagens."""
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: caminho base das bases de dados
        """
        self.base_path = Path(base_path)
        self.quality_data: Dict[str, float] = {}
        self.mean_quality: float = 0.0
        self._loaded = False
    
    def load_quality_files(self, datasets: List[str] = None):
        """
        Carrega arquivos de qualidade NFIQ2.
        
        Args:
            datasets: lista de datasets para carregar (ex: ["FVC2000", "FVC2002"])
        """
        if datasets is None:
            datasets = ["FVC2000", "FVC2002", "FVC2004"]
        
        all_qualities = []
        
        for dataset in datasets:
            quality_file = self.base_path / dataset / f"{dataset}_nfiq2_quality.csv"
            
            if not quality_file.exists():
                continue
            
            try:
                df = pd.read_csv(quality_file)
                
                for _, row in df.iterrows():
                    filepath = str(row.get("Filepath", ""))
                    quality = row.get("Quality_Score")
                    error = row.get("Error")
                    
                    # Extrair chave do arquivo (FVC20XX/DB*_B/NNN_N.png)
                    file_key = self._extract_filename_key(filepath)
                    
                    # Verificar se quality é válido e error está vazio ou é NaN
                    error_is_empty = pd.isna(error) or str(error).strip() == ""
                    
                    if file_key and pd.notna(quality) and error_is_empty:
                        self.quality_data[file_key] = float(quality)
                        all_qualities.append(float(quality))
                        
            except Exception as e:
                print(f"Erro ao carregar {quality_file}: {e}")
        
        # Calcular média para usar como fallback
        if all_qualities:
            self.mean_quality = np.mean(all_qualities)
        else:
            self.mean_quality = 50.0  # valor padrão
        
        self._loaded = True
    
    def _extract_relative_path(self, filepath: str, dataset: str) -> Optional[str]:
        """Extrai caminho relativo a partir do nome do dataset."""
        if not filepath:
            return None
        
        # Encontrar posição do dataset no caminho
        idx = filepath.find(f"/{dataset}/")
        if idx >= 0:
            return filepath[idx + 1:]  # Remove a barra inicial
        
        # Tentar sem barra inicial
        idx = filepath.find(f"{dataset}/")
        if idx >= 0:
            return filepath[idx:]
        
        return None
    
    def _extract_filename_key(self, filepath: str) -> str:
        """Extrai chave baseada em dataset/subdir/filename."""
        # Tentar extrair padrão: FVC20XX/DB*_B/NNN_N.png
        import re
        match = re.search(r'(FVC\d{4}/DB\d_B/\d+_\d+\.\w+)$', filepath)
        if match:
            return match.group(1)
        
        # Fallback: apenas nome do arquivo
        return Path(filepath).name
    
    def get_quality(self, filepath: str) -> float:
        """
        Retorna qualidade NFIQ2 para um arquivo.
        
        Args:
            filepath: caminho do arquivo (pode ser absoluto ou relativo)
        
        Returns:
            valor de qualidade (ou média se não encontrado)
        """
        if not self._loaded:
            return self.mean_quality
        
        # Extrair chave do arquivo
        file_key = self._extract_filename_key(filepath)
        
        # Buscar pela chave exata
        if file_key in self.quality_data:
            return self.quality_data[file_key]
        
        # Tentar pelo nome do arquivo apenas (fallback)
        filename = Path(filepath).name
        for key, quality in self.quality_data.items():
            if key.endswith(filename):
                return quality
        
        return self.mean_quality


class PairwiseEvaluator:
    """
    Avaliador que realiza comparações todas-contra-todas.
    
    Gera CSV com colunas:
    - Arquivo_A: nome do primeiro arquivo
    - Quali_A: qualidade NFIQ2 do primeiro arquivo
    - Arquivo_B: nome do segundo arquivo
    - Quali_B: qualidade NFIQ2 do segundo arquivo
    - mesma_fonte: 1 se mesma origem, 0 caso contrário
    - mesmo_arquivo: 1 se é o mesmo arquivo, 0 caso contrário
    - score: pontuação de similaridade
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        quality_loader: QualityLoader,
        device: torch.device = None,
        logger: logging.Logger = None,
    ):
        """
        Args:
            model: modelo para extrair embeddings
            quality_loader: carregador de qualidades NFIQ2
            device: dispositivo (CPU/GPU)
            logger: logger para mensagens
        """
        self.model = model
        self.quality_loader = quality_loader
        self.device = device or torch.device("cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        self.model.eval()
        self.model.to(self.device)
    
    def extract_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        file_paths: List[str],
        labels: List[int],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Extrai embeddings para todas as imagens.
        
        Args:
            dataloader: DataLoader com as imagens
            file_paths: lista de caminhos dos arquivos
            labels: lista de labels (origem)
        
        Returns:
            embeddings, file_paths, labels
        """
        embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Extraindo embeddings"):
                images = images.to(self.device)
                outputs = self.model(images)
                embedding = outputs["embedding"]
                embeddings.append(embedding.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        # Normalizar embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        return embeddings, file_paths, labels
    
    def compute_pairwise_comparisons(
        self,
        embeddings: np.ndarray,
        file_paths: List[str],
        labels: List[int],
        output_file: str,
        batch_size: int = 10000,
    ) -> Dict:
        """
        Realiza comparações todas-contra-todas e salva em CSV.
        
        Compara cada par apenas uma vez (i <= j), incluindo arquivo consigo mesmo.
        Total de comparações: N*(N+1)/2
        
        Args:
            embeddings: matriz de embeddings (N x D)
            file_paths: lista de caminhos dos arquivos
            labels: lista de labels (origem)
            output_file: caminho do arquivo CSV de saída
            batch_size: tamanho do lote para escrita
        
        Returns:
            estatísticas das comparações
        """
        n = len(embeddings)
        total_comparisons = n * (n + 1) // 2
        
        self.logger.info(f"Iniciando {total_comparisons} comparações ({n} amostras)")
        
        # Iniciar monitoramento de recursos
        monitor = ResourceMonitor(interval=0.5)
        monitor.start()
        
        # Preparar arquivo CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Estatísticas
        num_genuine = 0
        num_impostor = 0
        num_same_file = 0
        genuine_scores = []
        impostor_scores = []
        
        # Escrever em lotes
        rows = []
        
        with open(output_path, "w") as f:
            # Cabeçalho
            f.write("Arquivo_A,Quali_A,Arquivo_B,Quali_B,mesma_fonte,mesmo_arquivo,score\n")
            
            comparison_idx = 0
            for i in tqdm(range(n), desc="Comparações"):
                for j in range(i, n):  # i <= j para evitar duplicatas
                    # Calcular score (similaridade coseno)
                    score = float(np.dot(embeddings[i], embeddings[j]))
                    
                    # Determinar se é mesma fonte
                    same_source = 1 if labels[i] == labels[j] else 0
                    
                    # Determinar se é mesmo arquivo
                    same_file = 1 if i == j else 0
                    
                    # Obter qualidades
                    quali_a = self.quality_loader.get_quality(file_paths[i])
                    quali_b = self.quality_loader.get_quality(file_paths[j])
                    
                    # Extrair apenas nome do arquivo
                    file_a = Path(file_paths[i]).name
                    file_b = Path(file_paths[j]).name
                    
                    rows.append(
                        f"{file_a},{quali_a:.1f},{file_b},{quali_b:.1f},"
                        f"{same_source},{same_file},{score:.6f}\n"
                    )
                    
                    # Estatísticas (excluindo comparações consigo mesmo)
                    if same_file:
                        num_same_file += 1
                    elif same_source:
                        num_genuine += 1
                        genuine_scores.append(score)
                    else:
                        num_impostor += 1
                        impostor_scores.append(score)
                    
                    comparison_idx += 1
                    
                    # Escrever em lotes
                    if len(rows) >= batch_size:
                        f.writelines(rows)
                        rows = []
            
            # Escrever linhas restantes
            if rows:
                f.writelines(rows)
        
        # Parar monitoramento
        resource_stats = monitor.stop()
        
        # Calcular estatísticas finais
        stats = {
            "total_comparisons": total_comparisons,
            "num_samples": n,
            "num_genuine_pairs": num_genuine,
            "num_impostor_pairs": num_impostor,
            "num_same_file": num_same_file,
            "output_file": str(output_path),
            **resource_stats,
        }
        
        if genuine_scores:
            stats["genuine_score_mean"] = float(np.mean(genuine_scores))
            stats["genuine_score_std"] = float(np.std(genuine_scores))
        
        if impostor_scores:
            stats["impostor_score_mean"] = float(np.mean(impostor_scores))
            stats["impostor_score_std"] = float(np.std(impostor_scores))
        
        self.logger.info(f"Comparações concluídas em {resource_stats['total_time_seconds']:.2f}s")
        self.logger.info(f"Pares genuínos: {num_genuine}, Pares impostores: {num_impostor}")
        self.logger.info(f"Memória pico: {resource_stats['memory_peak_mb']:.1f} MB")
        self.logger.info(f"Resultados salvos em {output_path}")
        
        return stats


def run_pairwise_evaluation(
    model: torch.nn.Module,
    dataset,
    file_paths: List[str],
    labels: List[int],
    output_dir: str,
    experiment_name: str,
    mode: str = "debug",
    base_path: str = "/home/adelino/MegaSync/Forense/Papiloscopia/Compara_Metodos_Automaticos/Bases_de_Dados",
    batch_size: int = 32,
    device: torch.device = None,
    logger: logging.Logger = None,
) -> Dict:
    """
    Executa avaliação pairwise completa.
    
    Args:
        model: modelo treinado
        dataset: dataset com as imagens
        file_paths: lista de caminhos dos arquivos
        labels: lista de labels
        output_dir: diretório de saída
        experiment_name: nome do experimento
        mode: "debug" ou "prod"
        base_path: caminho base das bases de dados
        batch_size: tamanho do lote
        device: dispositivo
        logger: logger
    
    Returns:
        estatísticas da avaliação
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar qualidades NFIQ2
    logger.info("Carregando qualidades NFIQ2...")
    quality_loader = QualityLoader(base_path)
    quality_loader.load_quality_files(["FVC2000", "FVC2002", "FVC2004"])
    logger.info(f"Qualidades carregadas: {len(quality_loader.quality_data)} arquivos")
    logger.info(f"Qualidade média: {quality_loader.mean_quality:.1f}")
    
    # Criar DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Criar avaliador
    evaluator = PairwiseEvaluator(
        model=model,
        quality_loader=quality_loader,
        device=device,
        logger=logger,
    )
    
    # Extrair embeddings
    logger.info("Extraindo embeddings...")
    embeddings, file_paths, labels = evaluator.extract_embeddings(
        dataloader, file_paths, labels
    )
    logger.info(f"Embeddings extraídos: {embeddings.shape}")
    
    # Realizar comparações
    output_file = Path(output_dir) / "results" / f"pairwise_comparisons_{mode}.csv"
    
    logger.info("Realizando comparações todas-contra-todas...")
    stats = evaluator.compute_pairwise_comparisons(
        embeddings=embeddings,
        file_paths=file_paths,
        labels=labels,
        output_file=str(output_file),
    )
    
    # Salvar estatísticas
    import json
    stats_file = Path(output_dir) / "results" / f"pairwise_stats_{mode}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Estatísticas salvas em {stats_file}")
    
    return stats
