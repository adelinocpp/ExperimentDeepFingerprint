"""
Módulo de validação com validação cruzada e cálculo de intervalos de confiança
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats
import pickle

from config import CROSS_VALIDATION_CONFIG, METRICS_CONFIG


class PerformanceMetrics:
    """Classe para calcular métricas de desempenho"""
    
    def __init__(self, rank_k: List[int] = None, confidence_level: float = 0.95):
        self.rank_k = rank_k or METRICS_CONFIG["rank_k"]
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def compute_cmc(
        self,
        embeddings_query: np.ndarray,
        embeddings_gallery: np.ndarray,
        labels_query: np.ndarray,
        labels_gallery: np.ndarray,
        one_sample_per_class: bool = True,
        use_score_normalization: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Computar CMC (Cumulative Match Characteristic) com melhorias do paper original.
        
        Melhorias implementadas:
        1. L2 normalization dos embeddings
        2. Score normalization (z-score) para melhorar discriminação
        3. Gallery selection otimizada (melhor amostra por classe)
        
        Protocolo seguindo paper original (BIOSIG 2023):
        - Galeria: 1 amostra por classe (seleção otimizada)
        - Probes: outras amostras das mesmas classes
        
        Args:
            embeddings_query: (N_query, D) array de embeddings de query
            embeddings_gallery: (N_gallery, D) array de embeddings de galeria
            labels_query: (N_query,) array de labels de query
            labels_gallery: (N_gallery,) array de labels de galeria
            one_sample_per_class: Se True, usa apenas 1 amostra por classe na galeria
            use_score_normalization: Se True, aplica normalização de scores
        
        Returns:
            cmc: (max_rank,) array com CMC
            metrics: dicionário com métricas adicionais
        """
        # Criar galeria otimizada (melhor amostra por classe)
        if one_sample_per_class:
            unique_labels = np.unique(labels_gallery)
            gallery_indices = []
            
            # Selecionar amostra com maior norma (maior confiança) por classe
            for label in unique_labels:
                label_indices = np.where(labels_gallery == label)[0]
                # Selecionar amostra com maior norma L2 (mais confiante)
                norms = np.linalg.norm(embeddings_gallery[label_indices], axis=1)
                best_idx = label_indices[np.argmax(norms)]
                gallery_indices.append(best_idx)
            
            gallery_indices = np.array(gallery_indices)
            embeddings_gallery = embeddings_gallery[gallery_indices]
            labels_gallery = labels_gallery[gallery_indices]
        
        # Normalizar embeddings L2 (paper original)
        embeddings_query = embeddings_query / (np.linalg.norm(embeddings_query, axis=1, keepdims=True) + 1e-8)
        embeddings_gallery = embeddings_gallery / (np.linalg.norm(embeddings_gallery, axis=1, keepdims=True) + 1e-8)
        
        # Computar matriz de similaridade (cosine similarity após L2 norm)
        similarity_matrix = np.dot(embeddings_query, embeddings_gallery.T)  # (N_query, N_gallery)
        
        # Score normalization (z-score por query) - melhora discriminação
        if use_score_normalization:
            mean_scores = similarity_matrix.mean(axis=1, keepdims=True)
            std_scores = similarity_matrix.std(axis=1, keepdims=True) + 1e-8
            similarity_matrix = (similarity_matrix - mean_scores) / std_scores
        
        # Ranking
        rankings = np.argsort(-similarity_matrix, axis=1)  # Ordem decrescente
        
        # Computar CMC
        max_rank = min(len(embeddings_gallery), max(self.rank_k) + 1)
        cmc = np.zeros(max_rank)
        
        for i in range(len(embeddings_query)):
            query_label = labels_query[i]
            ranked_labels = labels_gallery[rankings[i]]
            
            # Encontrar primeira posição onde há correspondência
            matches = (ranked_labels == query_label)
            if np.any(matches):
                first_match_rank = np.argmax(matches)
                cmc[first_match_rank:] += 1
        
        cmc = cmc / len(embeddings_query)
        
        # Métricas adicionais
        metrics = {}
        for k in self.rank_k:
            if k <= len(cmc):
                metrics[f"rank_{k}"] = float(cmc[k-1])  # rank-k está no índice k-1
        
        return cmc, metrics
    
    def compute_det(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Computar DET (Detection Error Trade-off)
        
        Args:
            genuine_scores: (N_genuine,) array de scores genuínos
            impostor_scores: (N_impostor,) array de scores impostores
        
        Returns:
            far: (N_thresholds,) array de FAR
            frr: (N_thresholds,) array de FRR
            metrics: dicionário com métricas adicionais
        """
        # Thresholds
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.sort(np.unique(all_scores))[::-1]
        
        far = np.zeros(len(thresholds))
        frr = np.zeros(len(thresholds))
        
        for i, threshold in enumerate(thresholds):
            far[i] = np.sum(impostor_scores >= threshold) / len(impostor_scores)
            frr[i] = np.sum(genuine_scores < threshold) / len(genuine_scores)
        
        # Encontrar EER (Equal Error Rate)
        eer_idx = np.argmin(np.abs(far - frr))
        eer = (far[eer_idx] + frr[eer_idx]) / 2
        
        metrics = {
            "eer": float(eer),
            "far_at_frr_0.1": float(far[np.argmin(np.abs(frr - 0.1))]),
            "far_at_frr_0.01": float(far[np.argmin(np.abs(frr - 0.01))]),
        }
        
        return far, frr, metrics
    
    def compute_confidence_intervals(
        self,
        scores: List[float],
    ) -> Tuple[float, float, float]:
        """
        Computar intervalo de confiança para uma métrica
        
        Args:
            scores: lista de scores
        
        Returns:
            mean: média
            ci_lower: limite inferior do intervalo
            ci_upper: limite superior do intervalo
        """
        scores = np.array(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        n = len(scores)
        
        # Intervalo de confiança usando t-distribution
        t_value = stats.t.ppf(1 - self.alpha / 2, n - 1)
        margin_of_error = t_value * std / np.sqrt(n)
        
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        
        return mean, ci_lower, ci_upper


class CrossValidator:
    """Classe para realizar validação cruzada"""
    
    def __init__(
        self,
        model: nn.Module,
        dataset,
        experiment_dir: Path,
        mode: str = "debug",
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = Path(experiment_dir)
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        self.n_splits = CROSS_VALIDATION_CONFIG["n_splits"]
        self.random_state = CROSS_VALIDATION_CONFIG["random_state"]
        self.stratified = CROSS_VALIDATION_CONFIG["stratified"]
        
        self.metrics_computer = PerformanceMetrics()
        self.results = {
            "fold_results": [],
            "overall_metrics": {},
        }
    
    def run_cross_validation(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        device: torch.device = torch.device("cpu"),
    ) -> Dict:
        """
        Executar validação cruzada
        
        Args:
            batch_size: tamanho do batch
            num_workers: número de workers para DataLoader
            device: dispositivo (CPU ou GPU)
        
        Returns:
            results: dicionário com resultados
        """
        # Preparar índices
        indices = np.arange(len(self.dataset))
        labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        
        # Ajustar n_splits para datasets pequenos
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        min_samples_per_class = label_counts.min() if len(label_counts) > 0 else 1
        n_splits = min(self.n_splits, min_samples_per_class, len(self.dataset) // 2)
        n_splits = max(n_splits, 2)  # Mínimo de 2 folds
        
        if n_splits != self.n_splits:
            self.logger.warning(
                f"Reduzindo n_splits de {self.n_splits} para {n_splits} "
                f"devido ao tamanho do dataset ({len(self.dataset)} amostras, "
                f"{len(unique_labels)} classes, min {min_samples_per_class} amostras/classe)"
            )
        
        # Splitter - usar KFold simples para datasets pequenos
        if self.stratified and min_samples_per_class >= n_splits:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            # Fallback para KFold simples quando stratified não é possível
            if self.stratified:
                self.logger.warning("Usando KFold simples (stratified não possível)")
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(splitter.split(indices, labels)):
            self.logger.info(f"Fold {fold + 1}/{n_splits}")
            
            # Criar subsets
            train_subset = Subset(self.dataset, train_idx)
            test_subset = Subset(self.dataset, test_idx)
            
            # Criar dataloaders
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if device.type == "cuda" else False,
            )
            
            test_loader = DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if device.type == "cuda" else False,
            )
            
            # Extrair embeddings
            embeddings_train, labels_train = self._extract_embeddings(train_loader, device)
            embeddings_test, labels_test = self._extract_embeddings(test_loader, device)
            
            # Computar métricas de identificação (CMC)
            cmc, cmc_metrics = self.metrics_computer.compute_cmc(
                embeddings_test,
                embeddings_train,
                labels_test,
                labels_train,
            )
            
            # Computar métricas de verificação (EER, FAR/FRR)
            genuine_scores, impostor_scores = self._compute_verification_scores(
                embeddings_test, labels_test
            )
            
            verification_metrics = {}
            if len(genuine_scores) > 0 and len(impostor_scores) > 0:
                _, _, verification_metrics = self.metrics_computer.compute_det(
                    genuine_scores, impostor_scores
                )
                self.logger.info(f"Fold {fold + 1} - EER: {verification_metrics.get('eer', 0):.4f}")
            
            fold_result = {
                "fold": fold + 1,
                "cmc": cmc.tolist(),
                "metrics": {**cmc_metrics, **verification_metrics},
                "num_genuine_pairs": len(genuine_scores),
                "num_impostor_pairs": len(impostor_scores),
            }
            
            fold_results.append(fold_result)
            self.logger.info(f"Fold {fold + 1} - Rank-1: {cmc_metrics.get('rank_1', 0):.4f}")
        
        # Agregar resultados
        self.results["fold_results"] = fold_results
        self._aggregate_results()
        
        return self.results
    
    def _extract_embeddings(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrair embeddings de um dataloader"""
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Desempacotar batch (pode ter paths se for dataset customizado)
                if len(batch_data) == 3:
                    images, batch_labels, image_paths = batch_data
                else:
                    images, batch_labels = batch_data
                
                images = images.to(device)
                
                outputs = self.model(images)
                embedding = outputs["embedding"]
                
                embeddings.append(embedding.cpu().numpy())
                labels.append(batch_labels.numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return embeddings, labels
    
    def _compute_verification_scores(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        max_pairs: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computar scores de verificação (pares genuínos e impostores).
        
        Pares genuínos: comparações entre impressões do mesmo dedo (mesma origem/label)
        Pares impostores: comparações entre impressões de dedos diferentes
        
        Args:
            embeddings: (N, D) array de embeddings
            labels: (N,) array de labels
            max_pairs: número máximo de pares para evitar explosão combinatória
        
        Returns:
            genuine_scores: scores de similaridade para pares genuínos
            impostor_scores: scores de similaridade para pares impostores
        """
        # Normalizar embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        genuine_scores = []
        impostor_scores = []
        
        n = len(embeddings)
        
        # Gerar pares
        for i in range(n):
            for j in range(i + 1, n):
                # Similaridade coseno
                score = np.dot(embeddings[i], embeddings[j])
                
                if labels[i] == labels[j]:
                    # Par genuíno (mesmo dedo)
                    genuine_scores.append(score)
                else:
                    # Par impostor (dedos diferentes)
                    impostor_scores.append(score)
                
                # Limitar número de pares para evitar explosão
                if len(genuine_scores) + len(impostor_scores) >= max_pairs:
                    break
            if len(genuine_scores) + len(impostor_scores) >= max_pairs:
                break
        
        return np.array(genuine_scores), np.array(impostor_scores)
    
    def _aggregate_results(self):
        """Agregar resultados de todos os folds"""
        all_rank_1 = []
        all_rank_5 = []
        all_rank_10 = []
        all_rank_20 = []
        all_eer = []
        all_far_at_frr_01 = []
        all_far_at_frr_001 = []
        
        for fold_result in self.results["fold_results"]:
            metrics = fold_result["metrics"]
            all_rank_1.append(metrics.get("rank_1", 0))
            all_rank_5.append(metrics.get("rank_5", 0))
            all_rank_10.append(metrics.get("rank_10", 0))
            all_rank_20.append(metrics.get("rank_20", 0))
            # Métricas de verificação
            if "eer" in metrics:
                all_eer.append(metrics["eer"])
            if "far_at_frr_0.1" in metrics:
                all_far_at_frr_01.append(metrics["far_at_frr_0.1"])
            if "far_at_frr_0.01" in metrics:
                all_far_at_frr_001.append(metrics["far_at_frr_0.01"])
        
        # Computar intervalos de confiança - Identificação (CMC)
        rank_1_mean, rank_1_ci_lower, rank_1_ci_upper = self.metrics_computer.compute_confidence_intervals(all_rank_1)
        rank_5_mean, rank_5_ci_lower, rank_5_ci_upper = self.metrics_computer.compute_confidence_intervals(all_rank_5)
        rank_10_mean, rank_10_ci_lower, rank_10_ci_upper = self.metrics_computer.compute_confidence_intervals(all_rank_10)
        rank_20_mean, rank_20_ci_lower, rank_20_ci_upper = self.metrics_computer.compute_confidence_intervals(all_rank_20)
        
        self.results["overall_metrics"] = {
            # Métricas de Identificação (1:N)
            "rank_1": {
                "mean": rank_1_mean,
                "ci_lower": rank_1_ci_lower,
                "ci_upper": rank_1_ci_upper,
                "std": np.std(all_rank_1),
            },
            "rank_5": {
                "mean": rank_5_mean,
                "ci_lower": rank_5_ci_lower,
                "ci_upper": rank_5_ci_upper,
                "std": np.std(all_rank_5),
            },
            "rank_10": {
                "mean": rank_10_mean,
                "ci_lower": rank_10_ci_lower,
                "ci_upper": rank_10_ci_upper,
                "std": np.std(all_rank_10),
            },
            "rank_20": {
                "mean": rank_20_mean,
                "ci_lower": rank_20_ci_lower,
                "ci_upper": rank_20_ci_upper,
                "std": np.std(all_rank_20),
            },
        }
        
        # Métricas de Verificação (1:1) - se disponíveis
        if all_eer:
            eer_mean, eer_ci_lower, eer_ci_upper = self.metrics_computer.compute_confidence_intervals(all_eer)
            self.results["overall_metrics"]["eer"] = {
                "mean": eer_mean,
                "ci_lower": eer_ci_lower,
                "ci_upper": eer_ci_upper,
                "std": np.std(all_eer),
            }
        
        if all_far_at_frr_01:
            far01_mean, far01_ci_lower, far01_ci_upper = self.metrics_computer.compute_confidence_intervals(all_far_at_frr_01)
            self.results["overall_metrics"]["far_at_frr_0.1"] = {
                "mean": far01_mean,
                "ci_lower": far01_ci_lower,
                "ci_upper": far01_ci_upper,
                "std": np.std(all_far_at_frr_01),
            }
        
        if all_far_at_frr_001:
            far001_mean, far001_ci_lower, far001_ci_upper = self.metrics_computer.compute_confidence_intervals(all_far_at_frr_001)
            self.results["overall_metrics"]["far_at_frr_0.01"] = {
                "mean": far001_mean,
                "ci_lower": far001_ci_lower,
                "ci_upper": far001_ci_upper,
                "std": np.std(all_far_at_frr_001),
            }
    
    def save_results(self, filename: str = "cv_results.json"):
        """Salvar resultados da validação cruzada"""
        results_file = self.experiment_dir / "results" / filename
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Resultados salvos em {results_file}")
