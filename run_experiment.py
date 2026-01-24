"""
Script principal para executar experimentos do DeepPrint
"""

import argparse
import logging
from pathlib import Path
from typing import Dict
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import sys

from config import EXPERIMENTS, TRAINING_CONFIG, MODEL_CONFIG, CROSS_VALIDATION_CONFIG
from training import DeepPrintTrainer
from validation import CrossValidator
from data_loader import create_dummy_dataset, FVCDatasetLoader, load_fvc_datasets, load_datasets


class ExperimentRunner:
    """Classe para executar experimentos"""
    
    def __init__(self, experiment_name: str, mode: str = "debug"):
        self.experiment_name = experiment_name
        self.mode = mode
        
        if experiment_name not in EXPERIMENTS:
            raise ValueError(f"Experimento {experiment_name} não encontrado")
        
        self.experiment_config = EXPERIMENTS[experiment_name]
        self.experiment_dir = self.experiment_config["dir"]
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info(f"Iniciando experimento: {self.experiment_config['name']}")
        self.logger.info(f"Modo: {mode}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging"""
        logger = logging.getLogger(f"Experiment_{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # Handler para arquivo
        log_file = self.experiment_dir / "logs" / f"experiment_{self.mode}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Handler para console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _evaluate_on_test(self, model, test_loader, device) -> Dict:
        """
        Avaliar modelo no conjunto de teste.
        Calcula métricas de verificação (EER, FAR/FRR) e identificação.
        
        Protocolo seguindo paper original (BIOSIG 2023):
        - Pares genuínos: todas comparações intra-classe
        - Pares impostores: amostragem balanceada (mesmo número que genuínos por amostra)
        
        Args:
            model: modelo treinado
            test_loader: DataLoader do conjunto de teste
            device: dispositivo (CPU/GPU)
        
        Returns:
            results: dicionário com métricas
        """
        from validation import PerformanceMetrics
        import random
        
        model.eval()
        embeddings = []
        labels = []
        
        # Extrair embeddings
        with torch.no_grad():
            for images, batch_labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                embedding = outputs["embedding"]
                embeddings.append(embedding.cpu().numpy())
                labels.append(batch_labels.numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        self.logger.info(f"Embeddings extraídos: {embeddings.shape}")
        
        # Normalizar embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Agrupar por classe
        unique_labels = np.unique(labels)
        class_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        # Calcular pares genuínos (todas comparações intra-classe)
        genuine_scores = []
        for label in unique_labels:
            indices = class_indices[label]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    score = np.dot(embeddings[indices[i]], embeddings[indices[j]])
                    genuine_scores.append(score)
        
        # Calcular pares impostores (amostragem balanceada como no paper)
        # Para cada amostra, fazer N comparações com outras classes (N = impressões por classe)
        random.seed(42)
        impostor_scores = []
        samples_per_class = len(class_indices[unique_labels[0]]) if len(unique_labels) > 0 else 1
        
        for label in unique_labels:
            indices = class_indices[label]
            other_labels = [l for l in unique_labels if l != label]
            
            for idx in indices:
                # Selecionar N amostras de outras classes aleatoriamente
                n_impostors = min(samples_per_class, len(other_labels))
                selected_labels = random.sample(list(other_labels), n_impostors)
                
                for other_label in selected_labels:
                    other_idx = random.choice(class_indices[other_label])
                    score = np.dot(embeddings[idx], embeddings[other_idx])
                    impostor_scores.append(score)
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        self.logger.info(f"Pares genuínos: {len(genuine_scores)}")
        self.logger.info(f"Pares impostores: {len(impostor_scores)}")
        
        n = len(embeddings)
        results = {
            "num_samples": n,
            "num_classes": len(unique_labels),
            "num_genuine_pairs": len(genuine_scores),
            "num_impostor_pairs": len(impostor_scores),
        }
        
        # Calcular métricas de verificação
        if len(genuine_scores) > 0 and len(impostor_scores) > 0:
            metrics_computer = PerformanceMetrics()
            _, _, verification_metrics = metrics_computer.compute_det(
                genuine_scores, impostor_scores
            )
            
            results.update(verification_metrics)
            
            self.logger.info(f"EER: {verification_metrics['eer']:.4f}")
            self.logger.info(f"FAR@FRR=0.1: {verification_metrics['far_at_frr_0.1']:.4f}")
            self.logger.info(f"FAR@FRR=0.01: {verification_metrics['far_at_frr_0.01']:.4f}")
            
            # Estatísticas dos scores
            self.logger.info(f"Scores genuínos: média={genuine_scores.mean():.4f}, std={genuine_scores.std():.4f}")
            self.logger.info(f"Scores impostores: média={impostor_scores.mean():.4f}, std={impostor_scores.std():.4f}")
        else:
            self.logger.warning("Não há pares suficientes para calcular métricas de verificação")
        
        # Salvar resultados
        results_file = self.experiment_dir / "results" / f"test_results_{self.mode}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Resultados salvos em {results_file}")
        
        return results
    
    def _run_pairwise_evaluation(self, model, dataset, device) -> Dict:
        """
        Executa avaliação pairwise (todas-contra-todas).
        
        Gera CSV com comparações incluindo qualidade NFIQ2.
        
        Args:
            model: modelo treinado
            dataset: dataset de teste
            device: dispositivo
        
        Returns:
            estatísticas da avaliação
        """
        from evaluation import run_pairwise_evaluation
        from config import DATA_DIR
        
        # Obter caminhos e labels do dataset
        file_paths = dataset.get_file_paths()
        labels = dataset.get_labels()
        
        self.logger.info(f"Avaliação pairwise com {len(file_paths)} amostras")
        self.logger.info(f"Total de comparações: {len(file_paths) * (len(file_paths) + 1) // 2}")
        
        # Executar avaliação
        stats = run_pairwise_evaluation(
            model=model,
            dataset=dataset,
            file_paths=file_paths,
            labels=labels,
            output_dir=str(self.experiment_dir),
            experiment_name=self.experiment_name,
            mode=self.mode,
            base_path=str(DATA_DIR),
            batch_size=TRAINING_CONFIG[self.mode]["batch_size"],
            device=device,
            logger=self.logger,
        )
        
        # Log resumo
        self.logger.info(f"Comparações realizadas: {stats['total_comparisons']}")
        self.logger.info(f"Pares genuínos: {stats['num_genuine_pairs']}")
        self.logger.info(f"Pares impostores: {stats['num_impostor_pairs']}")
        self.logger.info(f"Tempo total: {stats['total_time_seconds']:.2f}s")
        self.logger.info(f"Memória pico: {stats['memory_peak_mb']:.1f} MB")
        self.logger.info(f"CSV salvo em: {stats['output_file']}")
        
        return stats
    
    def run(self, use_fvc: bool = True, resume: bool = False, fvc_only: bool = False, aggressive_aug: bool = False, sfinge_only: bool = False, sfinge_fvc: bool = False):
        """Executar experimento completo
        
        Args:
            use_fvc: Se True, usa as bases FVC reais. Se False, usa dataset dummy.
            resume: Se True, retoma treinamento do último checkpoint.
            fvc_only: Se True, treina apenas com FVC (sem SD302).
            aggressive_aug: Se True, usa data augmentation agressivo.
            sfinge_only: Se True, treina apenas com SFinge (FP_gen_0).
            sfinge_fvc: Se True, treina com SFinge + FVC combinados.
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Experimento: {self.experiment_config['name']}")
        self.logger.info(f"Descrição: {self.experiment_config['description']}")
        self.logger.info("=" * 80)
        
        # Carregar dataset
        config = TRAINING_CONFIG[self.mode]
        random_state = CROSS_VALIDATION_CONFIG["random_state"]
        
        # Carregar bases de dados (FVC + SD302)
        self.logger.info("Carregando bases de dados...")
        
        # Determinar datasets a usar
        if self.mode == "debug":
            datasets_to_use = ["FVC2000"]  # Apenas uma base no debug
            sample_size = config.get("sample_size", 100)
        elif sfinge_fvc:
            datasets_to_use = ["FVC2000", "FVC2002", "FVC2004", "SFinge"]  # SFinge + FVC
            sample_size = None
            self.logger.info("Modo SFinge+FVC: treinando com SFinge (FP_gen_0 + FP_gen_1) + bases FVC")
        elif sfinge_only:
            datasets_to_use = ["SFinge"]  # Apenas SFinge (FP_gen_0 + FP_gen_1)
            sample_size = None
            self.logger.info("Modo SFinge-only: treinando apenas com base SFinge (FP_gen_0 + FP_gen_1)")
        elif fvc_only:
            datasets_to_use = ["FVC2000", "FVC2002", "FVC2004"]  # Apenas FVC
            sample_size = None
            self.logger.info("Modo FVC-only: treinando apenas com bases FVC (sem SD302)")
        else:
            # PADRÃO PROD: Usar SFinge (FP_gen_0 + FP_gen_1) - 84.000 imagens, ~8.400 classes
            datasets_to_use = ["SFinge"]
            sample_size = None
            self.logger.info("Modo PROD padrão: treinando com SFinge (FP_gen_0 + FP_gen_1 - 84.000 imagens)")
        
        if aggressive_aug:
            self.logger.info("Modo AUGMENTATION AGRESSIVO ativado")
        
        try:
            train_dataset, val_dataset, test_dataset, loaders = load_datasets(
                datasets=datasets_to_use,
                random_state=random_state,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                augment_train=True,
                aggressive_augment=aggressive_aug,
            )
            
            # Log estatísticas por loader
            for loader_name, loader in loaders.items():
                stats = loader.get_statistics()
                self.logger.info(f"=== {loader_name} ===")
                self.logger.info(f"Total de imagens: {stats['total_images']}")
                self.logger.info(f"Total de origens únicas: {stats['total_origins']}")
                if 'images_per_dataset' in stats:
                    self.logger.info(f"Imagens por dataset: {stats['images_per_dataset']}")
                if 'images_per_device' in stats:
                    self.logger.info(f"Imagens por device: {stats['images_per_device']}")
                self.logger.info(f"Versões por origem: min={stats['versions_per_origin']['min']}, "
                                f"max={stats['versions_per_origin']['max']}, "
                                f"média={stats['versions_per_origin']['mean']:.1f}")
            
            # Guardar loaders para uso posterior
            self.dataset_loaders = loaders
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar datasets: {e}")
            raise RuntimeError(f"Não foi possível carregar as bases de dados: {e}")
        
        self.logger.info(f"Dataset dividido: train={len(train_dataset)}, "
                        f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        # Criar dataloaders
        config = TRAINING_CONFIG[self.mode]
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=config["use_gpu"],
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["use_gpu"],
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["use_gpu"],
        )
        
        # Determinar tipo de modelo
        model_type_map = {
            "exp0_baseline": "baseline",
            "exp1_representacao_aumentada": "enhanced_representation",
            "exp2_atencao_espacial": "spatial_attention",
            "exp3_reranking": "reranking",
        }
        
        model_type = model_type_map[self.experiment_name]
        texture_dims = MODEL_CONFIG["texture_embedding_dims"][self.experiment_name]
        minutia_dims = MODEL_CONFIG["minutia_embedding_dims"].get(self.experiment_name, 0)
        
        # Criar trainer
        if minutia_dims > 0:
            total_dims = texture_dims + minutia_dims
            self.logger.info(f"Criando modelo: {model_type} com {texture_dims}+{minutia_dims}={total_dims} dimensões")
        else:
            self.logger.info(f"Criando modelo: {model_type} com {texture_dims} dimensões")
        
        trainer = DeepPrintTrainer(
            model_type=model_type,
            experiment_dir=self.experiment_dir,
            mode=self.mode,
            texture_embedding_dims=texture_dims,
            minutia_embedding_dims=minutia_dims,
        )
        
        # Treinar
        self.logger.info("Iniciando treinamento...")
        trainer.train(train_loader, val_loader, resume=resume)
        
        # Avaliação no conjunto de teste (verificação e identificação)
        self.logger.info("=" * 80)
        self.logger.info("Avaliação no conjunto de teste:")
        self.logger.info("=" * 80)
        test_results = self._evaluate_on_test(trainer.model, test_loader, trainer.device)
        
        # Validação cruzada
        self.logger.info("Iniciando validação cruzada...")
        cv = CrossValidator(
            model=trainer.model,
            dataset=test_dataset,
            experiment_dir=self.experiment_dir,
            mode=self.mode,
            logger=self.logger,
        )
        
        cv_results = cv.run_cross_validation(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            device=trainer.device,
        )
        
        cv.save_results(f"cv_results_{self.mode}.json")
        
        # Log resultados
        self.logger.info("=" * 80)
        self.logger.info("Resultados da Validação Cruzada:")
        self.logger.info("=" * 80)
        
        for metric_name, metric_values in cv_results["overall_metrics"].items():
            self.logger.info(
                f"{metric_name}: {metric_values['mean']:.4f} "
                f"(CI: [{metric_values['ci_lower']:.4f}, {metric_values['ci_upper']:.4f}])"
            )
        
        # Avaliação pairwise (todas-contra-todas)
        self.logger.info("=" * 80)
        self.logger.info("Avaliação Pairwise (todas-contra-todas):")
        self.logger.info("=" * 80)
        
        pairwise_results = self._run_pairwise_evaluation(
            model=trainer.model,
            dataset=test_dataset,
            device=trainer.device,
        )
        
        self.logger.info("=" * 80)
        self.logger.info("Experimento concluído com sucesso!")
        
        return {
            "test_results": test_results,
            "cv_results": cv_results,
            "pairwise_results": pairwise_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Executar experimentos do DeepPrint")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()),
        required=True,
        help="Nome do experimento a executar"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "prod"],
        default="debug",
        help="Modo de execução (debug ou prod)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retomar treinamento do último checkpoint"
    )
    parser.add_argument(
        "--fvc-only",
        action="store_true",
        help="Treinar apenas com bases FVC (sem SD302)"
    )
    parser.add_argument(
        "--aggressive-aug",
        action="store_true",
        help="Usar data augmentation agressivo (para datasets pequenos)"
    )
    parser.add_argument(
        "--sfinge",
        action="store_true",
        help="Treinar apenas com base SFinge (FP_gen_0)"
    )
    parser.add_argument(
        "--sfinge-fvc",
        action="store_true",
        help="Treinar com SFinge + FVC combinados (sem SD302)"
    )
    
    args = parser.parse_args()
    
    # Executar experimento
    runner = ExperimentRunner(args.experiment, args.mode)
    results = runner.run(resume=args.resume, fvc_only=args.fvc_only, aggressive_aug=args.aggressive_aug, sfinge_only=args.sfinge, sfinge_fvc=args.sfinge_fvc)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
