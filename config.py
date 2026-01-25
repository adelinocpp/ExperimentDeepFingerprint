"""
Configuração centralizada para todos os experimentos do DeepPrint
"""

import os
import socket
from pathlib import Path

# Diretórios base
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT

# Seleção automática do diretório de dados baseado no hostname
HOSTNAME = socket.gethostname().lower()  # Normalizar para minúsculas
DATA_DIRS = {
    "STPM223": Path("/home/adelino/MegaSync/Forense/Papiloscopia/Compara_Metodos_Automaticos/Bases_de_Dados"),
    "westeros": Path("/media/DRAGONSTONE/MEGAsync/Forense/Papiloscopia/Compara_Metodos_Automaticos/Bases_de_Dados"),
}
# Usar o diretório correspondente ao host ou fallback para westeros
DATA_DIR = DATA_DIRS.get(HOSTNAME, DATA_DIRS["westeros"])

# Detectar número de CPUs da máquina
NUM_CPUS = os.cpu_count() or 4

# Configuração de experimentos
EXPERIMENTS = {
    "exp0_baseline": {
        "name": "DeepPrint Baseline",
        "description": "DeepPrint original completo (LocTexMinu): STN + 2 branches, 192 dims total",
        "dir": EXPERIMENTS_DIR / "exp0_baseline",
        "model_variant": "baseline",
    },
    "exp1_representacao_aumentada": {
        "name": "DeepPrint com Representação Aumentada",
        "description": "Aumentar representação de 512 para 1024 dimensões com atenção seletiva",
        "dir": EXPERIMENTS_DIR / "exp1_representacao_aumentada",
        "model_variant": "DeepPrint_LocTexMinu_1024",
    },
    "exp2_atencao_espacial": {
        "name": "DeepPrint com Atenção Espacial",
        "description": "Adicionar mecanismos de atenção espacial para focar em regiões de alta qualidade",
        "dir": EXPERIMENTS_DIR / "exp2_atencao_espacial",
        "model_variant": "DeepPrint_LocTexMinu_512_SpatialAttention",
    },
    "exp3_reranking": {
        "name": "DeepPrint com Re-ranking Aprimorado",
        "description": "Implementar re-ranking scheme com learning-to-rank",
        "dir": EXPERIMENTS_DIR / "exp3_reranking",
        "model_variant": "DeepPrint_LocTexMinu_512_Reranking",
    },
}

# Configuração de treinamento
TRAINING_CONFIG = {
    "debug": {
        "batch_size": 8,
        "num_epochs": 2,  # Debug rápido
        "num_workers": min(2, NUM_CPUS),  # Usa mínimo entre 2 e CPUs disponíveis
        "sample_size": 32,  # Mínimo possível para testar
        "use_gpu": True,  # Usar GPU para ser rápido
        "mixed_precision": False,
    },
    "prod": {
        "batch_size": 16,  # Reduzido para 16 - RTX 2070 8GB com 801 classes (~4GB VRAM)
        "num_epochs": 150,  # Reduzido de 256 para 150 (suficiente para 84k imagens)
        "num_workers": NUM_CPUS,  # Usa número de CPUs da máquina
        "sample_size": None,  # Usar todas as amostras (84k)
        "use_gpu": True,
        "mixed_precision": False,  # Desabilitado - causa colapso de pesos
    },
}

# Configuração de validação cruzada
CROSS_VALIDATION_CONFIG = {
    "n_splits": 5,  # 5-fold cross-validation
    "random_state": 42,
    "stratified": True,
}

# Configuração de métricas
METRICS_CONFIG = {
    "rank_k": [1, 5, 10, 20],  # Calcular rank-1, rank-5, rank-10, rank-20
    "confidence_level": 0.95,  # 95% de intervalo de confiança
}

# Configuração de otimizador
# Otimizado para SFinge (84k imagens, ~8.4k classes)
OPTIMIZER_CONFIG = {
    "adam": {
        "lr": 0.0005,  # Reduzido para 0.0005 (mais estável para 84k imagens)
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-4,  # Ajustado para regularização otimizada
    },
    "use_lr_scheduler": True,  # Ativar scheduler
    "scheduler_type": "cosine",  # Cosine annealing (melhor para datasets grandes)
    "warmup_epochs": 5,  # 5 épocas de warmup
    "min_lr": 1e-6,  # LR mínimo
}

# Configuração de modelo
MODEL_CONFIG = {
    "input_size": 299,
    "texture_embedding_dims": {
        "exp0_baseline": 96,  # Baseline: 96 (texture) + 96 (minutiae) = 192 total
        "exp1_representacao_aumentada": 512,  # Exp1: 512 (texture) + 512 (minutiae) = 1024 total
        "exp2_atencao_espacial": 96,  # Exp2: 96 (texture) + 96 (minutiae) = 192 total + SpatialAttention
        "exp3_reranking": 96,  # Exp3: 96 (texture) + 96 (minutiae) = 192 total + Reranking
    },
    "minutia_embedding_dims": {
        "exp0_baseline": 96,  # Baseline: 96 dims (minutiae)
        "exp1_representacao_aumentada": 512,  # Exp1: 512 dims (minutiae) - dimensão aumentada
        "exp2_atencao_espacial": 96,  # Exp2: 96 dims (minutiae) - mantém baseline
        "exp3_reranking": 96,  # Exp3: 96 dims (minutiae) - mantém baseline
    },
    "dropout_rate": 0.2,  # Original usa 0.2
    "use_localization": True,
    "use_texture": True,
    "use_minutiae": True,
}

# Configuração de perda (baseado no projeto original DeepPrint)
# Original deep_print_loss.py: W_CROSS_ENTROPY = 1.0, W_CENTER_LOSS = 0.125, W_MINUTIA_MAP_LOSS = 0.3
# Triplet loss NÃO é usado no DeepPrint original
LOSS_CONFIG = {
    "center_loss_weight": 0.125,  # Original: W_CENTER_LOSS
    "triplet_loss_weight": 0.0,   # DESABILITADO (não existe no original)
    "softmax_loss_weight": 1.0,   # Original: W_CROSS_ENTROPY
    "minutia_map_loss_weight": 0.3,  # Original: W_MINUTIA_MAP_LOSS (apenas para minutiae branch)
}

# Configuração de data augmentation (normal)
AUGMENTATION_CONFIG = {
    "rotation_range": 15,  # -15 a +15 graus
    "translation_range": 25,  # até 25 pixels em x e y
    "quality_augmentation": True,
    "contrast_range": (0.8, 1.2),
    "brightness_range": (0.8, 1.2),
}

# Configuração de data augmentation AGRESSIVO (para datasets pequenos)
AGGRESSIVE_AUGMENTATION_CONFIG = {
    "rotation_range": 30,  # -30 a +30 graus
    "translation_range": 40,  # até 40 pixels em x e y
    "quality_augmentation": True,
    "contrast_range": (0.6, 1.4),
    "brightness_range": (0.6, 1.4),
    "elastic_deformation": True,
    "elastic_alpha": 50,
    "elastic_sigma": 5,
    "gaussian_noise": True,
    "noise_std": 0.05,
    "random_erasing": True,
    "erasing_prob": 0.3,
    "erasing_scale": (0.02, 0.1),
}

# Configuração de logging
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_interval": 100,  # Log a cada 10 batches
    "save_checkpoint_interval": 5,  # Salvar checkpoint a cada 5 epochs
}

# Configuração de dispositivo
DEVICE_CONFIG = {
    "use_cuda": True,
    "cuda_visible_devices": "0",  # GPU 0
    "num_threads": 8,
}

# Configuração de caminhos de saída
OUTPUT_CONFIG = {
    "models_dir": "models",
    "logs_dir": "logs",
    "results_dir": "results",
    "checkpoints_dir": "checkpoints",
}
