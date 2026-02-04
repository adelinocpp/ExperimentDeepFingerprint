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
    "debug_minimal": {  # TESTE: Encontrar mínimo de classes para funcionar
        "batch_size": 8,
        "num_epochs": 10,
        "num_workers": 2,
        "sample_size": 200,  # ~14 classes (mesmo que debug que funcionou)
        "use_gpu": True,
        "mixed_precision": False,
    },
    "debug": {
        "batch_size": 8,
        "num_epochs": 5,  # Debug mais completo para verificar convergência
        "num_workers": min(2, NUM_CPUS),  # Usa mínimo entre 2 e CPUs disponíveis
        "sample_size": 200,  # ~20 classes x 10 amostras (SFinge tem minutiae)
        "use_gpu": True,  # Usar GPU para ser rápido
        "mixed_precision": False,  # Desabilitado (problemas de compatibilidade)
    },
    "medium": {  # NOVO: Teste intermediário - validar hipótese dataset/épocas
        "batch_size": 8,  # REDUZIDO: 16→8 para evitar CUDA OOM (RTX 2070 8GB)
        "num_epochs": 30,  # Mais épocas para convergência adequada
        "num_workers": min(4, NUM_CPUS),
        "sample_size": 5000,  # ~500 classes × 10 amostras - dataset significativo
        "use_gpu": True,
        "mixed_precision": False,
    },
    "prod": {
        "batch_size": 20,  # Ajustado para RTX 2070 8GB (paper usa 30)
        "num_epochs": 300,  # RESTAURADO: Paper original usa 256 épocas (ou 140K steps)
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
# TESTE: RMSprop (paper) vs Adam (implementação original)
OPTIMIZER_CONFIG = {
    "optimizer": "rmsprop",  # TESTE: trocar para "adam" se RMSprop não funcionar
    "rmsprop": {
        "lr": 0.00005,  # Mesmo LR base que Adam
        "alpha": 0.99,  # RMSprop decay (default PyTorch)
        "eps": 1e-8,
        "weight_decay": 0,  # Testar sem weight decay primeiro
        "momentum": 0,
    },
    "adam": {
        "lr": 0.0001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
    },
    "localization_network_lr_scale": 0.035,  # STN usa 3.5% do LR base
    "use_lr_scheduler": False,
    "scheduler_type": "cosine",
    "warmup_epochs": 5,
    "min_lr": 5e-5,
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

# Configuração de perda
# CORRIGIDO: Center Loss com peso do PAPER ORIGINAL (0.00125, NÃO 0.125!)
# O peso estava 100x MAIOR causando colapso prematuro dos embeddings
#
# CENTER LOSS ADAPTATIVO: O peso é ajustado baseado no número de classes
# - Paper usa 0.00125 para ~6000 classes (60k amostras treinamento)
# - Com menos classes, o peso deve ser MENOR para evitar convergência prematura
# - Fórmula: weight_adaptativo = base_weight × (num_classes / 6000)^expoente
#
# ARCFACE: Opção alternativa ao Center Loss
# - ArcFace (Deng et al., 2019) usa margem angular aditiva no espaço angular
# - Superior ao Center Loss em face recognition (LFW: 99.53% vs 99.28%)
# - Escala melhor para muitas classes (testado até 85K classes)
# - Embora desenvolvido para faces, os princípios são domain-agnostic
LOSS_CONFIG = {
    # Tipo de loss: "center" (original DeepPrint) ou "arcface" (superior)
    "loss_type": "center",  # Opções: "center" | "arcface"

    # Center Loss (original DeepPrint)
    "center_loss_base_weight": 0.00250,  # Peso base do paper (para 6000 classes)
    "center_loss_num_classes_reference": 6000,  # Número de classes do paper
    "center_loss_adaptive_exponent": 0.7,  # Expoente de escala (0.7 = BALANCEADO, 0.5 = muito fraco, 1.0 = linear)
    "center_loss_use_adaptive": True,  # DESABILITADO: peso adaptativo muito baixo para datasets pequenos (14 classes → 0.000018)
    "center_loss_min_weight": 1e-7,  # Peso mínimo (proteção contra zero)
    "center_loss_max_weight": 0.01,  # Peso máximo (proteção contra explosão)

    # ArcFace Loss (alternativa superior)
    # Valores do paper "ArcFace: Additive Angular Margin Loss" (CVPR 2019)
    "arcface_margin": 0.5,  # m: Angular margin em radianos (~28.6°) - controla separação inter-classe
    "arcface_scale": 64.0,  # s: Feature scale - controla magnitude dos logits
    "arcface_easy_margin": False,  # Se True, usa easy margin (mais permissivo); False = hard margin (padrão)

    # Outras losses
    "triplet_loss_weight": 0.0,   # DESABILITADO (não existe no original)
    "softmax_loss_weight": 1.0,   # λ1 = 1.0 (apenas usado com center loss)
    "minutia_map_loss_weight": 0.3,  # Implementação original: 0.3
}


def get_center_loss_weight(num_classes: int) -> float:
    """
    Calcula peso adaptativo do Center Loss baseado no número de classes.

    PROTEÇÕES:
    - Valida entrada (num_classes >= 1)
    - Clamp em [min_weight, max_weight] para evitar valores extremos
    - Usa expoente SUBLINEAR (0.5) por padrão para crescimento controlado

    Rationale:
    - Center Loss força embeddings de mesma classe a convergirem para centros
    - Com POUCAS classes: centros ficam próximos → Center Loss forte causa colapso prematuro
    - Com MUITAS classes: centros podem se espalhar → Center Loss precisa ser mais forte
    - Expoente 0.5 (raiz quadrada): crescimento SUBLINEAR mais seguro que linear

    Args:
        num_classes: Número de classes no treinamento atual (>= 1)

    Returns:
        Peso do Center Loss ajustado, garantido em [min_weight, max_weight]

    Examples:
        >>> get_center_loss_weight(6000)  # Paper original (exponent=0.5)
        0.00125
        >>> get_center_loss_weight(14)    # Debug mode (exponent=0.5)
        0.0000060  # ~208x menor (mais conservador que linear)
        >>> get_center_loss_weight(200)   # Teste intermediário (exponent=0.5)
        0.0000722  # ~17x menor
        >>> get_center_loss_weight(100000)  # Muitas classes (exponent=0.5)
        0.0051 (clamped to max=0.01)  # PROTEGIDO!

    Raises:
        ValueError: Se num_classes < 1
    """
    # Validação de entrada
    if num_classes < 1:
        raise ValueError(f"num_classes deve ser >= 1, recebido: {num_classes}")

    if not LOSS_CONFIG["center_loss_use_adaptive"]:
        # Modo fixo: usa valor base do paper
        return LOSS_CONFIG["center_loss_base_weight"]

    base_weight = LOSS_CONFIG["center_loss_base_weight"]
    num_classes_ref = LOSS_CONFIG["center_loss_num_classes_reference"]
    exponent = LOSS_CONFIG["center_loss_adaptive_exponent"]
    min_weight = LOSS_CONFIG["center_loss_min_weight"]
    max_weight = LOSS_CONFIG["center_loss_max_weight"]

    # Escala: weight = base × (N_atual / N_ref)^exponent
    scale_factor = (num_classes / num_classes_ref) ** exponent
    adaptive_weight = base_weight * scale_factor

    # CLAMP: garantir limites seguros
    adaptive_weight = max(min_weight, min(adaptive_weight, max_weight))

    return adaptive_weight

# Configuração de data augmentation (normal)
# CORRIGIDO: Paper usa augmentation mais agressivo (±60° rotation, ±80px translation)
# Augmentation anterior era muito conservador e impedia STN de aprender alinhamentos robustos
AUGMENTATION_CONFIG = {
    "rotation_range": 60,  # CORRIGIDO: Paper usa ±60°, não ±15°
    "translation_range": 80,  # CORRIGIDO: Paper usa ±80px, não ±25px
    "padding": 80,  # NOVO: Padding para comportar translações (paper)
    "border_mode": "white",  # CORRIGIDO: Paper usa fill branco (background natural), não REFLECT
    "quality_augmentation": True,
    "contrast_range": (0.9, 2.0),  # CORRIGIDO: Paper ranges
    "brightness_range": (0.9, 1.1),  # CORRIGIDO: Paper ranges
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
