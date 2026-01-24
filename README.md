# DeepPrint Experiments - Comparação de Melhorias Incrementais

Este projeto implementa 4 experimentos do DeepPrint para avaliar o impacto de melhorias incrementais no reconhecimento de impressões digitais.

## Estrutura do Projeto

```
deepprint_experiments/
├── config.py                          # Configuração centralizada
├── models_base.py                     # Modelos base e variantes
├── training.py                        # Módulo de treinamento
├── validation.py                      # Módulo de validação cruzada
├── data_loader.py                     # Carregamento de dados
├── run_experiment.py                  # Script principal
├── README.md                          # Este arquivo
│
├── exp0_baseline/                     # Experimento 0: DeepPrint Baseline
│   ├── IMPLEMENTATION.md              # Documentação detalhada
│   ├── models/
│   ├── logs/
│   ├── results/
│   └── checkpoints/
│
├── exp1_representacao_aumentada/      # Experimento 1: Representação Aumentada
│   ├── IMPLEMENTATION.md
│   ├── models/
│   ├── logs/
│   ├── results/
│   └── checkpoints/
│
├── exp2_atencao_espacial/             # Experimento 2: Atenção Espacial
│   ├── IMPLEMENTATION.md
│   ├── models/
│   ├── logs/
│   ├── results/
│   └── checkpoints/
│
└── exp3_reranking/                    # Experimento 3: Re-ranking Aprimorado
    ├── IMPLEMENTATION.md
    ├── models/
    ├── logs/
    ├── results/
    └── checkpoints/
```

## Experimentos

### Experimento 0: DeepPrint Baseline
- **Descrição**: DeepPrint puro sem modificações
- **Embedding**: 512 dimensões
- **Baseline**: Sim
- **Documentação**: `exp0_baseline/IMPLEMENTATION.md`

### Experimento 1: Representação Aumentada
- **Descrição**: Aumentar dimensionalidade para 1024 com refinamento
- **Embedding**: 1024 dimensões
- **Melhoria Esperada**: +2-5% em Rank-1
- **Documentação**: `exp1_representacao_aumentada/IMPLEMENTATION.md`

### Experimento 2: Atenção Espacial
- **Descrição**: Adicionar mecanismos de atenção para focar em regiões de qualidade
- **Embedding**: 512 dimensões
- **Melhoria Esperada**: +1-3% em Rank-1
- **Documentação**: `exp2_atencao_espacial/IMPLEMENTATION.md`

### Experimento 3: Re-ranking Aprimorado
- **Descrição**: Implementar rede neural para re-ranking de candidatos
- **Embedding**: 512 dimensões
- **Melhoria Esperada**: +1-2% em Rank-1, +2-3% em Rank-5
- **Documentação**: `exp3_reranking/IMPLEMENTATION.md`

## Requisitos

### Dependências Python

```bash
pip install torch torchvision
pip install numpy scipy scikit-learn
pip install opencv-python pillow
pip install tqdm psutil
pip install matplotlib seaborn
```

### Hardware Recomendado

- **CPU**: 8+ cores
- **RAM**: 64 GB
- **GPU**: NVIDIA com CUDA (recomendado para modo prod)

### Datasets

Os datasets devem estar localizados em:
```
/home/adelino/MegaSync/Forense/Papiloscopia/Compara_Metodos_Automaticos/Bases_de_Dados/
```

Estrutura esperada:
```
Bases_de_Dados/
├── NIST_SD27/
│   ├── train/
│   ├── val/
│   └── test/
├── FVC2004/
│   ├── train/
│   ├── val/
│   └── test/
└── ...
```

## Uso

### Modo Debug (Teste Rápido)

```bash
# Experimento 0
python run_experiment.py --experiment exp0_baseline --mode debug

# Experimento 1
python run_experiment.py --experiment exp1_representacao_aumentada --mode debug

# Experimento 2
python run_experiment.py --experiment exp2_atencao_espacial --mode debug

# Experimento 3
python run_experiment.py --experiment exp3_reranking --mode debug
```

**Tempo esperado**: ~5-7 minutos por experimento
**Dados**: 100 amostras (teste)

### Modo Production (Completo)

```bash
# Experimento 0
python run_experiment.py --experiment exp0_baseline --mode prod

# Experimento 1
python run_experiment.py --experiment exp1_representacao_aumentada --mode prod

# Experimento 2
python run_experiment.py --experiment exp2_atencao_espacial --mode prod

# Experimento 3
python run_experiment.py --experiment exp3_reranking --mode prod
```

**Tempo esperado**: ~2-4 horas por experimento
**Dados**: Todos os dados disponíveis

### Executar Todos os Experimentos

```bash
for exp in exp0_baseline exp1_representacao_aumentada exp2_atencao_espacial exp3_reranking; do
    echo "Executando $exp em modo debug..."
    python run_experiment.py --experiment $exp --mode debug
    
    echo "Executando $exp em modo prod..."
    python run_experiment.py --experiment $exp --mode prod
done
```

## Configuração

### Arquivo `config.py`

Todas as configurações estão centralizadas em `config.py`:

- **TRAINING_CONFIG**: Configurações de treinamento (batch size, epochs, etc.)
- **MODEL_CONFIG**: Configurações de modelo (dimensionalidade, dropout, etc.)
- **CROSS_VALIDATION_CONFIG**: Configurações de validação cruzada
- **METRICS_CONFIG**: Configurações de métricas
- **AUGMENTATION_CONFIG**: Configurações de data augmentation

### Modificar Configurações

```python
# Em config.py
TRAINING_CONFIG["prod"]["batch_size"] = 128  # Aumentar batch size
TRAINING_CONFIG["prod"]["num_epochs"] = 200  # Aumentar epochs
MODEL_CONFIG["texture_embedding_dims"]["exp0_baseline"] = 256  # Reduzir dimensionalidade
```

## Resultados

### Estrutura de Saídas

Cada experimento gera:

```
exp*/
├── models/
│   └── best_model.pt              # Melhor modelo treinado
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   └── ...
├── logs/
│   └── training_*.log             # Logs detalhados
└── results/
    ├── training_history_debug.json
    ├── training_history_prod.json
    ├── cv_results_debug.json
    └── cv_results_prod.json
```

### Analisar Resultados

```python
import json
import numpy as np

# Carregar resultados
with open("exp0_baseline/results/cv_results_prod.json") as f:
    results = json.load(f)

# Extrair métricas
print("Experimento 0 (Baseline):")
for metric, values in results["overall_metrics"].items():
    print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"    CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]")

# Comparar experimentos
experiments = [
    "exp0_baseline",
    "exp1_representacao_aumentada",
    "exp2_atencao_espacial",
    "exp3_reranking"
]

rank1_scores = []
for exp in experiments:
    with open(f"{exp}/results/cv_results_prod.json") as f:
        results = json.load(f)
        rank1 = results["overall_metrics"]["rank_1"]["mean"]
        rank1_scores.append(rank1)

# Calcular melhoria
baseline = rank1_scores[0]
for i, exp in enumerate(experiments[1:], 1):
    improvement = (rank1_scores[i] - baseline) / baseline * 100
    print(f"{exp}: {improvement:+.2f}% melhoria")
```

## Testes Estatísticos

### Teste de Significância

```python
from scipy import stats

# Carregar scores de cada fold
baseline_scores = [fold["metrics"]["rank_1"] for fold in baseline_results["fold_results"]]
enhanced_scores = [fold["metrics"]["rank_1"] for fold in enhanced_results["fold_results"]]

# Teste t pareado
t_stat, p_value = stats.ttest_rel(enhanced_scores, baseline_scores)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Melhoria é estatisticamente significante (p < 0.05)")
else:
    print("Melhoria não é estatisticamente significante (p >= 0.05)")
```

## Troubleshooting

### Erro: "Dataset não encontrado"

Verifique se os datasets estão em:
```
/home/adelino/MegaSync/Forense/Papiloscopia/Compara_Metodos_Automaticos/Bases_de_Dados/
```

Ou modifique `DATA_DIR` em `config.py`.

### Erro: "CUDA out of memory"

Reduza batch size em `config.py`:
```python
TRAINING_CONFIG["prod"]["batch_size"] = 32  # Reduzir de 64
```

### Erro: "Too many open files"

Reduza num_workers em `config.py`:
```python
TRAINING_CONFIG["prod"]["num_workers"] = 4  # Reduzir de 8
```

## Documentação Adicional

Para detalhes específicos de cada experimento, consulte:

- `exp0_baseline/IMPLEMENTATION.md` - DeepPrint Baseline
- `exp1_representacao_aumentada/IMPLEMENTATION.md` - Representação Aumentada
- `exp2_atencao_espacial/IMPLEMENTATION.md` - Atenção Espacial
- `exp3_reranking/IMPLEMENTATION.md` - Re-ranking Aprimorado

## Referências

1. Engelsma, J. J., Cao, K., & Jain, A. K. (2019). Learning a Fixed-Length Fingerprint Representation. IEEE Transactions on Pattern Analysis and Machine Intelligence.

2. Rohwedder, T., Osorio-Roig, D., Rathgeb, C., & Busch, C. (2023). Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types. BIOSIG 2023.

3. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. ECCV 2018.

4. Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2016). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. AAAI 2017.

## Autor

Manus AI - Projeto de Papiloscopia Computacional

## Licença

Este projeto segue a mesma licença do repositório original do DeepPrint.
# ExperimentDeepFingerprint
