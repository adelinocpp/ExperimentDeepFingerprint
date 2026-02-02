# DeepPrint Experiments - Comparação de Melhorias Incrementais

[![Status](https://img.shields.io/badge/Status-Baseline%20Validado-success)]()
[![Baseline EER](https://img.shields.io/badge/Baseline%20EER-0.19%20(medium)-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)]()

Este projeto implementa e valida melhorias incrementais no modelo DeepPrint para reconhecimento de impressões digitais usando representações de tamanho fixo.

---

## 🎯 Status Atual

### ✅ Experimento 0: Baseline (CONCLUÍDO)
- **Problema inicial**: Colapso intermitente de embeddings (EER → 0.5)
- **Causa raiz**: Otimizador incorreto (Adam vs RMSprop) + Center Loss 100x maior
- **Solução**: RMSprop + correções de hiperparâmetros conforme paper
- **Resultados validados**:
  - Debug (20 classes): EER **0.20**
  - Medium (350 classes): EER **0.19**, separação **96%** ✅
  - Produção (8000 classes, 84k amostras): 🔄 **EM ANDAMENTO**

📄 **Documentação detalhada**: [RESOLUCAO_COLAPSO.md](RESOLUCAO_COLAPSO.md)

### ⏳ Experimentos 1-3 (AGUARDANDO)
Aguardando validação de produção do baseline antes de prosseguir com melhorias incrementais.

---

## 📋 Índice

- [Estrutura do Projeto](#estrutura-do-projeto)
- [Experimentos Planejados](#experimentos-planejados)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Configuração](#configuração)
- [Resultados](#resultados)
- [Troubleshooting](#troubleshooting)
- [Documentação](#documentação)
- [Referências](#referências)

---

## 📁 Estrutura do Projeto

```
deepprint_experiments/
├── config.py                          # Configuração centralizada (CRÍTICO)
├── models_base.py                     # Modelos base e variantes
├── training.py                        # Módulo de treinamento
├── validation.py                      # Módulo de validação cruzada
├── data_loader.py                     # Carregamento de dados
├── minutia_map_generator.py           # Geração de mapas de minúcias
├── run_experiment.py                  # Script principal
│
├── README.md                          # Este arquivo
├── RESOLUCAO_COLAPSO.md              # Documentação da correção do baseline
├── TESTES_REALIZADOS.md              # Log de todos os testes
│
├── exp0_baseline/                     # ✅ Experimento 0: DeepPrint Baseline
│   ├── logs/
│   │   ├── experiment_debug.log
│   │   ├── experiment_medium.log
│   │   └── experiment_prod.log
│   ├── results/
│   │   ├── test_results_debug.json
│   │   ├── test_results_medium.json
│   │   ├── cv_results_medium.json
│   │   └── pairwise_comparisons_medium.csv
│   └── checkpoints/
│       ├── best_model.pt              # Melhor modelo (salvo por EER)
│       ├── checkpoint_latest.pt
│       └── checkpoint_medium_backup.pt
│
├── exp1_representacao_aumentada/      # ⏳ Experimento 1 (planejado)
├── exp2_atencao_espacial/             # ⏳ Experimento 2 (planejado)
└── exp3_reranking/                    # ⏳ Experimento 3 (planejado)
```

---

## 🧪 Experimentos Planejados

### Experimento 0: DeepPrint Baseline ✅

**Objetivo**: Reproduzir DeepPrint original fielmente
**Modelo**: STN + 2 branches (texture + minutiae)
**Embedding**: 192 dimensões (96 + 96)
**Otimizador**: RMSprop (paper original)
**Status**: ✅ Validado até 350 classes, produção em andamento

**Hiperparâmetros críticos**:
- Center Loss weight: **0.00125** (paper)
- Otimizador: **RMSprop** (não Adam!)
- LR: 0.0001 (base), 0.0000035 (STN, 3.5% do base)
- Checkpoint criterion: **EER** (não val_loss!)

**Resultados**:
| Modo | Classes | Amostras | Épocas | EER | Separação |
|------|---------|----------|--------|-----|-----------|
| Debug | 20 | 200 | 5 | 0.20 | 30% |
| Medium | 350 | 3500 | 30 | **0.19** | **96%** |
| Prod | 8000 | 84000 | 256 | *rodando* | - |

### Experimento 1: Representação Aumentada ⏳

**Objetivo**: Aumentar capacidade representacional
**Modificação**: 192 → 1024 dimensões (512 + 512)
**Melhoria Esperada**: +2-5% em Rank-1
**Status**: Aguardando validação de produção do baseline

### Experimento 2: Atenção Espacial ⏳

**Objetivo**: Focar em regiões de alta qualidade
**Modificação**: Adicionar CBAM (Convolutional Block Attention Module)
**Melhoria Esperada**: +1-3% em Rank-1
**Status**: Aguardando validação de produção do baseline

### Experimento 3: Re-ranking Aprimorado ⏳

**Objetivo**: Melhorar recuperação top-k
**Modificação**: Learning-to-rank para candidatos
**Melhoria Esperada**: +1-2% em Rank-1, +2-3% em Rank-5
**Status**: Aguardando validação de produção do baseline

---

## 🚀 Instalação

### Requisitos de Sistema

- **Python**: 3.8+
- **GPU**: NVIDIA com CUDA (recomendado para modo prod)
  - Testado: RTX 2070 8GB
  - Mínimo: 6GB VRAM
- **CPU**: 8+ cores (para data loading)
- **RAM**: 16GB+ (32GB recomendado para prod)
- **Disco**: 50GB+ livres

### Dependências Python

```bash
# PyTorch (verificar versão CUDA compatível)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Processamento de dados
pip install numpy scipy scikit-learn
pip install opencv-python pillow

# Utilidades
pip install tqdm psutil

# Visualização (opcional)
pip install matplotlib seaborn
```

### Datasets

Configurar diretório de dados em `config.py`:

```python
# Para máquina "westeros" (default)
DATA_DIR = Path("/media/DRAGONSTONE/MEGAsync/.../Bases_de_Dados")

# Para máquina "STPM223"
DATA_DIR = Path("/home/adelino/MegaSync/.../Bases_de_Dados")
```

**Datasets suportados**:
- ✅ SFinge (84.000 imagens sintéticas, 8.000 origens)
- ⏳ NIST SD27 (planejado)
- ⏳ FVC2004 (planejado)

---

## ⚡ Uso Rápido

### 1. Teste Rápido (Debug - 5 min)

Valida que pipeline funciona com dataset pequeno:

```bash
python run_experiment.py --experiment exp0_baseline --mode debug
```

**Configuração**:
- 200 amostras (~20 classes)
- 5 épocas
- Batch size: 8
- Tempo: ~5-7 minutos
- EER esperado: ~0.20

### 2. Teste Intermediário (Medium - 3 horas)

Valida escalabilidade com dataset médio:

```bash
python run_experiment.py --experiment exp0_baseline --mode medium
```

**Configuração**:
- 3.500 amostras train, 750 val, 750 test (~350 classes)
- 30 épocas
- Batch size: 8
- Tempo: ~3-4 horas
- EER esperado: ~0.19

### 3. Produção (Prod - 25-30 dias)

Treinamento completo com todos os dados:

```bash
# Rodar em background
nohup python run_experiment.py --experiment exp0_baseline --mode prod > prod_training.log 2>&1 &

# Monitorar progresso
tail -f prod_training.log

# Ver log detalhado
tail -f exp0_baseline/logs/experiment_prod.log

# Ver uso da GPU
watch -n 5 nvidia-smi
```

**Configuração**:
- 84.000 amostras (~8.000 classes)
- 256 épocas (paper original)
- Batch size: 20
- Tempo: ~600-700 horas (25-30 dias)
- EER esperado: ~0.02-0.05 (2-5%, conforme paper)

---

## ⚙️ Configuração

### Arquivo `config.py`

**Configurações centralizadas** (modificar aqui, não no código!):

```python
# Modos de treinamento
TRAINING_CONFIG = {
    "debug": {
        "batch_size": 8,
        "num_epochs": 5,
        "sample_size": 200,
    },
    "medium": {
        "batch_size": 8,        # Ajustado para RTX 2070 8GB
        "num_epochs": 30,
        "sample_size": 5000,
    },
    "prod": {
        "batch_size": 20,       # Paper usa 30, ajustado para 8GB
        "num_epochs": 256,      # Paper original
        "sample_size": None,    # Todas 84k amostras
    },
}

# Otimizador (CRÍTICO!)
OPTIMIZER_CONFIG = {
    "optimizer": "rmsprop",     # NÃO MUDAR para "adam"!
    "rmsprop": {
        "lr": 0.0001,
        "alpha": 0.99,
        "weight_decay": 0,
    },
    "localization_network_lr_scale": 0.035,  # STN: 3.5% do LR base
}

# Loss (CRÍTICO!)
LOSS_CONFIG = {
    "center_loss_base_weight": 0.00125,     # Valor exato do paper
    "center_loss_use_adaptive": False,      # Desabilitado
    "softmax_loss_weight": 1.0,
    "minutia_map_loss_weight": 0.3,
}
```

### Modificações Comuns

**Reduzir uso de memória**:
```python
TRAINING_CONFIG["prod"]["batch_size"] = 16  # De 20 para 16
TRAINING_CONFIG["prod"]["num_workers"] = 4   # Reduzir workers
```

**Acelerar convergência** (experimental):
```python
OPTIMIZER_CONFIG["rmsprop"]["lr"] = 0.0002  # Dobrar LR (cuidado!)
```

**⚠️ NÃO MODIFICAR** (causam colapso):
- `optimizer`: Deve ser `"rmsprop"`
- `center_loss_base_weight`: Deve ser `0.00125`
- `center_loss_use_adaptive`: Deve ser `False`

---

## 📊 Resultados

### Estrutura de Saídas

Cada experimento gera:

```
exp0_baseline/
├── checkpoints/
│   ├── best_model.pt              # Melhor EER (use este!)
│   ├── checkpoint_latest.pt       # Último checkpoint (retomar)
│   └── checkpoint_medium_backup.pt
├── logs/
│   ├── experiment_debug.log
│   ├── experiment_medium.log
│   └── experiment_prod.log
└── results/
    ├── test_results_medium.json          # Métricas no test set
    ├── cv_results_medium.json            # Validação cruzada 5-fold
    ├── training_history_medium.json      # Loss/EER por época
    └── pairwise_comparisons_medium.csv   # Todas comparações
```

### Analisar Resultados

**Ver métricas principais**:
```bash
cat exp0_baseline/results/test_results_medium.json
```

**Output**:
```json
{
  "num_samples": 750,
  "num_classes": 75,
  "eer": 0.1946,
  "far_at_frr_0.1": 0.2243,
  "genuine_score_mean": 0.9872,
  "impostor_score_mean": 0.0228
}
```

**Ver validação cruzada**:
```bash
cat exp0_baseline/results/cv_results_medium.json | grep "eer"
```

**Ver histórico de treinamento**:
```bash
cat exp0_baseline/results/training_history_medium.json
```

### Scripts de Análise (TODO)

```bash
# Visualizar distribuição de scores
python visualize_scores.py --results exp0_baseline/results/pairwise_comparisons_medium.csv

# Plotar curvas de treinamento
python plot_training_history.py --history exp0_baseline/results/training_history_medium.json

# Analisar embeddings (t-SNE)
python analyze_embeddings.py --checkpoint exp0_baseline/checkpoints/best_model.pt
```

---

## 🔧 Troubleshooting

### Erro: "CUDA out of memory"

**Solução 1**: Reduzir batch size
```python
# Em config.py
TRAINING_CONFIG["prod"]["batch_size"] = 16  # De 20 para 16
```

**Solução 2**: Reduzir num_workers
```python
TRAINING_CONFIG["prod"]["num_workers"] = 4  # De 8 para 4
```

**Solução 3**: Usar modo debug/medium para testar antes de prod
```bash
python run_experiment.py --mode medium  # Testa com menos dados
```

### Erro: "Dataset não encontrado"

**Causa**: Diretório de dados incorreto

**Solução**: Ajustar `DATA_DIR` em `config.py`:
```python
DATA_DIR = Path("/seu/caminho/para/Bases_de_Dados")
```

### Erro: EER muito alto (> 0.4) ou colapso (EER = 0.5)

**Causa provável**: Configuração incorreta

**Verificar**:
1. Otimizador é `"rmsprop"` (não "adam")
2. Center Loss weight é `0.00125` (não 0.125)
3. Center Loss adaptativo está `False`

**Arquivo**: `config.py`, linhas 105, 149, 152

### Warning: "Checkpoint salvo com val_loss"

**Ignorar**: É esperado. Checkpoint é salvo por EER, mas val_loss é registrado para compatibilidade.

### Processo travado / sem progresso

**Verificar**:
```bash
# Ver se processo está rodando
ps aux | grep run_experiment

# Ver últimas linhas do log
tail -20 exp0_baseline/logs/experiment_prod.log

# Ver uso da GPU
nvidia-smi
```

---

## 📚 Documentação

### Documentos Principais

- **[RESOLUCAO_COLAPSO.md](RESOLUCAO_COLAPSO.md)**: Documentação completa da correção do baseline
  - Cronologia da investigação
  - Análise técnica RMSprop vs Adam
  - Lições aprendidas
  - Arquivos modificados

- **[TESTES_REALIZADOS.md](TESTES_REALIZADOS.md)**: Log de todos os testes executados
  - Resultados de cada teste
  - Configurações usadas
  - Tempo de execução

### Estrutura de Código

**Arquivos críticos**:
- `config.py`: Configuração centralizada (**modificar aqui!**)
- `training.py`: Loop de treinamento, otimizador, checkpoint
- `models_base.py`: Arquitetura DeepPrint
- `data_loader.py`: Carregamento e augmentation

**Fluxo de treinamento**:
1. `run_experiment.py` → carrega config
2. `data_loader.py` → carrega datasets
3. `training.py` → treina modelo
4. `validation.py` → valida com cross-validation
5. Salva checkpoints, logs, resultados

### Arquitetura DeepPrint

```
Input (299x299 grayscale)
    ↓
[STN] Spatial Transformer Network
    ↓
Aligned image (299x299)
    ↓
Inception-ResNet-v2 (shared backbone)
    ↓
    ├─→ [Texture Branch]  → 96 dims
    │
    └─→ [Minutia Branch]  → 96 dims
         ↓
    Concatenate
         ↓
    Embedding (192 dims)
         ↓
    L2 Normalize
         ↓
    [Softmax] → Classification
    [Center Loss] → Embedding quality
    [Minutia Map Loss] → Minutiae localization
```

---

## 📖 Referências

### Paper Original

**DeepPrint**:
- Engelsma, J. J., Cao, K., & Jain, A. K. (2019). **Learning a Fixed-Length Fingerprint Representation**. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- arXiv: [1909.09901v2](https://arxiv.org/abs/1909.09901)

### Métodos Relacionados

**Center Loss**:
- Wen, Y., Zhang, K., Li, Z., & Qiao, Y. (2016). **A Discriminative Feature Learning Approach for Deep Face Recognition**. ECCV 2016.

**Inception-ResNet-v2**:
- Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2016). **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning**. AAAI 2017.

**Spatial Transformer Networks**:
- Jaderberg, M., Simonyan, K., & Zisserman, A. (2015). **Spatial Transformer Networks**. NeurIPS 2015.

### Otimizadores

**RMSprop**:
- Tieleman, T., & Hinton, G. (2012). **Lecture 6.5 - RMSprop**. COURSERA: Neural Networks for Machine Learning.

**Adam**:
- Kingma, D. P., & Ba, J. (2014). **Adam: A Method for Stochastic Optimization**. ICLR 2015.

### Benchmarks

**Fixed-Length Fingerprint Representations**:
- Rohwedder, T., Osorio-Roig, D., Rathgeb, C., & Busch, C. (2023). **Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types**. BIOSIG 2023.

---

## 👥 Autor

**Projeto**: Papiloscopia Computacional - Comparação de Métodos Automáticos
**Instituição**: [Informação não divulgada]
**Orientador**: Dr. Adelino [Sobrenome não divulgado]

---

## 📝 Licença

Este projeto segue a mesma licença do repositório original do DeepPrint.

---

## 🙏 Agradecimentos

Ao Dr. Adelino, que:
- Identificou o ciclo vicioso de raciocínio circular
- Estabeleceu o princípio: *"Se funciona para poucas amostras, pode funcionar para muitas"*
- Exigiu investigação profunda e rigorosa
- Forneceu feedback direto e honesto

> *"Sou Dr. e pesquisador e sei quando alguém está andando em círculos."*

---

## 📅 Histórico de Versões

### v0.2.0 (2026-02-02) - **BASELINE VALIDADO**
- ✅ Corrigido colapso de embeddings
- ✅ RMSprop + hiperparâmetros corretos
- ✅ Validado até 350 classes (EER 0.19)
- 🔄 Produção em andamento (8000 classes)

### v0.1.0 (2026-01-15) - Implementação Inicial
- Estrutura base do projeto
- 4 experimentos planejados
- Baseline com problemas de colapso

---

## 🚦 Status dos Componentes

| Componente | Status | Observações |
|------------|--------|-------------|
| Baseline (exp0) | ✅ Validado | Medium OK, prod rodando |
| Data loading | ✅ OK | SFinge 84k imagens |
| Training loop | ✅ OK | RMSprop, EER checkpoint |
| Validation | ✅ OK | 5-fold CV implementado |
| Exp1 (1024 dims) | ⏳ Aguardando | Após prod |
| Exp2 (Atenção) | ⏳ Aguardando | Após prod |
| Exp3 (Re-ranking) | ⏳ Aguardando | Após prod |
| Refatoração | ⏳ Planejado | utils/ modules |
| Scripts análise | ⏳ Planejado | visualize, plot |

---

**Última atualização**: 2026-02-02 01:30
**Próxima milestone**: Validação de produção (ETA: ~30 dias)

Para dúvidas ou problemas, consultar [RESOLUCAO_COLAPSO.md](RESOLUCAO_COLAPSO.md) ou verificar logs em `exp0_baseline/logs/`.
