#!/usr/bin/env python3
"""Script rápido para calcular EER do modelo atual"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader
import sys

# Carregar componentes do projeto
from data_loader import FingerprintDataset, load_datasets
from models_base import DeepPrintBaseline
from config import DATA_DIR

device = torch.device("cpu")  # Forçar CPU (GPU ocupada pelo treinamento)
print(f"Device: {device} (usando CPU para evitar conflito com treinamento)\n")

# Carregar dados usando a mesma função do experimento
print("Carregando dataset SFinge...")
train_dataset, val_dataset, test_dataset, loaders = load_datasets(
    datasets=["SFinge"],
    random_state=42,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    augment_train=False,
    aggressive_augment=False,
)

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)
print(f"Val dataset: {len(val_dataset)} amostras\n")

# Carregar modelo (usar latest do treinamento em execução)
model_path = Path("exp0_baseline/checkpoints/checkpoint_latest.pt")
print(f"Carregando modelo de {model_path}...")

model = DeepPrintBaseline(
    num_classes=6000,
    texture_embedding_dims=96,
    minutia_embedding_dims=96
)

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)
model.eval()

print(f"Modelo carregado (época {checkpoint.get('epoch', '?')})\n")
print("="*60)
print("CALCULANDO EER COM CÓDIGO CORRIGIDO")
print("="*60 + "\n")

# Usar a função corrigida do training.py
from training import DeepPrintTrainer

trainer = DeepPrintTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    experiment_dir=Path("exp0_baseline"),
    learning_rate=0.05,
    num_epochs=1,
    mode="prod"
)

# Calcular EER
result = trainer._compute_quick_eer(val_loader, max_samples=2000)

if result:
    print(f"\n{'='*60}")
    print(f"RESULTADO:")
    print(f"  EER: {result['eer']:.4f} ({result['eer']*100:.2f}%)")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  FAR@FRR=0.1: {result['far_at_frr_01']:.4f}")
    print(f"  Pares genuínos: {result['num_genuine']}")
    print(f"  Pares impostores: {result['num_impostor']}")
    print(f"  Classes: {result['num_classes']}")
    print(f"{'='*60}")
    
    print(f"\nInterpretação:")
    eer = result['eer']
    if eer > 0.4:
        print("  ⚠️  EER muito alto - modelo praticamente aleatório")
    elif eer > 0.2:
        print("  ⚠️  EER alto - modelo ainda está aprendendo")
    elif eer > 0.1:
        print("  ✓  EER razoável - modelo aprendendo bem")
    elif eer > 0.01:
        print("  ✓✓ EER bom - modelo com boa capacidade discriminativa")
    else:
        print("  ✓✓✓ EER excelente - modelo muito bom!")
else:
    print("ERRO ao calcular EER")
