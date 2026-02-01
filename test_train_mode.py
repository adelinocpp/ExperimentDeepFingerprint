#!/usr/bin/env python3
"""
Teste: Verificar se problema é BatchNorm comparando eval() vs train() mode
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from data_loader import load_datasets
from models_base import DeepPrintBaseline

print("="*60)
print("TESTE: EVAL() vs TRAIN() MODE")
print("="*60)

# Carregar dataset
_, val_dataset, _, _ = load_datasets(
    datasets=["SFinge"],
    random_state=42,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
batch_data = next(iter(val_loader))
images = batch_data[0][:3]  # 3 imagens

# Carregar modelo
model = DeepPrintBaseline(num_classes=6000, texture_embedding_dims=96, minutia_embedding_dims=96)
checkpoint_path = Path("exp0_baseline/checkpoints/checkpoint_latest.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

print(f"\nModelo carregado (época {checkpoint.get('epoch', '?')})")

# TESTE 1: EVAL() MODE (atual)
print("\n" + "="*60)
print("1. EVAL() MODE (BatchNorm usa running stats)")
print("="*60)

model.eval()

with torch.no_grad():
    outputs_eval = model(images)
    emb_eval = outputs_eval["embedding"]

diff_eval = torch.abs(emb_eval[0] - emb_eval[1]).mean().item()
sim_eval = torch.dot(emb_eval[0], emb_eval[1]).item()

print(f"\nEmbeddings em EVAL():")
print(f"   Diferença [0] vs [1]: {diff_eval:.6f}")
print(f"   Similaridade [0] · [1]: {sim_eval:.6f}")

if diff_eval < 1e-6:
    print(f"   ❌ Embeddings IDÊNTICOS em eval()")
else:
    print(f"   ✓ Embeddings diferentes em eval()")

# TESTE 2: TRAIN() MODE (BatchNorm usa batch stats)
print("\n" + "="*60)
print("2. TRAIN() MODE (BatchNorm usa batch stats)")
print("="*60)

model.train()

with torch.no_grad():
    outputs_train = model(images)
    emb_train = outputs_train["embedding"]

diff_train = torch.abs(emb_train[0] - emb_train[1]).mean().item()
sim_train = torch.dot(emb_train[0], emb_train[1]).item()

print(f"\nEmbeddings em TRAIN():")
print(f"   Diferença [0] vs [1]: {diff_train:.6f}")
print(f"   Similaridade [0] · [1]: {sim_train:.6f}")

if diff_train < 1e-6:
    print(f"   ❌ Embeddings IDÊNTICOS em train()")
else:
    print(f"   ✓ Embeddings DIFERENTES em train()")

# CONCLUSÃO
print("\n" + "="*60)
print("CONCLUSÃO")
print("="*60)

if diff_eval < 1e-6 and diff_train > 1e-3:
    print("\n🎯 CONFIRMADO: Problema é BatchNorm com running stats ruins!")
    print("\nCAUSA:")
    print("   - Em eval(): BatchNorm usa running_mean/running_var (acumulados)")
    print("   - Com apenas 3 épocas, essas estatísticas estão ruins")
    print("   - BatchNorm 'esmaga' diferenças entre amostras")
    print("\nSOLUÇÕES:")
    print("   1. Treinar por mais épocas (>20) para running stats estabilizarem")
    print("   2. Usar train() mode durante inferência (não recomendado)")
    print("   3. Substituir BatchNorm por GroupNorm (mais estável)")
    print("   4. Inicializar modelo com pesos pré-treinados")
elif diff_eval < 1e-6 and diff_train < 1e-6:
    print("\n❌ Problema persiste em AMBOS os modos!")
    print("   Não é apenas BatchNorm, há outro bug na arquitetura")
else:
    print("\n✓ Embeddings funcionam corretamente em ambos os modos")
    print("   Problema deve estar em outro lugar")

print("="*60)
