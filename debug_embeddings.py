#!/usr/bin/env python3
"""
Debug: Investigar por que embeddings colapsaram (todos idênticos)
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data_loader import load_datasets
from models_base import DeepPrintBaseline

print("="*60)
print("DEBUG: INVESTIGANDO COLAPSO DE EMBEDDINGS")
print("="*60)

# Carregar dataset
print("\n1. Carregando dataset...")
_, val_dataset, _, _ = load_datasets(
    datasets=["SFinge"],
    random_state=42,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Carregar modelo
print("2. Carregando modelo...")
model = DeepPrintBaseline(num_classes=6000, texture_embedding_dims=96, minutia_embedding_dims=96)
checkpoint_path = Path("exp0_baseline/checkpoints/checkpoint_latest.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

epoch = checkpoint.get('epoch', '?')
print(f"   Modelo carregado (época {epoch})")

# Extrair embeddings SEM normalização
print("\n3. Extraindo embeddings (SEM normalização)...")
embeddings_raw = []
embeddings_normalized = []
labels_list = []

with torch.no_grad():
    for i, batch_data in enumerate(val_loader):
        if i >= 5:  # 5 batches = 40 amostras
            break
        
        images, labels = batch_data[0], batch_data[1]
        outputs = model(images)
        
        # Pegar texture_embedding ANTES da normalização
        if "texture_embedding" in outputs:
            emb_raw = outputs["texture_embedding"].cpu().numpy()
            embeddings_raw.append(emb_raw)
        
        # Pegar embedding final (já normalizado)
        emb_norm = outputs["embedding"].cpu().numpy()
        embeddings_normalized.append(emb_norm)
        labels_list.extend(labels.numpy())

embeddings_raw = np.vstack(embeddings_raw)
embeddings_normalized = np.vstack(embeddings_normalized)

print(f"   Extraídos {len(embeddings_raw)} embeddings")

# Análise detalhada
print("\n" + "="*60)
print("ANÁLISE DOS EMBEDDINGS")
print("="*60)

print("\n📊 EMBEDDINGS RAW (antes da normalização):")
print(f"   Shape: {embeddings_raw.shape}")
print(f"   Min:   {embeddings_raw.min():.6f}")
print(f"   Max:   {embeddings_raw.max():.6f}")
print(f"   Mean:  {embeddings_raw.mean():.6f}")
print(f"   Std:   {embeddings_raw.std():.6f}")

# Verificar se são todos iguais
first_emb = embeddings_raw[0]
all_same = np.all(np.abs(embeddings_raw - first_emb) < 1e-6)
print(f"\n   ⚠️  Todos embeddings RAW são idênticos? {all_same}")

if all_same:
    print(f"\n   DEBUG - Primeiro embedding RAW (primeiros 10 valores):")
    print(f"   {first_emb[:10]}")
else:
    print(f"\n   ✓ Embeddings RAW são DIFERENTES (isso é bom)")
    # Mostrar diferença entre primeiro e segundo
    diff = np.abs(embeddings_raw[0] - embeddings_raw[1]).mean()
    print(f"   Diferença média entre emb[0] e emb[1]: {diff:.6f}")

print("\n📊 EMBEDDINGS NORMALIZED (após normalização):")
print(f"   Shape: {embeddings_normalized.shape}")
print(f"   Min:   {embeddings_normalized.min():.6f}")
print(f"   Max:   {embeddings_normalized.max():.6f}")
print(f"   Mean:  {embeddings_normalized.mean():.6f}")
print(f"   Std:   {embeddings_normalized.std():.6f}")

# Verificar normas (devem ser ~1.0)
norms = np.linalg.norm(embeddings_normalized, axis=1)
print(f"\n   Normas L2 (devem ser ~1.0):")
print(f"   Min:  {norms.min():.6f}")
print(f"   Max:  {norms.max():.6f}")
print(f"   Mean: {norms.mean():.6f}")

# Verificar se são todos iguais após normalização
first_emb_norm = embeddings_normalized[0]
all_same_norm = np.all(np.abs(embeddings_normalized - first_emb_norm) < 1e-6)
print(f"\n   ⚠️  Todos embeddings NORMALIZED são idênticos? {all_same_norm}")

if all_same_norm:
    print(f"\n   DEBUG - Primeiro embedding NORMALIZED (primeiros 10 valores):")
    print(f"   {first_emb_norm[:10]}")
else:
    print(f"\n   ✓ Embeddings NORMALIZED são DIFERENTES (isso é bom)")

# Calcular similaridades
print("\n📊 SIMILARIDADES (produto escalar):")
sim_01 = np.dot(embeddings_normalized[0], embeddings_normalized[1])
sim_02 = np.dot(embeddings_normalized[0], embeddings_normalized[2])
print(f"   emb[0] · emb[1] = {sim_01:.6f}")
print(f"   emb[0] · emb[2] = {sim_02:.6f}")

if sim_01 > 0.999 and sim_02 > 0.999:
    print(f"\n   ⚠️  PROBLEMA: Similaridades muito altas (> 0.999)")
    print(f"   Embeddings são praticamente idênticos!")

# Verificar pesos do modelo
print("\n" + "="*60)
print("ANÁLISE DOS PESOS DO MODELO")
print("="*60)

total_params = 0
zero_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    zeros = (param.abs() < 1e-8).sum().item()
    zero_params += zeros
    
    if zeros / param.numel() > 0.5:  # Mais de 50% zeros
        print(f"\n⚠️  {name}:")
        print(f"   Shape: {param.shape}")
        print(f"   Zeros: {zeros}/{param.numel()} ({100*zeros/param.numel():.1f}%)")

print(f"\n📊 RESUMO:")
print(f"   Total de parâmetros: {total_params:,}")
print(f"   Parâmetros ~zero: {zero_params:,} ({100*zero_params/total_params:.2f}%)")

if zero_params / total_params > 0.8:
    print(f"\n   🚨 CRÍTICO: Mais de 80% dos pesos são zero!")
    print(f"   Modelo pode ter colapsado durante treinamento.")

print("\n" + "="*60)
print("CONCLUSÃO")
print("="*60)

if all_same:
    print("❌ PROBLEMA: Embeddings RAW são todos iguais")
    print("   → Modelo não está gerando features diferentes")
    print("   → Possível causa: pesos colapsaram ou bug na arquitetura")
elif all_same_norm:
    print("❌ PROBLEMA: Embeddings NORMALIZED são todos iguais")
    print("   → Normalização causou colapso")
    print("   → Embeddings RAW são OK mas normalização falha")
else:
    print("✓ Embeddings são diferentes (raw e normalized)")
    print("   → Problema deve estar no cálculo de EER ou outra parte")

print("\n" + "="*60)
