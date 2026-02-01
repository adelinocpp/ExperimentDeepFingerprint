#!/usr/bin/env python3
"""
Debug: Verificar se imagens de entrada são diferentes
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data_loader import load_datasets
from models_base import DeepPrintBaseline

print("="*60)
print("DEBUG: VERIFICANDO IMAGENS DE ENTRADA")
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

# Pegar primeiro batch
print("2. Carregando primeiro batch de imagens...")
batch_data = next(iter(val_loader))
images, labels = batch_data[0], batch_data[1]

print(f"   Batch shape: {images.shape}")
print(f"   Labels: {labels.numpy()}")

# Verificar se imagens são diferentes
print("\n" + "="*60)
print("ANÁLISE DAS IMAGENS")
print("="*60)

print("\n📊 ESTATÍSTICAS DAS IMAGENS:")
for i in range(min(5, len(images))):
    img = images[i].numpy()
    print(f"\nImagem {i} (label={labels[i].item()}):")
    print(f"   Min:  {img.min():.6f}")
    print(f"   Max:  {img.max():.6f}")
    print(f"   Mean: {img.mean():.6f}")
    print(f"   Std:  {img.std():.6f}")

# Verificar se são todas iguais
print("\n📊 COMPARAÇÃO ENTRE IMAGENS:")
img0 = images[0].numpy()
img1 = images[1].numpy()
img2 = images[2].numpy()

diff_01 = np.abs(img0 - img1).mean()
diff_02 = np.abs(img0 - img2).mean()
diff_12 = np.abs(img1 - img2).mean()

print(f"   Diferença média img[0] vs img[1]: {diff_01:.6f}")
print(f"   Diferença média img[0] vs img[2]: {diff_02:.6f}")
print(f"   Diferença média img[1] vs img[2]: {diff_12:.6f}")

if diff_01 < 1e-6 and diff_02 < 1e-6:
    print("\n   ❌ PROBLEMA: Imagens são IDÊNTICAS!")
    print("   → Bug no data loader")
else:
    print("\n   ✓ Imagens são DIFERENTES (isso é esperado)")

# Carregar modelo e passar imagens
print("\n" + "="*60)
print("DEBUG: ANALISANDO FORWARD PASS")
print("="*60)

print("\n3. Carregando modelo...")
model = DeepPrintBaseline(num_classes=6000, texture_embedding_dims=96, minutia_embedding_dims=96)
checkpoint_path = Path("exp0_baseline/checkpoints/checkpoint_latest.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

print("4. Passando imagens pelo modelo (passo a passo)...")

with torch.no_grad():
    # Pegar apenas 3 imagens
    img_batch = images[:3]
    
    print(f"\n   Input shape: {img_batch.shape}")
    
    # 1. STN (Spatial Transformer Network)
    if hasattr(model, 'localization'):
        x_aligned = model.localization(img_batch)
        print(f"\n   Após STN shape: {x_aligned.shape}")
        
        # Verificar se transformações são diferentes
        diff_stn_01 = torch.abs(x_aligned[0] - x_aligned[1]).mean().item()
        diff_stn_02 = torch.abs(x_aligned[0] - x_aligned[2]).mean().item()
        print(f"   Diferença após STN [0] vs [1]: {diff_stn_01:.6f}")
        print(f"   Diferença após STN [0] vs [2]: {diff_stn_02:.6f}")
        
        if diff_stn_01 < 1e-6 and diff_stn_02 < 1e-6:
            print(f"   ❌ PROBLEMA: STN está retornando MESMA transformação para todas imagens!")
        else:
            print(f"   ✓ STN está funcionando corretamente")
    else:
        x_aligned = img_batch
        print("   (Modelo sem STN)")
    
    # 2. Stem
    if hasattr(model, 'stem'):
        features = model.stem(x_aligned)
        print(f"\n   Após Stem shape: {features.shape}")
        
        diff_stem_01 = torch.abs(features[0] - features[1]).mean().item()
        diff_stem_02 = torch.abs(features[0] - features[2]).mean().item()
        print(f"   Diferença após Stem [0] vs [1]: {diff_stem_01:.6f}")
        print(f"   Diferença após Stem [0] vs [2]: {diff_stem_02:.6f}")
        
        if diff_stem_01 < 1e-6 and diff_stem_02 < 1e-6:
            print(f"   ❌ PROBLEMA: Stem está retornando MESMA feature para todas imagens!")
        else:
            print(f"   ✓ Stem está funcionando corretamente")
    
    # 3. Texture branch
    if hasattr(model, 'texture_branch'):
        texture_emb = model.texture_branch(features)
        print(f"\n   Texture embedding shape: {texture_emb.shape}")
        
        diff_tex_01 = torch.abs(texture_emb[0] - texture_emb[1]).mean().item()
        diff_tex_02 = torch.abs(texture_emb[0] - texture_emb[2]).mean().item()
        print(f"   Diferença texture [0] vs [1]: {diff_tex_01:.6f}")
        print(f"   Diferença texture [0] vs [2]: {diff_tex_02:.6f}")
        
        if diff_tex_01 < 1e-6 and diff_tex_02 < 1e-6:
            print(f"   ❌ PROBLEMA: Texture branch está retornando MESMO embedding!")
        else:
            print(f"   ✓ Texture branch está funcionando corretamente")

print("\n" + "="*60)
print("CONCLUSÃO")
print("="*60)
print("\nSe alguma camada acima mostrou '❌ PROBLEMA', essa é a causa raiz.")
print("Se todas mostraram '✓', o bug está em outra parte do código.")
print("="*60)
