#!/usr/bin/env python3
"""
Debug: Analisar checkpoint época 3 (treinamento NOVO com STN corrigido)
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data_loader import load_datasets
from models_base import DeepPrintBaseline

print("="*60)
print("DEBUG: CHECKPOINT DA ÉPOCA (TREINAMENTO NOVO)")
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
batch_data = next(iter(val_loader))
images, labels = batch_data[0], batch_data[1]

# Carregar modelo
print("2. Carregando modelo (época atual)...")
model = DeepPrintBaseline(num_classes=6000, texture_embedding_dims=96, minutia_embedding_dims=96)
checkpoint_path = Path("exp0_baseline/checkpoints/checkpoint_latest.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

epoch = checkpoint.get('epoch', '?')
print(f"   Modelo carregado (época {epoch})")

print("\n" + "="*60)
print("TESTE: Passar 3 imagens REAIS pelo modelo")
print("="*60)

img_batch = images[:3]

print(f"\nImagens de entrada diferentes?")
diff_01 = torch.abs(img_batch[0] - img_batch[1]).mean().item()
diff_02 = torch.abs(img_batch[0] - img_batch[2]).mean().item()
print(f"   img[0] vs img[1]: {diff_01:.6f}")
print(f"   img[0] vs img[2]: {diff_02:.6f}")

with torch.no_grad():
    # 1. Passar pelo STN
    print(f"\n1. STN:")
    stn = model.localization
    
    # Verificar thetas gerados
    resized = stn.resize(img_batch)
    xs = stn.localization(resized)
    xs_flat = xs.view(3, -1)
    theta_xy_raw = stn.fc_loc(xs_flat)
    
    print(f"   Thetas RAW (antes tanh):")
    for i in range(3):
        print(f"   img[{i}]: {theta_xy_raw[i].numpy()}")
    
    # Aplicar tanh (como no forward corrigido)
    theta_limited = torch.tanh(theta_xy_raw[:, 0]) * 3.14159
    offset_x = torch.tanh(theta_xy_raw[:, 1])
    offset_y = torch.tanh(theta_xy_raw[:, 2])
    
    print(f"\n   Thetas APÓS tanh (limitados):")
    for i in range(3):
        print(f"   img[{i}]: theta={theta_limited[i].item():.6f}, offset_x={offset_x[i].item():.6f}, offset_y={offset_y[i].item():.6f}")
    
    # Output do STN
    x_aligned = stn(img_batch)
    
    diff_stn_01 = torch.abs(x_aligned[0] - x_aligned[1]).mean().item()
    diff_stn_02 = torch.abs(x_aligned[0] - x_aligned[2]).mean().item()
    print(f"\n   Outputs do STN diferentes?")
    print(f"   out[0] vs out[1]: {diff_stn_01:.6f}")
    print(f"   out[0] vs out[2]: {diff_stn_02:.6f}")
    
    if diff_stn_01 < 1e-6:
        print(f"   ❌ STN ainda retorna outputs IDÊNTICOS!")
    else:
        print(f"   ✓ STN funciona corretamente")
    
    # 2. Passar pelo Stem
    print(f"\n2. Stem:")
    stem = model.stem
    features = stem(x_aligned)
    
    diff_stem_01 = torch.abs(features[0] - features[1]).mean().item()
    diff_stem_02 = torch.abs(features[0] - features[2]).mean().item()
    print(f"   Outputs do Stem diferentes?")
    print(f"   feat[0] vs feat[1]: {diff_stem_01:.6f}")
    print(f"   feat[0] vs feat[2]: {diff_stem_02:.6f}")
    
    if diff_stem_01 < 1e-6:
        print(f"   ❌ Stem retorna features IDÊNTICAS!")
    else:
        print(f"   ✓ Stem funciona corretamente")
    
    # 3. Texture branch
    print(f"\n3. Texture branch:")
    texture_emb = model.texture_branch(features)
    
    diff_tex_01 = torch.abs(texture_emb[0] - texture_emb[1]).mean().item()
    diff_tex_02 = torch.abs(texture_emb[0] - texture_emb[2]).mean().item()
    print(f"   Embeddings diferentes?")
    print(f"   emb[0] vs emb[1]: {diff_tex_01:.6f}")
    print(f"   emb[0] vs emb[2]: {diff_tex_02:.6f}")
    
    if diff_tex_01 < 1e-6:
        print(f"   ❌ Embeddings IDÊNTICOS!")
    else:
        print(f"   ✓ Embeddings diferentes")
    
    # Similaridades (produto escalar)
    print(f"\n4. Similaridades (cosine):")
    sim_01 = torch.dot(texture_emb[0], texture_emb[1]).item()
    sim_02 = torch.dot(texture_emb[0], texture_emb[2]).item()
    print(f"   emb[0] · emb[1] = {sim_01:.6f}")
    print(f"   emb[0] · emb[2] = {sim_02:.6f}")

print("\n" + "="*60)
print("ANÁLISE DOS PESOS DO STN (época 3)")
print("="*60)

fc_final = stn.fc_loc[2]
weight = fc_final.weight.data.numpy()
bias = fc_final.bias.data.numpy()

print(f"\nÚltima camada do STN:")
print(f"   Weight norm: {np.linalg.norm(weight):.6f}")
print(f"   Bias: {bias}")

weight_small = np.abs(weight).max() < 0.1
if weight_small:
    print(f"\n   ⚠️  Pesos muito pequenos (< 0.1)")
    print(f"   STN pode não ter aprendido o suficiente em 3 épocas")
else:
    print(f"\n   ✓ Pesos razoáveis")

print("\n" + "="*60)
print("CONCLUSÃO")
print("="*60)

if diff_stn_01 < 1e-6:
    print("\n❌ PROBLEMA ESTÁ NO STN:")
    print("   Mesmo com correção tanh(), STN retorna outputs idênticos")
    print("   Possíveis causas:")
    print("   1. Pesos do STN muito pequenos (não treinaram suficiente)")
    print("   2. Bug na correção do tanh()")
    print("   3. Problema com BatchNorm no STN")
elif diff_stem_01 < 1e-6:
    print("\n❌ PROBLEMA ESTÁ NO STEM:")
    print("   STN funciona mas Stem retorna features idênticas")
    print("   Possível causa: BatchNorm com running stats ruins")
elif diff_tex_01 < 1e-6:
    print("\n❌ PROBLEMA ESTÁ NO TEXTURE BRANCH:")
    print("   STN e Stem funcionam mas embeddings são idênticos")
else:
    print("\n✓ Modelo parece funcionar corretamente")
    print("   Problema pode estar no cálculo de EER ou outro lugar")

print("="*60)
