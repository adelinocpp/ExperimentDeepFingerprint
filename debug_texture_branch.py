#!/usr/bin/env python3
"""
Debug: Investigar passo a passo onde Texture Branch colapsa
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data_loader import load_datasets
from models_base import DeepPrintBaseline

print("="*60)
print("DEBUG: TEXTURE BRANCH PASSO A PASSO")
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
batch_data = next(iter(val_loader))
images = batch_data[0]

# Carregar modelo
print("2. Carregando modelo (época 3)...")
model = DeepPrintBaseline(num_classes=6000, texture_embedding_dims=96, minutia_embedding_dims=96)
checkpoint_path = Path("exp0_baseline/checkpoints/checkpoint_latest.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

print(f"   Modelo carregado (época {checkpoint.get('epoch', '?')})")

# Pegar 3 imagens
img_batch = images[:3]

print("\n" + "="*60)
print("PASSANDO 3 IMAGENS PELO TEXTURE BRANCH (PASSO A PASSO)")
print("="*60)

with torch.no_grad():
    # Até o Stem (já sabemos que funciona)
    x = model.localization(img_batch)
    x = model.stem(x)
    
    print(f"\nEntrada do Texture Branch (após Stem):")
    print(f"   Shape: {x.shape}")
    diff_input = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_input:.6f}")
    
    # Texture branch components
    tb = model.texture_branch
    
    # Block 0
    print(f"\n1. _0_block (4x Inception_A + Reduction_A):")
    x = tb._0_block(x)
    print(f"   Shape: {x.shape}")
    diff_0 = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_0:.6f}")
    if diff_0 < 1e-6:
        print(f"   ❌ COLAPSO AQUI! Block 0 retorna outputs idênticos")
    
    # Block 1
    print(f"\n2. _1_block (7x Inception_B + Reduction_B):")
    x = tb._1_block(x)
    print(f"   Shape: {x.shape}")
    diff_1 = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_1:.6f}")
    if diff_1 < 1e-6:
        print(f"   ❌ COLAPSO AQUI! Block 1 retorna outputs idênticos")
    
    # Block 2
    print(f"\n3. _2_block (3x Inception_C):")
    x = tb._2_block(x)
    print(f"   Shape: {x.shape}")
    diff_2 = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_2:.6f}")
    if diff_2 < 1e-6:
        print(f"   ❌ COLAPSO AQUI! Block 2 retorna outputs idênticos")
    
    # AvgPool2d
    print(f"\n4. AvgPool2d (kernel_size=8):")
    x = tb._3_avg_pool2d(x)
    print(f"   Shape: {x.shape}")
    diff_3 = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_3:.6f}")
    if diff_3 < 1e-6:
        print(f"   ❌ COLAPSO AQUI! AvgPool retorna outputs idênticos")
    
    # Flatten
    print(f"\n5. Flatten:")
    x = tb._4_flatten(x)
    print(f"   Shape: {x.shape}")
    diff_4 = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_4:.6f}")
    
    # Dropout (em eval() não faz nada)
    print(f"\n6. Dropout (p=0.2, mas desligado em eval()):")
    x = tb._5_dropout(x)
    print(f"   Shape: {x.shape}")
    diff_5 = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_5:.6f}")
    
    # Linear
    print(f"\n7. Linear (1536 → 96):")
    x_before_linear = x.clone()
    x = tb._6_linear(x)
    print(f"   Shape: {x.shape}")
    diff_6 = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_6:.6f}")
    
    if diff_6 < 1e-6:
        print(f"   ❌ COLAPSO AQUI! Linear retorna outputs idênticos")
        
        # Verificar pesos da linear
        weight = tb._6_linear.weight.data.numpy()
        bias = tb._6_linear.bias.data.numpy()
        
        print(f"\n   Análise dos pesos da Linear:")
        print(f"   Weight shape: {weight.shape}")
        print(f"   Weight norm: {np.linalg.norm(weight):.6f}")
        print(f"   Weight min/max: {weight.min():.6f} / {weight.max():.6f}")
        print(f"   Bias: min={bias.min():.6f}, max={bias.max():.6f}, mean={bias.mean():.6f}")
        
        # Testar output da linear manualmente
        out_manual_0 = torch.nn.functional.linear(x_before_linear[0:1], tb._6_linear.weight, tb._6_linear.bias)
        out_manual_1 = torch.nn.functional.linear(x_before_linear[1:2], tb._6_linear.weight, tb._6_linear.bias)
        diff_manual = torch.abs(out_manual_0 - out_manual_1).mean().item()
        print(f"\n   Teste manual da linear:")
        print(f"   Output[0]: {out_manual_0[0, :5].numpy()}")
        print(f"   Output[1]: {out_manual_1[0, :5].numpy()}")
        print(f"   Diferença: {diff_manual:.6f}")
    
    # Normalize
    print(f"\n8. F.normalize (L2 norm):")
    x = torch.nn.functional.normalize(x.squeeze(-1).squeeze(-1) if x.dim() > 2 else x, dim=1)
    print(f"   Shape: {x.shape}")
    diff_7 = torch.abs(x[0] - x[1]).mean().item()
    print(f"   Diferença [0] vs [1]: {diff_7:.6f}")
    
    if diff_7 < 1e-6:
        print(f"   ❌ COLAPSO AQUI! Normalização retorna outputs idênticos")
    
    # Similaridade final
    print(f"\n9. Similaridade final (produto escalar):")
    sim = torch.dot(x[0], x[1]).item()
    print(f"   x[0] · x[1] = {sim:.6f}")

print("\n" + "="*60)
print("CONCLUSÃO")
print("="*60)

print("\nO passo que mostrou '❌ COLAPSO AQUI!' é a causa raiz.")
print("="*60)
