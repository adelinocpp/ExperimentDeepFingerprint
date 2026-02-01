#!/usr/bin/env python3
"""
Debug: Verificar pesos do STN (Spatial Transformer Network)
"""

import torch
import numpy as np
from pathlib import Path

from models_base import DeepPrintBaseline

print("="*60)
print("DEBUG: ANALISANDO PESOS DO STN")
print("="*60)

# Carregar modelo
print("\n1. Carregando modelo...")
model = DeepPrintBaseline(num_classes=6000, texture_embedding_dims=96, minutia_embedding_dims=96)
checkpoint_path = Path("exp0_baseline/checkpoints/checkpoint_latest.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

epoch = checkpoint.get('epoch', '?')
print(f"   Modelo carregado (época {epoch})")

# Analisar pesos do STN
print("\n" + "="*60)
print("ANÁLISE DOS PESOS DO STN")
print("="*60)

stn = model.localization

print("\n📊 ÚLTIMA CAMADA DO STN (fc_loc[2] - gera theta):")
fc_final = stn.fc_loc[2]  # Linear(64, 3)

weight = fc_final.weight.data.numpy()
bias = fc_final.bias.data.numpy()

print(f"   Weight shape: {weight.shape}")
print(f"   Bias shape: {bias.shape}")

print(f"\n   Weight stats:")
print(f"   Min:  {weight.min():.6f}")
print(f"   Max:  {weight.max():.6f}")
print(f"   Mean: {weight.mean():.6f}")
print(f"   Std:  {weight.std():.6f}")
print(f"   Norm: {np.linalg.norm(weight):.6f}")

print(f"\n   Bias: {bias}")

# Verificar se pesos são ~zero (não foram treinados)
weight_is_zero = np.abs(weight).max() < 1e-6
bias_is_zero = np.abs(bias).max() < 1e-6

if weight_is_zero and bias_is_zero:
    print("\n   ❌ PROBLEMA CRÍTICO: Pesos da última camada STN são ZERO!")
    print("   → STN não foi treinado!")
    print("   → Sempre retorna [0, 0, 0] = transformação identidade")
elif weight_is_zero:
    print("\n   ❌ PROBLEMA: Weight é zero mas bias não")
    print("   → STN sempre retorna bias = transformação constante")
else:
    print("\n   ✓ Pesos foram treinados (não são zero)")
    print("   → STN deveria funcionar corretamente")

# Testar output do STN com inputs sintéticos
print("\n" + "="*60)
print("TESTE: Output do STN com inputs diferentes")
print("="*60)

stn.eval()

# Criar 3 imagens diferentes (ruído aleatório)
torch.manual_seed(42)
img1 = torch.randn(1, 1, 299, 299)
torch.manual_seed(123)
img2 = torch.randn(1, 1, 299, 299) 
torch.manual_seed(456)
img3 = torch.randn(1, 1, 299, 299)

print("\n   Passando 3 imagens DIFERENTES pelo STN...")
with torch.no_grad():
    out1 = stn(img1)
    out2 = stn(img2)
    out3 = stn(img3)

# Verificar se outputs são diferentes
diff_12 = torch.abs(out1 - out2).mean().item()
diff_13 = torch.abs(out1 - out3).mean().item()
diff_23 = torch.abs(out2 - out3).mean().item()

print(f"\n   Diferença out1 vs out2: {diff_12:.6f}")
print(f"   Diferença out1 vs out3: {diff_13:.6f}")
print(f"   Diferença out2 vs out3: {diff_23:.6f}")

if diff_12 < 1e-6 and diff_13 < 1e-6:
    print("\n   ❌ CONFIRMADO: STN retorna MESMA transformação para inputs diferentes!")
else:
    print("\n   ✓ STN retorna transformações diferentes")

# Analisar theta gerado
print("\n" + "="*60)
print("ANÁLISE DO THETA GERADO")
print("="*60)

with torch.no_grad():
    # Passar pela localization network
    resized_x1 = stn.resize(img1)
    xs1 = stn.localization(resized_x1)
    xs1_flat = xs1.view(-1, 8 * 8 * 64)
    theta_xy1 = stn.fc_loc(xs1_flat)
    
    resized_x2 = stn.resize(img2)
    xs2 = stn.localization(resized_x2)
    xs2_flat = xs2.view(-1, 8 * 8 * 64)
    theta_xy2 = stn.fc_loc(xs2_flat)

print(f"\n   Theta para img1: {theta_xy1[0].numpy()}")
print(f"   Theta para img2: {theta_xy2[0].numpy()}")

theta_diff = torch.abs(theta_xy1 - theta_xy2).mean().item()
print(f"\n   Diferença entre thetas: {theta_diff:.6f}")

if theta_diff < 1e-6:
    print(f"\n   ❌ PROBLEMA: fc_loc retorna MESMO theta para inputs diferentes!")
    print(f"   → Pesos provavelmente não foram treinados")
else:
    print(f"\n   ✓ fc_loc gera thetas diferentes")

# Verificar features antes do fc_loc
print("\n📊 Features ANTES do fc_loc:")
print(f"   xs1_flat: min={xs1_flat.min():.6f}, max={xs1_flat.max():.6f}, mean={xs1_flat.mean():.6f}")
print(f"   xs2_flat: min={xs2_flat.min():.6f}, max={xs2_flat.max():.6f}, mean={xs2_flat.mean():.6f}")

feat_diff = torch.abs(xs1_flat - xs2_flat).mean().item()
print(f"\n   Diferença entre features: {feat_diff:.6f}")

if feat_diff < 1e-6:
    print(f"\n   ❌ PROBLEMA: Convoluções do STN retornam MESMAS features!")
else:
    print(f"\n   ✓ Convoluções do STN funcionam corretamente")

print("\n" + "="*60)
print("CONCLUSÃO")
print("="*60)

if weight_is_zero and bias_is_zero:
    print("\n🚨 CAUSA RAIZ: Última camada do STN não foi treinada (pesos=0)")
    print("\nSOLUÇÕES:")
    print("1. Verificar se STN está no optimizer (parâmetros incluídos?)")
    print("2. Verificar se gradientes chegam ao STN (backward pass)")
    print("3. Considerar remover STN temporariamente para testar")
elif diff_12 < 1e-6:
    print("\n🚨 PROBLEMA: STN gera mesma transformação apesar de pesos != 0")
    print("   → Bug no forward ou problema com BatchNorm em eval()")
else:
    print("\n✓ STN parece OK com inputs sintéticos")
    print("   → Problema pode ser específico com imagens reais do dataset")

print("="*60)
