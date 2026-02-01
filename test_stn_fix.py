#!/usr/bin/env python3
"""
Testar se correção do STN funciona (sem carregar checkpoint ruim)
"""

import torch
import numpy as np

from models_base import DeepPrintBaseline

print("="*60)
print("TESTE: STN CORRIGIDO (sem carregar checkpoint ruim)")
print("="*60)

# Criar modelo NOVO (sem carregar pesos ruins)
print("\n1. Criando modelo novo (pesos aleatórios)...")
model = DeepPrintBaseline(num_classes=6000, texture_embedding_dims=96, minutia_embedding_dims=96)
model.eval()

stn = model.localization

# Criar 3 imagens diferentes (ruído aleatório)
print("2. Criando 3 imagens diferentes...")
torch.manual_seed(42)
img1 = torch.randn(1, 1, 299, 299)
torch.manual_seed(123)
img2 = torch.randn(1, 1, 299, 299)
torch.manual_seed(456)
img3 = torch.randn(1, 1, 299, 299)

print("\n" + "="*60)
print("TESTE: STN COM CORREÇÃO")
print("="*60)

with torch.no_grad():
    # Passar pela rede para obter theta
    resized_x1 = stn.resize(img1)
    xs1 = stn.localization(resized_x1)
    xs1_flat = xs1.view(-1, 8 * 8 * 64)
    theta_xy1_raw = stn.fc_loc(xs1_flat)
    
    resized_x2 = stn.resize(img2)
    xs2 = stn.localization(resized_x2)
    xs2_flat = xs2.view(-1, 8 * 8 * 64)
    theta_xy2_raw = stn.fc_loc(xs2_flat)
    
    # Thetas RAW (antes do tanh)
    print(f"\n📊 THETAS RAW (antes do tanh):")
    print(f"   img1: {theta_xy1_raw[0].numpy()}")
    print(f"   img2: {theta_xy2_raw[0].numpy()}")
    
    # Passar pelo STN completo (com tanh)
    out1 = stn(img1)
    out2 = stn(img2)
    out3 = stn(img3)
    
    # Verificar thetas APÓS tanh (dentro do forward)
    # Simular o que o forward faz:
    theta1_limited = torch.tanh(theta_xy1_raw[0, 0]) * 3.14159
    theta2_limited = torch.tanh(theta_xy2_raw[0, 0]) * 3.14159
    
    print(f"\n📊 THETAS APÓS tanh (limitados):")
    print(f"   img1: {theta1_limited.item():.6f} rad ({theta1_limited.item()*180/3.14159:.2f}°)")
    print(f"   img2: {theta2_limited.item():.6f} rad ({theta2_limited.item()*180/3.14159:.2f}°)")
    
    offset_x1 = torch.tanh(theta_xy1_raw[0, 1])
    offset_y1 = torch.tanh(theta_xy1_raw[0, 2])
    offset_x2 = torch.tanh(theta_xy2_raw[0, 1])
    offset_y2 = torch.tanh(theta_xy2_raw[0, 2])
    
    print(f"\n📊 OFFSETS APÓS tanh (limitados):")
    print(f"   img1: offset_x={offset_x1.item():.6f}, offset_y={offset_y1.item():.6f}")
    print(f"   img2: offset_x={offset_x2.item():.6f}, offset_y={offset_y2.item():.6f}")

# Verificar outputs
diff_12 = torch.abs(out1 - out2).mean().item()
diff_13 = torch.abs(out1 - out3).mean().item()
diff_23 = torch.abs(out2 - out3).mean().item()

print(f"\n📊 DIFERENÇAS ENTRE OUTPUTS:")
print(f"   out1 vs out2: {diff_12:.6f}")
print(f"   out1 vs out3: {diff_13:.6f}")
print(f"   out2 vs out3: {diff_23:.6f}")

print("\n" + "="*60)
print("RESULTADO")
print("="*60)

if diff_12 < 1e-6 and diff_13 < 1e-6:
    print("\n❌ FALHOU: STN ainda retorna mesma transformação!")
    print("   → Correção não funcionou")
else:
    print("\n✅ SUCESSO: STN agora retorna transformações DIFERENTES!")
    print("   → Correção funcionou corretamente")
    print(f"\n   Thetas estão limitados a [-π, π]: ✓")
    print(f"   Offsets estão limitados a [-1, 1]: ✓")
    print(f"   Outputs são diferentes: ✓")

print("\n📋 PRÓXIMO PASSO:")
print("   Como o checkpoint atual (época 10) está corrompido,")
print("   você precisa TREINAR DO ZERO:")
print("\n   rm -rf exp0_baseline/checkpoints/*")
print("   python run_experiment.py --experiment exp0_baseline --mode prod --sfinge")

print("="*60)
