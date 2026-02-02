"""
DEBUG 1: Verificar se Center Loss está sendo aplicado corretamente.

Testa:
- Se Center Loss retorna valores não-zero
- Se o peso adaptativo está sendo usado
- Breakdown detalhado de cada componente da loss
"""

import torch
import numpy as np
from pathlib import Path
from models_base import DeepPrintBaseline
from data_loader import load_datasets
from config import MODEL_CONFIG, LOSS_CONFIG, get_center_loss_weight
from training import CenterLoss

def debug_loss_breakdown():
    """Verifica se cada componente da loss está funcionando."""

    print("=" * 80)
    print("DEBUG 1: BREAKDOWN DE LOSS COMPONENTS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar modelo e dados
    print("\n1. Carregando modelo e dados...")
    num_classes = 14  # Debug usa ~14 classes

    model = DeepPrintBaseline(
        texture_embedding_dims=MODEL_CONFIG["texture_embedding_dims"]["exp0_baseline"],
        minutia_embedding_dims=MODEL_CONFIG["minutia_embedding_dims"]["exp0_baseline"],
        num_classes=num_classes,
    ).to(device)

    model.train()  # IMPORTANTE: modo training

    # Carregar batch de dados
    _, _, test_dataset, _ = load_datasets(
        datasets=["SFinge"],
        sample_size=200,
        random_state=42,
    )

    # Pegar um batch
    batch_size = 8
    images = torch.stack([test_dataset[i][0] for i in range(batch_size)]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(batch_size)]).to(device)

    print(f"✅ Batch: {batch_size} amostras, {len(labels.unique())} classes únicas")

    # Forward pass
    print("\n2. Forward pass...")
    outputs = model(images)

    print(f"   Texture embedding: {outputs['texture_embedding'].shape}")
    print(f"   Minutia embedding: {outputs['minutia_embedding'].shape}")
    print(f"   Texture logits: {outputs['texture_logits'].shape}")
    print(f"   Minutia logits: {outputs['minutia_logits'].shape}")

    # Criar Center Loss
    print("\n3. Criando Center Loss...")
    center_loss_texture = CenterLoss(
        num_classes=num_classes,
        feat_dim=MODEL_CONFIG["texture_embedding_dims"]["exp0_baseline"],
        alpha=0.01
    ).to(device)

    center_loss_minutia = CenterLoss(
        num_classes=num_classes,
        feat_dim=MODEL_CONFIG["minutia_embedding_dims"]["exp0_baseline"],
        alpha=0.01
    ).to(device)

    # Calcular peso adaptativo
    center_loss_weight = get_center_loss_weight(num_classes)
    print(f"   Peso adaptativo calculado: {center_loss_weight:.10f}")
    print(f"   Peso base (paper): {LOSS_CONFIG['center_loss_base_weight']:.10f}")
    print(f"   Fator de escala: {center_loss_weight / LOSS_CONFIG['center_loss_base_weight']:.4f}x")

    # Calcular cada componente da loss
    print("\n" + "=" * 80)
    print("LOSS BREAKDOWN")
    print("=" * 80)

    # 1. Softmax Loss (Cross-Entropy)
    ce_loss_texture = torch.nn.functional.cross_entropy(outputs['texture_logits'], labels)
    ce_loss_minutia = torch.nn.functional.cross_entropy(outputs['minutia_logits'], labels)

    print(f"\n1. SOFTMAX LOSS (Cross-Entropy):")
    print(f"   Texture: {ce_loss_texture.item():.6f}")
    print(f"   Minutia: {ce_loss_minutia.item():.6f}")
    print(f"   Total: {(ce_loss_texture + ce_loss_minutia).item():.6f}")

    # 2. Center Loss
    cl_texture = center_loss_texture(outputs['texture_embedding'], labels)
    cl_minutia = center_loss_minutia(outputs['minutia_embedding'], labels)

    print(f"\n2. CENTER LOSS (RAW - antes do peso):")
    print(f"   Texture: {cl_texture.item():.6f}")
    print(f"   Minutia: {cl_minutia.item():.6f}")
    print(f"   Total: {(cl_texture + cl_minutia).item():.6f}")

    print(f"\n3. CENTER LOSS (WEIGHTED - após aplicar peso {center_loss_weight:.10f}):")
    cl_texture_weighted = center_loss_weight * cl_texture
    cl_minutia_weighted = center_loss_weight * cl_minutia
    print(f"   Texture: {cl_texture_weighted.item():.10f}")
    print(f"   Minutia: {cl_minutia_weighted.item():.10f}")
    print(f"   Total: {(cl_texture_weighted + cl_minutia_weighted).item():.10f}")

    # 3. Total Loss (como seria calculado no treinamento)
    total_loss = (
        LOSS_CONFIG["softmax_loss_weight"] * (ce_loss_texture + ce_loss_minutia) +
        center_loss_weight * (cl_texture + cl_minutia)
    )

    print(f"\n4. TOTAL LOSS:")
    print(f"   Softmax contribution: {LOSS_CONFIG['softmax_loss_weight'] * (ce_loss_texture + ce_loss_minutia).item():.6f}")
    print(f"   Center contribution: {(center_loss_weight * (cl_texture + cl_minutia)).item():.10f}")
    print(f"   TOTAL: {total_loss.item():.6f}")

    # Análise
    print("\n" + "=" * 80)
    print("ANÁLISE")
    print("=" * 80)

    center_contrib = (center_loss_weight * (cl_texture + cl_minutia)).item()
    softmax_contrib = (LOSS_CONFIG["softmax_loss_weight"] * (ce_loss_texture + ce_loss_minutia)).item()

    print(f"\n1. Proporção Center/Softmax: {center_contrib / softmax_contrib * 100:.6f}%")

    if center_contrib < 1e-6:
        print("   ❌ PROBLEMA: Center Loss contribuição QUASE ZERO!")
        print(f"      Center Loss está tendo IMPACTO MÍNIMO no treinamento")
    elif center_contrib < 0.001:
        print("   ⚠️  ATENÇÃO: Center Loss contribuição muito baixa (<0.1%)")
        print(f"      Pode não ser suficiente para criar separação")
    else:
        print("   ✅ Center Loss tem contribuição significativa")

    print(f"\n2. Softmax domina? {softmax_contrib / total_loss.item() * 100:.2f}% da loss total")

    if softmax_contrib / total_loss.item() > 0.99:
        print("   ❌ PROBLEMA: Softmax domina >99% da loss!")
        print("      Modelo está focando apenas em classificação, não em embeddings discriminativos")

    # Verificar se gradientes fluem
    print("\n3. Testando gradientes...")
    total_loss.backward()

    # Verificar gradientes dos embeddings
    texture_emb_grad = outputs['texture_embedding'].grad
    minutia_emb_grad = outputs['minutia_embedding'].grad

    if texture_emb_grad is not None:
        print(f"   Texture embedding grad: norm={texture_emb_grad.norm().item():.6f}")
    else:
        print("   ❌ Texture embedding grad: None!")

    if minutia_emb_grad is not None:
        print(f"   Minutia embedding grad: norm={minutia_emb_grad.norm().item():.6f}")
    else:
        print("   ❌ Minutia embedding grad: None!")

    # Verificar centros do Center Loss
    print("\n4. Verificando centros do Center Loss...")
    centers_texture_norm = center_loss_texture.centers.norm(dim=1).mean()
    centers_minutia_norm = center_loss_minutia.centers.norm(dim=1).mean()

    print(f"   Texture centers norm média: {centers_texture_norm.item():.6f}")
    print(f"   Minutia centers norm média: {centers_minutia_norm.item():.6f}")

    if centers_texture_norm < 0.1 or centers_minutia_norm < 0.1:
        print("   ⚠️  Centros têm norma muito baixa - podem estar colapsando!")

    print("\n" + "=" * 80)
    print("FIM DO DEBUG")
    print("=" * 80)


if __name__ == "__main__":
    debug_loss_breakdown()
