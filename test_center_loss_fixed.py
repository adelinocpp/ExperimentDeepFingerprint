"""
Script de teste rápido: Validar correção do Center Loss weight.

Compara:
- ANTES: Center Loss adaptativo (0.000018) → 0.012% da loss total
- DEPOIS: Center Loss fixo (0.00125) → esperado ~10-30% da loss total

Execução rápida: ~30 segundos
"""

import torch
import numpy as np
from pathlib import Path
from models_base import DeepPrintBaseline
from data_loader import load_datasets
from config import MODEL_CONFIG, LOSS_CONFIG, get_center_loss_weight
from training import CenterLoss

def test_center_loss_contribution():
    """Testa se Center Loss tem contribuição significativa após correção."""

    print("=" * 80)
    print("TESTE: CENTER LOSS WEIGHT - ANTES vs DEPOIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar dados
    print("\n1. Carregando dados...")
    _, _, test_dataset, _ = load_datasets(
        datasets=["SFinge"],
        sample_size=200,
        random_state=42,
    )

    # Pegar um batch com múltiplas classes
    # Garantir diversidade pegando amostras espaçadas
    batch_size = 16
    indices = list(range(0, min(len(test_dataset), 200), 200 // batch_size))[:batch_size]
    images = torch.stack([test_dataset[i][0] for i in indices]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in indices]).to(device)

    num_classes = len(labels.unique())
    print(f"   ✅ Batch: {batch_size} amostras, {num_classes} classes únicas")

    if num_classes < 2:
        print(f"   ⚠️  Apenas 1 classe - usando todas classes disponíveis no dataset")
        # Pegar pelo menos 2 amostras de cada classe
        from collections import defaultdict
        class_samples = defaultdict(list)
        for idx in range(len(test_dataset)):
            label = test_dataset[idx][1]
            class_samples[label].append(idx)

        indices = []
        for label, samples in class_samples.items():
            indices.extend(samples[:2])  # 2 amostras por classe
            if len(indices) >= 16:
                break

        indices = indices[:16]
        images = torch.stack([test_dataset[i][0] for i in indices]).to(device)
        labels = torch.tensor([test_dataset[i][1] for i in indices]).to(device)
        num_classes = len(labels.unique())
        print(f"   ✅ Ajustado: {len(images)} amostras, {num_classes} classes únicas")

    # Criar modelo
    print("\n2. Criando modelo...")
    model = DeepPrintBaseline(
        texture_embedding_dims=MODEL_CONFIG["texture_embedding_dims"]["exp0_baseline"],
        minutia_embedding_dims=MODEL_CONFIG["minutia_embedding_dims"]["exp0_baseline"],
        num_classes=num_classes,
    ).to(device)

    model.train()

    # Forward pass
    print("\n3. Forward pass...")
    outputs = model(images)

    # Criar Center Loss
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

    # Calcular losses
    ce_loss_texture = torch.nn.functional.cross_entropy(outputs['texture_logits'], labels)
    ce_loss_minutia = torch.nn.functional.cross_entropy(outputs['minutia_logits'], labels)

    cl_texture = center_loss_texture(outputs['texture_embedding'], labels)
    cl_minutia = center_loss_minutia(outputs['minutia_embedding'], labels)

    # ANTES: Peso adaptativo (simulado)
    print("\n" + "=" * 80)
    print("COMPARAÇÃO: ANTES (adaptativo) vs DEPOIS (fixo)")
    print("=" * 80)

    # Simular peso adaptativo
    adaptive_weight = 0.00125 * (num_classes / 6000) ** 0.7

    # Calcular losses ANTES (adaptativo)
    center_contrib_before = adaptive_weight * (cl_texture + cl_minutia).item()
    softmax_contrib = (ce_loss_texture + ce_loss_minutia).item()
    total_before = softmax_contrib + center_contrib_before

    print(f"\n📉 ANTES (Center Loss adaptativo):")
    print(f"   Peso adaptativo: {adaptive_weight:.10f}")
    print(f"   Center Loss RAW: {(cl_texture + cl_minutia).item():.6f}")
    print(f"   Center Loss WEIGHTED: {center_contrib_before:.10f}")
    print(f"   Softmax Loss: {softmax_contrib:.6f}")
    print(f"   Total Loss: {total_before:.6f}")

    if softmax_contrib > 0:
        proportion_before = center_contrib_before / softmax_contrib * 100
        print(f"   Proporção Center/Softmax: {proportion_before:.6f}%")
        if proportion_before < 0.1:
            print("   ❌ CENTER LOSS < 0.1% - PRATICAMENTE DESABILITADO!")
    else:
        print("   ⚠️  Softmax Loss = 0 (batch com classe única?)")

    # DEPOIS: Peso fixo do paper
    fixed_weight = get_center_loss_weight(num_classes)  # Agora retorna 0.00125 (fixo)

    center_contrib_after = fixed_weight * (cl_texture + cl_minutia).item()
    total_after = softmax_contrib + center_contrib_after

    print(f"\n📈 DEPOIS (Center Loss fixo do paper):")
    print(f"   Peso fixo: {fixed_weight:.10f}")
    print(f"   Center Loss RAW: {(cl_texture + cl_minutia).item():.6f}")
    print(f"   Center Loss WEIGHTED: {center_contrib_after:.10f}")
    print(f"   Softmax Loss: {softmax_contrib:.6f}")
    print(f"   Total Loss: {total_after:.6f}")

    if softmax_contrib > 0:
        proportion_after = center_contrib_after / softmax_contrib * 100
        print(f"   Proporção Center/Softmax: {proportion_after:.6f}%")
    else:
        print("   ⚠️  Softmax Loss = 0 (batch com classe única?)")
        proportion_after = 0

    # Análise
    print("\n" + "=" * 80)
    print("ANÁLISE DA CORREÇÃO")
    print("=" * 80)

    if center_contrib_before > 0:
        improvement_factor = center_contrib_after / center_contrib_before
        print(f"\n✅ Melhoria:")
        print(f"   Center Loss contribuição aumentou {improvement_factor:.1f}x")

    if softmax_contrib > 0:
        print(f"   De {center_contrib_before / softmax_contrib * 100:.6f}% → {center_contrib_after / softmax_contrib * 100:.2f}%")
        proportion_pct = center_contrib_after / softmax_contrib * 100
    else:
        print(f"   ⚠️  Softmax Loss = 0, não é possível calcular proporção")
        proportion_pct = 0

    # Expectativa

    if proportion_pct < 0.1:
        print(f"\n   ❌ AINDA BAIXO: Center Loss < 0.1% da Softmax")
        print(f"      Pode não ser suficiente para treinar embeddings discriminativos")
    elif proportion_pct < 1.0:
        print(f"\n   ⚠️  FRACO: Center Loss < 1% da Softmax")
        print(f"      Pode funcionar mas embeddings podem convergir lentamente")
    elif proportion_pct < 10:
        print(f"\n   ✅ MODERADO: Center Loss {proportion_pct:.2f}% da Softmax")
        print(f"      Balanço razoável - esperado para embedding learning")
    elif proportion_pct < 50:
        print(f"\n   ✅ FORTE: Center Loss {proportion_pct:.2f}% da Softmax")
        print(f"      Center Loss terá impacto significativo nos embeddings")
    else:
        print(f"\n   ⚠️  MUITO FORTE: Center Loss {proportion_pct:.2f}% da Softmax")
        print(f"      Risco de convergência prematura - monitorar colapso")

    print("\n" + "=" * 80)
    print("CONCLUSÃO")
    print("=" * 80)

    if proportion_pct >= 1.0:
        print("\n✅ CORREÇÃO VÁLIDA!")
        print("   Center Loss agora tem contribuição significativa (>1%)")
        print("   Modelo deve treinar embeddings discriminativos")
        print("\n📋 Próximos passos:")
        print("   1. Rodar treinamento debug: python run_experiment.py --mode debug --experiment exp0_baseline")
        print("   2. Verificar EER melhora significativamente")
        print("   3. Verificar separação genuínos vs impostores aumenta")
    else:
        print("\n❌ PROBLEMA PERSISTE!")
        print(f"   Center Loss ainda muito fraco ({proportion_pct:.6f}%)")
        print("   Investigar outras causas do quasi-collapse")


if __name__ == "__main__":
    test_center_loss_contribution()
