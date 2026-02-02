"""
DEBUG 2: Verificar se gradientes estão fluindo corretamente.

Testa:
- Gradientes em cada camada
- Gradientes zerando ou explodindo
- Backward pass funcionando
"""

import torch
import numpy as np
from pathlib import Path
from models_base import DeepPrintBaseline
from data_loader import load_datasets
from config import MODEL_CONFIG, get_center_loss_weight
from training import CenterLoss

def debug_gradient_flow():
    """Verifica se gradientes estão fluindo por todas as camadas."""

    print("=" * 80)
    print("DEBUG 2: GRADIENT FLOW")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar modelo
    print("\n1. Carregando modelo...")
    num_classes = 14

    model = DeepPrintBaseline(
        texture_embedding_dims=MODEL_CONFIG["texture_embedding_dims"]["exp0_baseline"],
        minutia_embedding_dims=MODEL_CONFIG["minutia_embedding_dims"]["exp0_baseline"],
        num_classes=num_classes,
    ).to(device)

    model.train()

    # Carregar dados
    _, _, test_dataset, _ = load_datasets(
        datasets=["SFinge"],
        sample_size=200,
        random_state=42,
    )

    batch_size = 8
    images = torch.stack([test_dataset[i][0] for i in range(batch_size)]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(batch_size)]).to(device)

    print(f"✅ Batch: {batch_size} amostras")

    # Forward pass
    print("\n2. Forward pass...")
    outputs = model(images)

    # Calcular loss COMPLETA
    print("\n3. Calculando loss completa...")

    # Softmax
    ce_loss_texture = torch.nn.functional.cross_entropy(outputs['texture_logits'], labels)
    ce_loss_minutia = torch.nn.functional.cross_entropy(outputs['minutia_logits'], labels)

    # Center Loss
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

    cl_texture = center_loss_texture(outputs['texture_embedding'], labels)
    cl_minutia = center_loss_minutia(outputs['minutia_embedding'], labels)

    center_loss_weight = get_center_loss_weight(num_classes)

    total_loss = (
        1.0 * (ce_loss_texture + ce_loss_minutia) +
        center_loss_weight * (cl_texture + cl_minutia)
    )

    print(f"   Total loss: {total_loss.item():.6f}")

    # Backward pass
    print("\n4. Backward pass...")
    total_loss.backward()
    print("   ✅ Backward concluído")

    # Analisar gradientes de cada camada
    print("\n" + "=" * 80)
    print("ANÁLISE DE GRADIENTES POR CAMADA")
    print("=" * 80)

    gradient_stats = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()
            grad_min = param.grad.abs().min().item()

            gradient_stats.append({
                "name": name,
                "norm": grad_norm,
                "mean": grad_mean,
                "std": grad_std,
                "max": grad_max,
                "min": grad_min,
                "shape": tuple(param.shape),
            })
        else:
            gradient_stats.append({
                "name": name,
                "norm": None,
                "mean": None,
                "std": None,
                "max": None,
                "min": None,
                "shape": tuple(param.shape),
            })

    # Mostrar gradientes
    print(f"\n{'Layer':<50} {'Grad Norm':>12} {'Mean':>10} {'Max':>10}")
    print("-" * 80)

    for stats in gradient_stats:
        name = stats['name']
        if len(name) > 47:
            name = "..." + name[-44:]

        if stats['norm'] is not None:
            print(f"{name:<50} {stats['norm']:>12.6f} {stats['mean']:>10.6f} {stats['max']:>10.6f}")
        else:
            print(f"{name:<50} {'None':>12} {'N/A':>10} {'N/A':>10}")

    # Análise de problemas
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO DE PROBLEMAS")
    print("=" * 80)

    # 1. Gradientes zerados
    none_grads = [s for s in gradient_stats if s['norm'] is None]
    if none_grads:
        print(f"\n❌ PROBLEMA 1: {len(none_grads)} camadas SEM gradientes:")
        for s in none_grads[:5]:  # Mostrar primeiras 5
            print(f"   - {s['name']}")
        if len(none_grads) > 5:
            print(f"   ... e mais {len(none_grads) - 5} camadas")

    # 2. Gradientes muito pequenos (vanishing)
    small_grads = [s for s in gradient_stats if s['norm'] is not None and s['norm'] < 1e-7]
    if small_grads:
        print(f"\n⚠️  PROBLEMA 2: {len(small_grads)} camadas com gradientes MUITO PEQUENOS (<1e-7):")
        for s in small_grads[:5]:
            print(f"   - {s['name']}: norm={s['norm']:.2e}")
        if len(small_grads) > 5:
            print(f"   ... e mais {len(small_grads) - 5} camadas")

    # 3. Gradientes muito grandes (exploding)
    large_grads = [s for s in gradient_stats if s['norm'] is not None and s['norm'] > 100]
    if large_grads:
        print(f"\n⚠️  PROBLEMA 3: {len(large_grads)} camadas com gradientes MUITO GRANDES (>100):")
        for s in large_grads[:5]:
            print(f"   - {s['name']}: norm={s['norm']:.2e}")
        if len(large_grads) > 5:
            print(f"   ... e mais {len(large_grads) - 5} camadas")

    # 4. Estatísticas gerais
    valid_norms = [s['norm'] for s in gradient_stats if s['norm'] is not None]
    if valid_norms:
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   Camadas com gradientes: {len(valid_norms)}/{len(gradient_stats)}")
        print(f"   Grad norm média: {np.mean(valid_norms):.6f}")
        print(f"   Grad norm std: {np.std(valid_norms):.6f}")
        print(f"   Grad norm min: {np.min(valid_norms):.6f}")
        print(f"   Grad norm max: {np.max(valid_norms):.6f}")

    # 5. Verificar camadas específicas críticas
    print(f"\n🔍 CAMADAS CRÍTICAS:")

    critical_layers = [
        "texture_embedding",
        "minutia_embedding",
        "texture_classifier",
        "minutia_classifier",
    ]

    for layer_name in critical_layers:
        matching = [s for s in gradient_stats if layer_name in s['name']]
        if matching:
            avg_norm = np.mean([s['norm'] for s in matching if s['norm'] is not None])
            print(f"   {layer_name}: {len(matching)} params, avg norm={avg_norm:.6f}")
        else:
            print(f"   {layer_name}: não encontrado")

    # 6. Ratio de gradientes embedding vs classificador
    print(f"\n📈 RATIO: Embedding vs Classificador")

    emb_grads = [s for s in gradient_stats if 'embedding' in s['name'] and s['norm'] is not None]
    cls_grads = [s for s in gradient_stats if 'classifier' in s['name'] and s['norm'] is not None]

    if emb_grads and cls_grads:
        avg_emb = np.mean([s['norm'] for s in emb_grads])
        avg_cls = np.mean([s['norm'] for s in cls_grads])

        print(f"   Embedding layers: avg={avg_emb:.6f}")
        print(f"   Classifier layers: avg={avg_cls:.6f}")
        print(f"   Ratio (emb/cls): {avg_emb / avg_cls:.4f}")

        if avg_emb / avg_cls < 0.01:
            print("   ❌ PROBLEMA: Gradientes de embedding são 100x menores que classificador!")
            print("      Embeddings não estão aprendendo adequadamente")

    print("\n" + "=" * 80)
    print("FIM DO DEBUG")
    print("=" * 80)


if __name__ == "__main__":
    debug_gradient_flow()
