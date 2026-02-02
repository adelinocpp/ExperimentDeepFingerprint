"""
DEBUG 3: Verificar se L2 normalization está correta.

Testa:
- Embeddings raw vs normalizados
- Normas L2 após normalização
- Impacto no cálculo de similaridade
"""

import torch
import numpy as np
from pathlib import Path
from models_base import DeepPrintBaseline
from data_loader import load_datasets
from config import MODEL_CONFIG
from sklearn.metrics.pairwise import cosine_similarity

def debug_normalization():
    """Verifica se normalização L2 está funcionando corretamente."""

    print("=" * 80)
    print("DEBUG 3: L2 NORMALIZATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar modelo
    print("\n1. Carregando modelo...")

    checkpoint_path = Path("exp0_baseline/models/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        num_classes = checkpoint['model_state_dict']['texture_classifier.0.weight'].shape[0]
        print(f"   ✅ Checkpoint: {num_classes} classes")

        model = DeepPrintBaseline(
            texture_embedding_dims=MODEL_CONFIG["texture_embedding_dims"]["exp0_baseline"],
            minutia_embedding_dims=MODEL_CONFIG["minutia_embedding_dims"]["exp0_baseline"],
            num_classes=num_classes,
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("   ⚠️  Sem checkpoint, usando modelo aleatório")
        num_classes = 14
        model = DeepPrintBaseline(
            texture_embedding_dims=MODEL_CONFIG["texture_embedding_dims"]["exp0_baseline"],
            minutia_embedding_dims=MODEL_CONFIG["minutia_embedding_dims"]["exp0_baseline"],
            num_classes=num_classes,
        ).to(device)

    model.eval()

    # Carregar dados
    print("\n2. Carregando dados...")
    _, _, test_dataset, _ = load_datasets(
        datasets=["SFinge"],
        sample_size=200,
        random_state=42,
    )

    # Extrair embeddings de várias amostras
    num_samples = 50
    embeddings_raw = []
    labels_list = []

    print(f"\n3. Extraindo embeddings de {num_samples} amostras...")

    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            image, label = test_dataset[i][:2]
            image = image.unsqueeze(0).to(device)

            outputs = model(image)
            embedding = outputs['embedding'].cpu().numpy()[0]

            embeddings_raw.append(embedding)
            labels_list.append(label)

    embeddings_raw = np.array(embeddings_raw)
    labels = np.array(labels_list)

    print(f"   ✅ Extraídos {len(embeddings_raw)} embeddings")
    print(f"   Shape: {embeddings_raw.shape}")

    # Análise PRÉ-normalização
    print("\n" + "=" * 80)
    print("ANÁLISE PRÉ-NORMALIZAÇÃO (raw embeddings)")
    print("=" * 80)

    norms_raw = np.linalg.norm(embeddings_raw, axis=1)

    print(f"\n1. Normas L2 RAW:")
    print(f"   Média: {norms_raw.mean():.6f}")
    print(f"   Std: {norms_raw.std():.6f}")
    print(f"   Min: {norms_raw.min():.6f}")
    print(f"   Max: {norms_raw.max():.6f}")

    # Verificar se normas são constantes
    if norms_raw.std() < 1e-6:
        print("   ✅ Normas são CONSTANTES (já normalizadas pelo modelo)")
    elif norms_raw.std() < 0.01:
        print("   ⚠️  Normas quase constantes (variação muito baixa)")
    else:
        print("   ⚠️  Normas VARIAM (modelo não normaliza internamente)")

    # Normalização manual
    print("\n2. Aplicando normalização L2 manual...")

    def normalize_manual(embeddings, epsilon=1e-6):
        """Normaliza com L2 norm."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, epsilon)  # Evitar divisão por zero
        return embeddings / norms

    embeddings_normalized = normalize_manual(embeddings_raw)

    norms_normalized = np.linalg.norm(embeddings_normalized, axis=1)

    print(f"\n   Normas L2 APÓS normalização:")
    print(f"   Média: {norms_normalized.mean():.10f}")
    print(f"   Std: {norms_normalized.std():.10f}")
    print(f"   Min: {norms_normalized.min():.10f}")
    print(f"   Max: {norms_normalized.max():.10f}")

    if np.allclose(norms_normalized, 1.0, atol=1e-5):
        print("   ✅ Normalização CORRETA (todas normas = 1.0)")
    else:
        print("   ❌ PROBLEMA: Normas não são 1.0 após normalização!")

    # Análise de similaridade
    print("\n" + "=" * 80)
    print("ANÁLISE DE SIMILARIDADE")
    print("=" * 80)

    # Similaridade coseno RAW
    print("\n1. Similaridade coseno (RAW embeddings):")
    cos_sim_raw = cosine_similarity(embeddings_raw)

    # Remover diagonal
    mask = ~np.eye(cos_sim_raw.shape[0], dtype=bool)
    off_diag_raw = cos_sim_raw[mask]

    print(f"   Média: {off_diag_raw.mean():.6f}")
    print(f"   Std: {off_diag_raw.std():.6f}")
    print(f"   Min: {off_diag_raw.min():.6f}")
    print(f"   Max: {off_diag_raw.max():.6f}")

    # Similaridade coseno NORMALIZADO
    print("\n2. Similaridade coseno (Normalized embeddings):")
    cos_sim_norm = cosine_similarity(embeddings_normalized)

    off_diag_norm = cos_sim_norm[mask]

    print(f"   Média: {off_diag_norm.mean():.6f}")
    print(f"   Std: {off_diag_norm.std():.6f}")
    print(f"   Min: {off_diag_norm.min():.6f}")
    print(f"   Max: {off_diag_norm.max():.6f}")

    # Comparar
    print("\n3. Comparação RAW vs NORMALIZED:")
    print(f"   Diferença absoluta média: {np.abs(off_diag_raw - off_diag_norm).mean():.10f}")

    if np.allclose(off_diag_raw, off_diag_norm, atol=1e-5):
        print("   ✅ Similaridades SÃO IGUAIS (modelo já normaliza internamente)")
    else:
        print("   ⚠️  Similaridades DIFEREM (normalização afeta resultado)")

    # Análise por classe
    print("\n" + "=" * 80)
    print("ANÁLISE INTRA-CLASSE vs INTER-CLASSE")
    print("=" * 80)

    unique_labels = np.unique(labels)
    print(f"\nClasses únicas: {len(unique_labels)}")

    # Calcular similaridade intra-classe e inter-classe
    intra_class_sims = []
    inter_class_sims = []

    for i in range(len(embeddings_normalized)):
        for j in range(i + 1, len(embeddings_normalized)):
            sim = cos_sim_norm[i, j]

            if labels[i] == labels[j]:
                intra_class_sims.append(sim)
            else:
                inter_class_sims.append(sim)

    print(f"\n1. Similaridade INTRA-classe (mesma classe):")
    if intra_class_sims:
        print(f"   Média: {np.mean(intra_class_sims):.6f}")
        print(f"   Std: {np.std(intra_class_sims):.6f}")
        print(f"   Min: {np.min(intra_class_sims):.6f}")
        print(f"   Max: {np.max(intra_class_sims):.6f}")
    else:
        print("   (sem pares intra-classe)")

    print(f"\n2. Similaridade INTER-classe (classes diferentes):")
    if inter_class_sims:
        print(f"   Média: {np.mean(inter_class_sims):.6f}")
        print(f"   Std: {np.std(inter_class_sims):.6f}")
        print(f"   Min: {np.min(inter_class_sims):.6f}")
        print(f"   Max: {np.max(inter_class_sims):.6f}")
    else:
        print("   (sem pares inter-classe)")

    if intra_class_sims and inter_class_sims:
        separation = np.mean(intra_class_sims) - np.mean(inter_class_sims)
        print(f"\n3. SEPARAÇÃO (intra - inter):")
        print(f"   {separation:.6f}")

        if separation < 0:
            print("   ❌ PROBLEMA: Separação NEGATIVA!")
            print("      Inter-classe > Intra-classe (inverso do esperado)")
        elif separation < 0.01:
            print("   ❌ PROBLEMA: Separação < 1%")
            print("      Quasi-collapse - embeddings não discriminativos")
        elif separation < 0.05:
            print("   ⚠️  ATENÇÃO: Separação baixa (<5%)")
        else:
            print("   ✅ Separação adequada (>5%)")

    # Verificar se normalização preserva ordem
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO: Normalização preserva ordem de similaridade?")
    print("=" * 80)

    # Pegar 10 pares aleatórios
    np.random.seed(42)
    num_pairs = 10
    pairs = []

    for _ in range(num_pairs):
        i, j = np.random.choice(len(embeddings_raw), 2, replace=False)
        sim_raw = cos_sim_raw[i, j]
        sim_norm = cos_sim_norm[i, j]
        pairs.append((i, j, sim_raw, sim_norm))

    print(f"\n{'Pair':>10} {'Raw Sim':>12} {'Norm Sim':>12} {'Diff':>12}")
    print("-" * 50)

    for i, j, sim_raw, sim_norm in pairs:
        diff = abs(sim_raw - sim_norm)
        print(f"({i:3d},{j:3d}) {sim_raw:>12.6f} {sim_norm:>12.6f} {diff:>12.8f}")

    print("\n" + "=" * 80)
    print("FIM DO DEBUG")
    print("=" * 80)


if __name__ == "__main__":
    debug_normalization()
