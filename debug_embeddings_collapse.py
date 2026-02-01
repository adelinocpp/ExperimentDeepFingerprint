"""
Script para debugar colapso de embeddings.

Verifica se embeddings são realmente idênticos ou apenas muito similares.
"""

import torch
import numpy as np
from pathlib import Path
from models_base import create_model
from data_loader import load_datasets
from config import MODEL_CONFIG

def inspect_embeddings():
    """Inspeciona embeddings para detectar colapso."""

    print("=" * 80)
    print("DEBUG: Inspeção de Embeddings")
    print("=" * 80)

    # Carregar modelo
    print("\n1. Carregando modelo...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models_base import DeepPrintBaseline

    # Primeiro, carregar checkpoint para descobrir número de classes
    checkpoint_path = Path("exp0_baseline/models/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        num_classes = checkpoint['model_state_dict']['texture_classifier.0.weight'].shape[0]
        print(f"✅ Checkpoint detectado: {num_classes} classes")
    else:
        num_classes = 3
        print("⚠️  Checkpoint não encontrado, usando 3 classes")

    model = DeepPrintBaseline(
        texture_embedding_dims=MODEL_CONFIG["texture_embedding_dims"]["exp0_baseline"],
        minutia_embedding_dims=MODEL_CONFIG["minutia_embedding_dims"]["exp0_baseline"],
        num_classes=num_classes,
    )

    # Carregar pesos do checkpoint
    if checkpoint_path.exists():
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint carregado: epoch {checkpoint.get('epoch', '?')}")
    else:
        print("⚠️  Usando modelo sem treino (inicialização aleatória)")

    model.to(device)
    model.eval()

    # Carregar dados (usar mesmo sample_size do treinamento)
    print("\n2. Carregando dados de teste...")
    _, _, test_dataset, _ = load_datasets(
        datasets=["SFinge"],
        sample_size=200,  # Mesmo tamanho do treinamento debug
        random_state=42,
    )

    # Extrair embeddings
    print("\n3. Extraindo embeddings...")
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(min(20, len(test_dataset))):  # Primeiros 20 samples
            sample = test_dataset[i]
            if len(sample) == 3:
                image, label, _ = sample
            else:
                image, label = sample

            image = image.unsqueeze(0).to(device)
            outputs = model(image)
            embedding = outputs["embedding"].cpu().numpy()[0]

            embeddings_list.append(embedding)
            labels_list.append(label)

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)

    print(f"✅ Extraídos {len(embeddings)} embeddings")

    # Análise estatística
    print("\n" + "=" * 80)
    print("ANÁLISE ESTATÍSTICA")
    print("=" * 80)

    # 1. Normas L2
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n1. Normas L2:")
    print(f"   Média: {norms.mean():.6f}")
    print(f"   Std:   {norms.std():.6f}")
    print(f"   Min:   {norms.min():.6f}")
    print(f"   Max:   {norms.max():.6f}")

    # 2. Valores únicos
    unique_vecs = np.unique(embeddings, axis=0)
    print(f"\n2. Vetores únicos: {len(unique_vecs)} de {len(embeddings)}")
    if len(unique_vecs) == 1:
        print("   ❌ TODOS OS EMBEDDINGS SÃO IDÊNTICOS!")
    elif len(unique_vecs) < len(embeddings) // 2:
        print(f"   ⚠️  Muitos embeddings duplicados!")

    # 3. Variância por dimensão
    var_per_dim = embeddings.var(axis=0)
    print(f"\n3. Variância por dimensão:")
    print(f"   Média: {var_per_dim.mean():.8f}")
    print(f"   Min:   {var_per_dim.min():.8f}")
    print(f"   Max:   {var_per_dim.max():.8f}")
    print(f"   Dims com var=0: {(var_per_dim == 0).sum()} de {len(var_per_dim)}")

    # 4. Similaridade coseno
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = cosine_similarity(embeddings)

    # Separar diagonal (self-similarity = 1.0)
    mask = ~np.eye(cos_sim.shape[0], dtype=bool)
    off_diagonal = cos_sim[mask]

    print(f"\n4. Similaridade Coseno (pares diferentes):")
    print(f"   Média: {off_diagonal.mean():.6f}")
    print(f"   Std:   {off_diagonal.std():.6f}")
    print(f"   Min:   {off_diagonal.min():.6f}")
    print(f"   Max:   {off_diagonal.max():.6f}")

    if off_diagonal.std() < 1e-6:
        print("   ❌ STD ≈ 0: Todos os pares têm mesma similaridade!")

    # 5. Primeiros 5 embeddings (valores brutos)
    print(f"\n5. Primeiros 5 embeddings (primeiras 10 dims):")
    for i in range(min(5, len(embeddings))):
        print(f"   [{i}] label={labels[i]}: {embeddings[i][:10]}")

    # 6. Análise por classe
    print(f"\n6. Análise por classe:")
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        class_embs = embeddings[mask]
        if len(class_embs) > 1:
            intra_sim = cosine_similarity(class_embs)
            intra_mask = ~np.eye(intra_sim.shape[0], dtype=bool)
            intra_off_diag = intra_sim[intra_mask]
            print(f"   Classe {label}: {len(class_embs)} amostras, "
                  f"similaridade intra-classe={intra_off_diag.mean():.6f}±{intra_off_diag.std():.6f}")

    print("\n" + "=" * 80)
    print("FIM DA ANÁLISE")
    print("=" * 80)

if __name__ == "__main__":
    inspect_embeddings()
