"""Diagnóstico de embeddings para detectar colapso."""

import torch
import numpy as np
from pathlib import Path
from data_loader import load_datasets
from models_base import create_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_model(model_path: Path, datasets: list = None):
    """Diagnosticar modelo para detectar colapso de embeddings."""
    
    if datasets is None:
        datasets = ["FVC2000", "FVC2002", "FVC2004", "SD302"]
    
    # Carregar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando device: {device}")
    
    model = create_model("baseline", texture_embedding_dims=512).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    logger.info(f"Modelo carregado de: {model_path}")
    logger.info(f"Época do checkpoint: {checkpoint.get('epoch', 'N/A')}")
    
    # Carregar dataset de teste
    logger.info("Carregando datasets...")
    train_dataset, val_dataset, test_dataset, loaders = load_datasets(
        datasets=datasets,
        random_state=42,
        augment_train=False,
        image_size=(299, 299)
    )
    
    # Usar dataset de teste
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Amostras no teste: {len(test_dataset)}")
    
    # Extrair embeddings
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            embedding = outputs["embedding"]
            
            embeddings_list.append(embedding.cpu().numpy())
            labels_list.extend(labels.numpy())
            
            if batch_idx == 0:
                logger.info(f"Shape do embedding: {embedding.shape}")
                logger.info(f"Amostra do primeiro embedding: {embedding[0, :10]}")
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.array(labels_list)
    
    logger.info(f"\n{'='*60}")
    logger.info("DIAGNÓSTICO DE EMBEDDINGS")
    logger.info(f"{'='*60}")
    
    # 1. Estatísticas básicas
    logger.info(f"\n1. ESTATÍSTICAS BÁSICAS:")
    logger.info(f"   Shape: {embeddings.shape}")
    logger.info(f"   Mean: {embeddings.mean():.6f}")
    logger.info(f"   Std: {embeddings.std():.6f}")
    logger.info(f"   Min: {embeddings.min():.6f}")
    logger.info(f"   Max: {embeddings.max():.6f}")
    
    # 2. Normas L2
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"\n2. NORMAS L2 (antes da normalização):")
    logger.info(f"   Mean: {norms.mean():.6f}")
    logger.info(f"   Std: {norms.std():.6f}")
    logger.info(f"   Min: {norms.min():.6f}")
    logger.info(f"   Max: {norms.max():.6f}")
    
    # 3. Verificar se há NaN ou Inf
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    logger.info(f"\n3. VALORES INVÁLIDOS:")
    logger.info(f"   Contém NaN: {has_nan}")
    logger.info(f"   Contém Inf: {has_inf}")
    
    # 4. Variância por dimensão
    var_per_dim = embeddings.var(axis=0)
    logger.info(f"\n4. VARIÂNCIA POR DIMENSÃO:")
    logger.info(f"   Mean: {var_per_dim.mean():.6f}")
    logger.info(f"   Std: {var_per_dim.std():.6f}")
    logger.info(f"   Dimensões com var < 1e-6: {(var_per_dim < 1e-6).sum()}/{len(var_per_dim)}")
    
    # 5. Normalizar e calcular similaridades
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Pegar 10 pares aleatórios
    np.random.seed(42)
    idx = np.random.choice(len(embeddings_norm), size=min(10, len(embeddings_norm)), replace=False)
    
    logger.info(f"\n5. SIMILARIDADES COSENO (amostra de pares):")
    for i in range(len(idx)-1):
        sim = np.dot(embeddings_norm[idx[i]], embeddings_norm[idx[i+1]])
        same_label = labels[idx[i]] == labels[idx[i+1]]
        logger.info(f"   Par {i}: sim={sim:.6f}, mesma_classe={same_label}")
    
    # 6. Estatísticas de similaridade intra-classe vs inter-classe
    logger.info(f"\n6. SIMILARIDADES INTRA vs INTER-CLASSE:")
    
    # Pegar primeiras 100 amostras para não demorar muito
    n_samples = min(100, len(embeddings_norm))
    intra_sims = []
    inter_sims = []
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            sim = np.dot(embeddings_norm[i], embeddings_norm[j])
            if labels[i] == labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)
    
    if intra_sims:
        logger.info(f"   Intra-classe (genuínos):")
        logger.info(f"      Mean: {np.mean(intra_sims):.6f}")
        logger.info(f"      Std: {np.std(intra_sims):.6f}")
        logger.info(f"      Min: {np.min(intra_sims):.6f}")
        logger.info(f"      Max: {np.max(intra_sims):.6f}")
    
    if inter_sims:
        logger.info(f"   Inter-classe (impostores):")
        logger.info(f"      Mean: {np.mean(inter_sims):.6f}")
        logger.info(f"      Std: {np.std(inter_sims):.6f}")
        logger.info(f"      Min: {np.min(inter_sims):.6f}")
        logger.info(f"      Max: {np.max(inter_sims):.6f}")
    
    # 7. Verificar se todos embeddings são iguais
    logger.info(f"\n7. VERIFICAÇÃO DE COLAPSO:")
    unique_embeddings = np.unique(embeddings_norm, axis=0)
    logger.info(f"   Embeddings únicos: {len(unique_embeddings)}/{len(embeddings_norm)}")
    
    # Calcular distância máxima entre embeddings
    if len(embeddings_norm) > 1:
        sample_idx = np.random.choice(len(embeddings_norm), size=min(50, len(embeddings_norm)), replace=False)
        sample_embs = embeddings_norm[sample_idx]
        pairwise_dists = np.linalg.norm(sample_embs[:, None] - sample_embs[None, :], axis=2)
        max_dist = pairwise_dists.max()
        mean_dist = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)].mean()
        logger.info(f"   Distância L2 entre embeddings (amostra):")
        logger.info(f"      Mean: {mean_dist:.6f}")
        logger.info(f"      Max: {max_dist:.6f}")
        
        if max_dist < 1e-5:
            logger.warning("   ⚠️  COLAPSO DETECTADO: Todos embeddings são praticamente idênticos!")
    
    logger.info(f"\n{'='*60}\n")


if __name__ == "__main__":
    model_path = Path("exp0_baseline/models/best_model.pt")
    
    if not model_path.exists():
        logger.error(f"Modelo não encontrado: {model_path}")
    else:
        diagnose_model(model_path)
