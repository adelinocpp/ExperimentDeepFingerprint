#!/usr/bin/env python3
"""
Script para calcular EER do modelo DeepPrint - SIMPLIFICADO

Uso:
    # GPU (padrão, 1000 amostras)
    python calculate_eer.py
    
    # CPU (evita conflito com treinamento)
    python calculate_eer.py --device cpu
    
    # Mais rápido (menos amostras)
    python calculate_eer.py --device cpu --samples 500
    
    # Checkpoint específico
    python calculate_eer.py --checkpoint exp0_baseline/checkpoints/checkpoint_epoch_10.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import logging

from data_loader import load_datasets
from models_base import DeepPrintBaseline

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_eer_simple(model, val_loader, device, max_samples=1000):
    """
    Calcular EER com código corrigido e simplificado
    
    Estratégia:
    1. Coletar embeddings agrupados por label
    2. Criar pares genuínos (mesma classe)
    3. Criar pares impostores (classes diferentes)
    4. Calcular EER no ponto onde FAR ≈ FRR
    """
    model.eval()
    embeddings_by_label = {}
    
    logger.info(f"Extraindo embeddings (max {max_samples} amostras)...")
    with torch.no_grad():
        total_samples = 0
        for batch_idx, batch_data in enumerate(val_loader):
            if total_samples >= max_samples:
                break
            
            if len(batch_data) == 3:
                images, batch_labels, _ = batch_data
            else:
                images, batch_labels = batch_data
            
            images = images.to(device)
            outputs = model(images)
            embedding = outputs["embedding"]
            
            batch_embeddings = embedding.cpu().numpy()
            batch_labels_np = batch_labels.numpy()
            
            for emb, lbl in zip(batch_embeddings, batch_labels_np):
                if lbl not in embeddings_by_label:
                    embeddings_by_label[lbl] = []
                embeddings_by_label[lbl].append(emb)
                total_samples += 1
                if total_samples >= max_samples:
                    break
            
            # Liberar memória
            del images, outputs, embedding
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if (batch_idx + 1) % 20 == 0:
                logger.info(f"  {total_samples}/{max_samples} amostras, {len(embeddings_by_label)} classes")
    
    logger.info(f"Total: {total_samples} amostras, {len(embeddings_by_label)} classes\n")
    
    if len(embeddings_by_label) < 2:
        logger.error("Classes insuficientes!")
        return None
    
    # Criar pares genuínos
    logger.info("Criando pares genuínos (mesma classe)...")
    genuine_scores = []
    
    for label, embs in embeddings_by_label.items():
        embs = np.array(embs)
        if len(embs) < 2:
            continue
        
        # Normalizar
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        
        # Até 10 pares por classe
        n_pairs = min(10, len(embs) * (len(embs) - 1) // 2)
        for _ in range(n_pairs):
            i, j = np.random.choice(len(embs), size=2, replace=False)
            score = np.dot(embs[i], embs[j])
            genuine_scores.append(score)
    
    logger.info(f"  {len(genuine_scores)} pares genuínos")
    
    # Criar pares impostores
    logger.info("Criando pares impostores (classes diferentes)...")
    impostor_scores = []
    labels_list = list(embeddings_by_label.keys())
    
    max_impostor_pairs = len(genuine_scores) * 3  # 3x mais impostores
    for _ in range(max_impostor_pairs):
        if len(labels_list) < 2:
            break
        lbl1, lbl2 = np.random.choice(labels_list, size=2, replace=False)
        
        embs1 = np.array(embeddings_by_label[lbl1])
        embs2 = np.array(embeddings_by_label[lbl2])
        
        embs1 = embs1 / (np.linalg.norm(embs1, axis=1, keepdims=True) + 1e-8)
        embs2 = embs2 / (np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-8)
        
        i = np.random.choice(len(embs1))
        j = np.random.choice(len(embs2))
        
        score = np.dot(embs1[i], embs2[j])
        impostor_scores.append(score)
    
    logger.info(f"  {len(impostor_scores)} pares impostores\n")
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        logger.error("Pares insuficientes!")
        return None
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    logger.info("Estatísticas:")
    logger.info(f"  Genuínos:   min={genuine_scores.min():.4f}, max={genuine_scores.max():.4f}, média={genuine_scores.mean():.4f}")
    logger.info(f"  Impostores: min={impostor_scores.min():.4f}, max={impostor_scores.max():.4f}, média={impostor_scores.mean():.4f}\n")
    
    # Calcular EER
    logger.info("Calculando EER (ponto onde FAR ≈ FRR)...")
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)
    
    best_eer = 1.0
    best_threshold = 0.0
    best_diff = 1.0
    far_at_frr_01 = 1.0
    
    for threshold in thresholds:
        # FAR = impostor aceito como genuíno
        far = np.sum(impostor_scores >= threshold) / len(impostor_scores)
        # FRR = genuíno rejeitado como impostor
        frr = np.sum(genuine_scores < threshold) / len(genuine_scores)
        
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2
            best_threshold = threshold
        
        if abs(frr - 0.1) < 0.02:
            far_at_frr_01 = far
    
    return {
        "eer": best_eer,
        "threshold": best_threshold,
        "far_at_frr_01": far_at_frr_01,
        "num_genuine": len(genuine_scores),
        "num_impostor": len(impostor_scores),
        "num_classes": len(embeddings_by_label),
    }


def main():
    parser = argparse.ArgumentParser(description='Calcular EER do modelo DeepPrint')
    parser.add_argument('--checkpoint', type=str, default='exp0_baseline/checkpoints/checkpoint_latest.pt',
                        help='Caminho para checkpoint')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='cpu (seguro, lento) ou cuda (rápido, pode dar OOM)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Número de amostras (500=rápido, 2000=preciso)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA não disponível, usando CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("CÁLCULO DE EER - SIMPLIFICADO")
    print("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Checkpoint: {args.checkpoint}\n")
    
    # Carregar dataset
    logger.info("Carregando dataset SFinge...")
    train_dataset, val_dataset, test_dataset, _ = load_datasets(
        datasets=["SFinge"],
        random_state=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment_train=False,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    logger.info(f"Val dataset: {len(val_dataset)} amostras\n")
    
    # Carregar modelo
    model_path = Path(args.checkpoint)
    if not model_path.exists():
        logger.error(f"Checkpoint não encontrado: {model_path}")
        return 1
    
    logger.info(f"Carregando modelo...")
    model = DeepPrintBaseline(
        num_classes=6000,
        texture_embedding_dims=96,
        minutia_embedding_dims=96
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    logger.info(f"Modelo carregado (época {epoch})\n")
    
    # Calcular EER
    print("="*60)
    result = calculate_eer_simple(model, val_loader, device, max_samples=args.samples)
    
    if result:
        print("="*60)
        print("RESULTADO:")
        print("="*60)
        print(f"  EER: {result['eer']:.4f} ({result['eer']*100:.2f}%)")
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  FAR@FRR=0.1: {result['far_at_frr_01']:.4f}")
        print(f"  Pares genuínos: {result['num_genuine']}")
        print(f"  Pares impostores: {result['num_impostor']}")
        print(f"  Classes: {result['num_classes']}")
        print("="*60)
        
        print("\nInterpretação:")
        eer = result['eer']
        if eer > 0.4:
            print("  ⚠️  EER muito alto - modelo praticamente aleatório")
        elif eer > 0.2:
            print("  ⚠️  EER alto - modelo ainda aprendendo")
        elif eer > 0.1:
            print("  ✓  EER razoável - modelo aprendendo bem")
        elif eer > 0.01:
            print("  ✓✓ EER bom - boa capacidade discriminativa")
        else:
            print("  ✓✓✓ EER excelente!")
        
        print(f"\nNota: EER {result['eer']:.4f} é o valor REAL do modelo na época {epoch}")
        return 0
    else:
        logger.error("Falha ao calcular EER")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
