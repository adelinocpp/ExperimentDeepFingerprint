#!/usr/bin/env python3
"""Calcular EER na CPU com código corrigido"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data_loader import load_datasets
from models_base import DeepPrintBaseline

print("="*60)
print("CALCULANDO EER COM CÓDIGO CORRIGIDO (CPU)")
print("="*60 + "\n")

device = torch.device("cpu")
print(f"Device: {device}\n")

# Carregar dataset
print("Carregando dataset SFinge...")
train_dataset, val_dataset, test_dataset, _ = load_datasets(
    datasets=["SFinge"],
    random_state=42,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    augment_train=False,
    aggressive_augment=False,
)

val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=2)
print(f"Val dataset: {len(val_dataset)} amostras\n")

# Carregar modelo
model_path = Path("exp0_baseline/checkpoints/checkpoint_latest.pt")
print(f"Carregando modelo de {model_path}...")

model = DeepPrintBaseline(
    num_classes=6000,
    texture_embedding_dims=96,
    minutia_embedding_dims=96
)

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)
model.eval()

print(f"Modelo carregado (época {checkpoint.get('epoch', '?')})\n")

# Extrair embeddings (CÓDIGO CORRIGIDO)
print("Extraindo embeddings (max 2000 amostras)...")
embeddings_by_label = {}
max_samples = 2000

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
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Processados {total_samples} amostras, {len(embeddings_by_label)} classes...")

print(f"\nTotal: {total_samples} amostras, {len(embeddings_by_label)} classes")

if len(embeddings_by_label) < 2:
    print("ERRO: Classes insuficientes!")
    exit(1)

# Criar pares genuínos e impostores (CÓDIGO CORRIGIDO)
print("\nCriando pares genuínos...")
genuine_scores = []
impostor_scores = []

labels_list = list(embeddings_by_label.keys())

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

print(f"Pares genuínos: {len(genuine_scores)}")

print("Criando pares impostores...")
max_impostor_pairs = len(genuine_scores) * 3
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

print(f"Pares impostores: {len(impostor_scores)}\n")

if len(genuine_scores) == 0 or len(impostor_scores) == 0:
    print("ERRO: Pares insuficientes!")
    exit(1)

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print("Estatísticas dos scores:")
print(f"  Genuínos:   min={genuine_scores.min():.4f}, max={genuine_scores.max():.4f}, mean={genuine_scores.mean():.4f}")
print(f"  Impostores: min={impostor_scores.min():.4f}, max={impostor_scores.max():.4f}, mean={impostor_scores.mean():.4f}")

# Calcular EER (CÓDIGO CORRIGIDO)
print("\nCalculando EER...")
all_scores = np.concatenate([genuine_scores, impostor_scores])
thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)

best_eer = 1.0
best_threshold = 0.0
best_diff = 1.0
far_at_frr_01 = 1.0

for threshold in thresholds:
    far = np.sum(impostor_scores >= threshold) / len(impostor_scores)
    frr = np.sum(genuine_scores < threshold) / len(genuine_scores)
    
    diff = abs(far - frr)
    if diff < best_diff:
        best_diff = diff
        best_eer = (far + frr) / 2
        best_threshold = threshold
    
    if abs(frr - 0.1) < 0.02:
        far_at_frr_01 = far

print("\n" + "="*60)
print("RESULTADO:")
print("="*60)
print(f"  EER: {best_eer:.4f} ({best_eer*100:.2f}%)")
print(f"  Threshold: {best_threshold:.4f}")
print(f"  FAR@FRR=0.1: {far_at_frr_01:.4f}")
print(f"  Pares genuínos: {len(genuine_scores)}")
print(f"  Pares impostores: {len(impostor_scores)}")
print(f"  Classes: {len(embeddings_by_label)}")
print("="*60)

print(f"\nInterpretação:")
if best_eer > 0.4:
    print("  ⚠️  EER muito alto - modelo praticamente aleatório")
elif best_eer > 0.2:
    print("  ⚠️  EER alto - modelo ainda está aprendendo")
elif best_eer > 0.1:
    print("  ✓  EER razoável - modelo aprendendo bem")
elif best_eer > 0.01:
    print("  ✓✓ EER bom - modelo com boa capacidade discriminativa")
else:
    print("  ✓✓✓ EER excelente - modelo muito bom!")

print(f"\nNota: O EER=1.0 reportado no log era devido a bug no cálculo.")
print(f"Este valor ({best_eer:.4f}) é o EER REAL do modelo na época {checkpoint.get('epoch', '?')}.")
