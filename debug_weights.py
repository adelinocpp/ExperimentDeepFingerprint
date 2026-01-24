#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, '.')
from models_base import DeepPrintBaseline

def analyze_weights():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar checkpoint
    checkpoint = torch.load("exp0_baseline/models/best_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    print("=== ANÁLISE DOS PESOS DO MODELO TREINADO ===\n")
    
    # Analisar pesos do Stem
    print("STEM WEIGHTS:")
    stem_keys = [k for k in state_dict.keys() if k.startswith('stem.')]
    
    for key in stem_keys[:15]:  # Primeiras 15 camadas
        w = state_dict[key]
        if w.numel() > 0:
            mean = w.mean().item()
            std = w.std().item()
            min_val = w.min().item()
            max_val = w.max().item()
            zeros = (w == 0).sum().item() / w.numel() * 100
            
            status = ""
            if std < 1e-8:
                status = " ⚠️ COLLAPSED!"
            elif zeros > 90:
                status = " ⚠️ >90% ZEROS!"
            
            print(f"  {key}: mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}, zeros={zeros:.1f}%{status}")
    
    # Comparar com modelo novo
    print("\n\nCOMPARAÇÃO COM MODELO NOVO:")
    model_new = DeepPrintBaseline(texture_embedding_dims=512, num_classes=2120)
    
    new_state = model_new.state_dict()
    for key in stem_keys[:5]:
        if key in new_state:
            w_trained = state_dict[key]
            w_new = new_state[key]
            
            print(f"\n{key}:")
            print(f"  Novo:     mean={w_new.mean():.6f}, std={w_new.std():.6f}")
            print(f"  Treinado: mean={w_trained.mean():.6f}, std={w_trained.std():.6f}")
            
            # Verificar se são muito diferentes
            if w_trained.std() < 1e-6 and w_new.std() > 1e-3:
                print("  ⚠️ PESO COLAPSOU DURANTE TREINAMENTO!")

    # Verificar BatchNorm
    print("\n\nBATCHNORM STATISTICS:")
    bn_keys = [k for k in state_dict.keys() if 'bn' in k.lower() or 'running' in k.lower()]
    for key in bn_keys[:10]:
        w = state_dict[key]
        if w.numel() > 0:
            print(f"  {key}: mean={w.mean():.6f}, std={w.std():.6f}, min={w.min():.6f}, max={w.max():.6f}")

if __name__ == "__main__":
    analyze_weights()
