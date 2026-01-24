#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import numpy as np

sys.path.insert(0, '.')
from models_base import DeepPrintBaseline

def analyze_tensor(name, tensor):
    """Analisa estatísticas de um tensor"""
    if tensor is None:
        print(f"  {name}: None")
        return
    
    t = tensor.detach()
    if t.numel() == 0:
        print(f"  {name}: empty tensor")
        return
    
    print(f"  {name}: shape={tuple(t.shape)}, mean={t.mean():.6f}, std={t.std():.6f}, min={t.min():.6f}, max={t.max():.6f}")
    
    # Verificar valores problemáticos
    if torch.isnan(t).any():
        print(f"    ⚠️  Contém NaN!")
    if torch.isinf(t).any():
        print(f"    ⚠️  Contém Inf!")
    if t.std() < 1e-6:
        print(f"    ⚠️  Variância muito baixa (possível colapso)!")

def debug_forward_pass():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Criar modelo
    model = DeepPrintBaseline(texture_embedding_dims=512, num_classes=100)
    model = model.to(device)
    model.eval()
    
    print("\n=== MODELO NOVO (não treinado) ===")
    
    # Input aleatório
    x = torch.randn(4, 1, 299, 299).to(device)
    print(f"\nInput:")
    analyze_tensor("x", x)
    
    # Forward manual para debug
    with torch.no_grad():
        # Stem
        stem_out = model.stem(x)
        print(f"\nApós Stem:")
        analyze_tensor("stem_out", stem_out)
        
        # Texture branch - camada por camada
        tb = model.texture_branch
        
        t0 = tb._0_block(stem_out)
        print(f"\nApós _0_block (Inception_A + Reduction_A):")
        analyze_tensor("t0", t0)
        
        t1 = tb._1_block(t0)
        print(f"\nApós _1_block (Inception_B + Reduction_B):")
        analyze_tensor("t1", t1)
        
        t2 = tb._2_block(t1)
        print(f"\nApós _2_block (Inception_C):")
        analyze_tensor("t2", t2)
        
        t3 = tb._3_avg_pool2d(t2)
        print(f"\nApós avg_pool2d:")
        analyze_tensor("t3", t3)
        
        t4 = tb._4_flatten(t3)
        print(f"\nApós flatten:")
        analyze_tensor("t4", t4)
        
        t5 = tb._5_dropout(t4)  # Dropout não faz nada em eval mode
        print(f"\nApós dropout:")
        analyze_tensor("t5", t5)
        
        t6 = tb._6_linear(t5)
        print(f"\nApós linear:")
        analyze_tensor("t6", t6)
        
        # Normalização L2
        t7 = torch.nn.functional.normalize(torch.squeeze(t6), dim=1)
        print(f"\nApós normalize:")
        analyze_tensor("t7 (embedding final)", t7)
        
        # Forward completo
        result = model(x)
        print(f"\nForward completo:")
        analyze_tensor("embedding", result["embedding"])
        if "logits" in result:
            analyze_tensor("logits", result["logits"])
    
    # Agora carregar modelo treinado
    print("\n\n=== MODELO TREINADO ===")
    
    import os
    checkpoint_path = "exp0_baseline/models/best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Obter num_classes do checkpoint
        state_dict = checkpoint['model_state_dict']
        if 'classifier.0.weight' in state_dict:
            num_classes = state_dict['classifier.0.weight'].shape[0]
            print(f"Num classes do checkpoint: {num_classes}")
        else:
            num_classes = 2120
            print(f"Usando num_classes default: {num_classes}")
        
        # Recriar modelo com classes corretas
        model2 = DeepPrintBaseline(texture_embedding_dims=512, num_classes=num_classes)
        model2.load_state_dict(state_dict, strict=False)
        model2 = model2.to(device)
        model2.eval()
        
        print(f"\nModelo carregado da época: {checkpoint.get('epoch', 'N/A')}")
        
        with torch.no_grad():
            # Stem
            stem_out = model2.stem(x)
            print(f"\nApós Stem:")
            analyze_tensor("stem_out", stem_out)
            
            # Texture branch
            tb = model2.texture_branch
            
            t0 = tb._0_block(stem_out)
            print(f"\nApós _0_block:")
            analyze_tensor("t0", t0)
            
            t1 = tb._1_block(t0)
            print(f"\nApós _1_block:")
            analyze_tensor("t1", t1)
            
            t2 = tb._2_block(t1)
            print(f"\nApós _2_block:")
            analyze_tensor("t2", t2)
            
            t3 = tb._3_avg_pool2d(t2)
            print(f"\nApós avg_pool2d:")
            analyze_tensor("t3", t3)
            
            t4 = tb._4_flatten(t3)
            print(f"\nApós flatten:")
            analyze_tensor("t4", t4)
            
            t5 = tb._5_dropout(t4)
            print(f"\nApós dropout:")
            analyze_tensor("t5", t5)
            
            t6 = tb._6_linear(t5)
            print(f"\nApós linear:")
            analyze_tensor("t6", t6)
            
            t7 = torch.nn.functional.normalize(torch.squeeze(t6), dim=1)
            print(f"\nApós normalize:")
            analyze_tensor("t7 (embedding final)", t7)
            
            # Forward completo
            result = model2(x)
            print(f"\nForward completo:")
            analyze_tensor("embedding", result["embedding"])
            
            # Verificar se múltiplos inputs geram mesma saída
            print("\n\n=== TESTE DE VARIABILIDADE ===")
            x1 = torch.randn(2, 1, 299, 299).to(device)
            x2 = torch.randn(2, 1, 299, 299).to(device)
            
            emb1 = model2(x1)["embedding"]
            emb2 = model2(x2)["embedding"]
            
            print(f"Embedding 1:")
            analyze_tensor("emb1", emb1)
            print(f"Embedding 2:")
            analyze_tensor("emb2", emb2)
            
            # Similaridade entre embeddings de inputs diferentes
            sim = torch.nn.functional.cosine_similarity(emb1[0:1], emb2[0:1])
            print(f"\nSimilaridade coseno entre inputs DIFERENTES: {sim.item():.6f}")
            
            if sim.item() > 0.99:
                print("⚠️  PROBLEMA: Inputs diferentes geram embeddings quase idênticos!")
    else:
        print(f"Checkpoint não encontrado: {checkpoint_path}")

if __name__ == "__main__":
    debug_forward_pass()
