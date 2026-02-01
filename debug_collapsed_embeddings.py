"""
Diagnóstico: Por que embeddings estão colapsados (todos iguais)?
"""
import torch
import numpy as np
from pathlib import Path

def diagnose_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar checkpoint
    checkpoint_path = Path("exp0_baseline/models/best_model.pt")
    if not checkpoint_path.exists():
        print(f"ERRO: Checkpoint não encontrado: {checkpoint_path}")
        return
    
    print("=" * 80)
    print("1. Analisando checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"Época: {checkpoint.get('epoch', '?')}")
    print(f"Número de classes: {checkpoint.get('num_classes', '?')}")
    
    # Analisar pesos do modelo
    print("\n2. Analisando pesos do modelo...")
    model_state = checkpoint["model_state_dict"]
    
    # Verificar pesos das camadas de embedding
    texture_linear = None
    minutia_linear = None
    
    for key, value in model_state.items():
        if 'texture_branch._6_linear.weight' in key:
            texture_linear = value.cpu().numpy()
            print(f"\nTexture embedding linear layer:")
            print(f"  Shape: {texture_linear.shape}")
            print(f"  Mean: {texture_linear.mean():.6f}")
            print(f"  Std: {texture_linear.std():.6f}")
            print(f"  Todos zeros: {np.allclose(texture_linear, 0)}")
            
        if 'minutia_embedding._4_linear.weight' in key:
            minutia_linear = value.cpu().numpy()
            print(f"\nMinutia embedding linear layer:")
            print(f"  Shape: {minutia_linear.shape}")
            print(f"  Mean: {minutia_linear.mean():.6f}")
            print(f"  Std: {minutia_linear.std():.6f}")
            print(f"  Todos zeros: {np.allclose(minutia_linear, 0)}")
    
    # Verificar Center Loss centers
    print("\n3. Analisando Center Loss centers...")
    if "center_loss_texture_state" in checkpoint and checkpoint["center_loss_texture_state"]:
        centers = checkpoint["center_loss_texture_state"]["centers"].cpu().numpy()
        print(f"Texture centers shape: {centers.shape}")
        print(f"  Mean: {centers.mean():.6f}")
        print(f"  Std: {centers.std():.6f}")
        print(f"  Todos zeros: {np.allclose(centers, 0)}")
        print(f"  Todos iguais: {np.allclose(centers, centers[0])}")
    
    if "center_loss_minutia_state" in checkpoint and checkpoint["center_loss_minutia_state"]:
        centers = checkpoint["center_loss_minutia_state"]["centers"].cpu().numpy()
        print(f"Minutia centers shape: {centers.shape}")
        print(f"  Mean: {centers.mean():.6f}")
        print(f"  Std: {centers.std():.6f}")
        print(f"  Todos zeros: {np.allclose(centers, 0)}")
        print(f"  Todos iguais: {np.allclose(centers, centers[0])}")
    
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO:")
    print("Verificar se pesos das camadas de embedding e centers da Center Loss")
    print("têm variação suficiente. Se tudo for zero ou igual, modelo colapsou.")
            
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Verificar saídas
            print(f"\nBatch {i+1}:")
            print(f"  - Texture embedding shape: {outputs['texture_embedding'].shape}")
            print(f"  - Minutia embedding shape: {outputs['minutia_embedding'].shape}")
            print(f"  - Combined embedding shape: {outputs['embedding'].shape}")
            
            # Estatísticas dos embeddings ANTES da normalização
            texture_emb = outputs['texture_embedding'].cpu().numpy()
            minutia_emb = outputs['minutia_embedding'].cpu().numpy()
            combined_emb = outputs['embedding'].cpu().numpy()
            
            print(f"\n  Texture embedding (antes de salvar):")
            print(f"    - Mean: {texture_emb.mean():.6f}")
            print(f"    - Std: {texture_emb.std():.6f}")
            print(f"    - Min: {texture_emb.min():.6f}")
            print(f"    - Max: {texture_emb.max():.6f}")
            print(f"    - Contém NaN: {np.isnan(texture_emb).any()}")
            print(f"    - Todos zeros: {np.allclose(texture_emb, 0)}")
            
            print(f"\n  Minutia embedding (antes de salvar):")
            print(f"    - Mean: {minutia_emb.mean():.6f}")
            print(f"    - Std: {minutia_emb.std():.6f}")
            print(f"    - Min: {minutia_emb.min():.6f}")
            print(f"    - Max: {minutia_emb.max():.6f}")
            print(f"    - Contém NaN: {np.isnan(minutia_emb).any()}")
            print(f"    - Todos zeros: {np.allclose(minutia_emb, 0)}")
            
            print(f"\n  Combined embedding (usado na avaliação):")
            print(f"    - Mean: {combined_emb.mean():.6f}")
            print(f"    - Std: {combined_emb.std():.6f}")
            print(f"    - Min: {combined_emb.min():.6f}")
            print(f"    - Max: {combined_emb.max():.6f}")
            print(f"    - Contém NaN: {np.isnan(combined_emb).any()}")
            print(f"    - Todos iguais: {np.allclose(combined_emb, combined_emb[0])}")
            
            # Verificar norma L2
            norms = np.linalg.norm(combined_emb, axis=1)
            print(f"    - Norma L2: mean={norms.mean():.6f}, std={norms.std():.6f}")
            
            embeddings_list.append(combined_emb)
            labels_list.append(labels.numpy())
    
    # Concatenar
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    # Análise global
    print("\n" + "=" * 80)
    print("4. Análise Global (16 amostras):")
    print(f"  - Shape: {all_embeddings.shape}")
    print(f"  - Mean: {all_embeddings.mean():.6f}")
    print(f"  - Std: {all_embeddings.std():.6f}")
    print(f"  - Min: {all_embeddings.min():.6f}")
    print(f"  - Max: {all_embeddings.max():.6f}")
    
    # Verificar se todos são idênticos
    all_same = True
    for i in range(1, len(all_embeddings)):
        if not np.allclose(all_embeddings[0], all_embeddings[i], atol=1e-6):
            all_same = False
            break
    
    print(f"  - Todos embeddings idênticos: {all_same}")
    
    # Calcular similaridade coseno
    print("\n5. Similaridade Coseno (após normalização):")
    normalized = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Par 0-1 (mesmo label ou diferente?)
    sim_01 = np.dot(normalized[0], normalized[1])
    print(f"  - Entre amostra 0 e 1: {sim_01:.6f} (labels: {all_labels[0]}, {all_labels[1]})")
    
    # Par 0-2
    sim_02 = np.dot(normalized[0], normalized[2])
    print(f"  - Entre amostra 0 e 2: {sim_02:.6f} (labels: {all_labels[0]}, {all_labels[2]})")
    
    # Média de todas as similaridades
    similarities = []
    for i in range(len(normalized)):
        for j in range(i+1, len(normalized)):
            sim = np.dot(normalized[i], normalized[j])
            similarities.append(sim)
    
    print(f"\n  - Similaridade média (todos pares): {np.mean(similarities):.6f}")
    print(f"  - Similaridade std (todos pares): {np.std(similarities):.6f}")
    print(f"  - Similaridade min: {np.min(similarities):.6f}")
    print(f"  - Similaridade max: {np.max(similarities):.6f}")
    
    print("\n" + "=" * 80)
    print("6. DIAGNÓSTICO:")
    if all_same:
        print("  ❌ PROBLEMA CRÍTICO: Todos embeddings SÃO IDÊNTICOS!")
        print("     - Modelo não está aprendendo")
        print("     - Verificar se checkpoint foi carregado corretamente")
        print("     - Verificar se gradientes estão fluindo durante treino")
    elif np.allclose(all_embeddings, 0):
        print("  ❌ PROBLEMA: Embeddings são todos ZEROS!")
        print("     - Modelo pode ter colapsado durante treino")
        print("     - Verificar loss e gradientes")
    elif np.std(similarities) < 0.01:
        print("  ❌ PROBLEMA: Similaridades quase idênticas (baixa variação)!")
        print("     - Modelo não está discriminando bem")
        print("     - Loss pode estar otimizando para valores ruins")
    else:
        print("  ✓ Embeddings parecem ter variação")
        print("    Mas EER=0.5 indica problema no cálculo de scores ou threshold")

if __name__ == "__main__":
    diagnose_embeddings()
