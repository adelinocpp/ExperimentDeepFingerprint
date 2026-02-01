"""
Diagnóstico: Analisar checkpoint para verificar se modelo treinou
"""
import torch
import numpy as np
from pathlib import Path

def analyze_checkpoint():
    checkpoint_path = Path("exp0_baseline/models/best_model.pt")
    if not checkpoint_path.exists():
        print(f"ERRO: Checkpoint não encontrado: {checkpoint_path}")
        return
    
    print("=" * 80)
    print("ANÁLISE DO CHECKPOINT")
    print("=" * 80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n1. Informações Básicas:")
    print(f"   Época: {checkpoint.get('epoch', '?')}")
    print(f"   Número de classes: {checkpoint.get('num_classes', '?')}")
    
    # Analisar pesos das camadas de embedding
    print(f"\n2. Pesos das Camadas de Embedding:")
    model_state = checkpoint["model_state_dict"]
    
    for key in model_state.keys():
        if 'texture_branch._6_linear.weight' in key:
            weights = model_state[key].cpu().numpy()
            print(f"\n   Texture embedding linear:")
            print(f"     Shape: {weights.shape}")
            print(f"     Mean: {weights.mean():.6f}")
            print(f"     Std: {weights.std():.6f}")
            print(f"     Min: {weights.min():.6f}")
            print(f"     Max: {weights.max():.6f}")
            
        if 'minutia_embedding._4_linear.weight' in key:
            weights = model_state[key].cpu().numpy()
            print(f"\n   Minutia embedding linear:")
            print(f"     Shape: {weights.shape}")
            print(f"     Mean: {weights.mean():.6f}")
            print(f"     Std: {weights.std():.6f}")
            print(f"     Min: {weights.min():.6f}")
            print(f"     Max: {weights.max():.6f}")
    
    # Analisar Center Loss centers
    print(f"\n3. Center Loss Centers:")
    
    if "center_loss_texture_state" in checkpoint and checkpoint["center_loss_texture_state"]:
        centers = checkpoint["center_loss_texture_state"]["centers"].cpu().numpy()
        print(f"\n   Texture centers:")
        print(f"     Shape: {centers.shape}")
        print(f"     Mean: {centers.mean():.6f}")
        print(f"     Std: {centers.std():.6f}")
        print(f"     Min: {centers.min():.6f}")
        print(f"     Max: {centers.max():.6f}")
        
        # Verificar se centers são todos iguais
        center_diffs = np.std(centers, axis=0)
        print(f"     Std por dimensão (mean): {center_diffs.mean():.6f}")
        print(f"     Dimensões com std=0: {np.sum(center_diffs < 1e-8)}/{len(center_diffs)}")
    
    if "center_loss_minutia_state" in checkpoint and checkpoint["center_loss_minutia_state"]:
        centers = checkpoint["center_loss_minutia_state"]["centers"].cpu().numpy()
        print(f"\n   Minutia centers:")
        print(f"     Shape: {centers.shape}")
        print(f"     Mean: {centers.mean():.6f}")
        print(f"     Std: {centers.std():.6f}")
        print(f"     Min: {centers.min():.6f}")
        print(f"     Max: {centers.max():.6f}")
        
        center_diffs = np.std(centers, axis=0)
        print(f"     Std por dimensão (mean): {center_diffs.mean():.6f}")
        print(f"     Dimensões com std=0: {np.sum(center_diffs < 1e-8)}/{len(center_diffs)}")
    
    # Analisar histórico de loss
    print(f"\n4. Histórico de Treinamento:")
    if "history" in checkpoint:
        history = checkpoint["history"]
        if "train_loss" in history and len(history["train_loss"]) > 0:
            train_losses = history["train_loss"]
            print(f"   Épocas treinadas: {len(train_losses)}")
            print(f"   Train loss inicial: {train_losses[0]:.4f}")
            print(f"   Train loss final: {train_losses[-1]:.4f}")
            print(f"   Redução: {(train_losses[0] - train_losses[-1]):.4f} ({100*(train_losses[0]-train_losses[-1])/train_losses[0]:.1f}%)")
            
            if len(train_losses) >= 5:
                last_5_std = np.std(train_losses[-5:])
                print(f"   Últimas 5 épocas std: {last_5_std:.4f} (plateau se < 0.1)")
    
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO:")
    print("=" * 80)
    
    # Verificar se modelo aprendeu algo
    if "history" in checkpoint and "train_loss" in checkpoint["history"]:
        train_losses = checkpoint["history"]["train_loss"]
        if len(train_losses) > 0:
            initial_loss = train_losses[0]
            final_loss = train_losses[-1]
            reduction = (initial_loss - final_loss) / initial_loss
            
            if final_loss > 15:
                print("❌ CRÍTICO: Train loss muito alta (>15)")
                print("   Para 6000 classes, loss aleatória ~17-18")
                print("   Modelo NÃO treinou adequadamente")
                print("\n   Possíveis causas:")
                print("   - LR muito baixa")
                print("   - Batch size muito pequeno")
                print("   - Precisa treinar MUITO mais épocas")
            elif reduction < 0.2:
                print("⚠️  Loss caiu menos de 20%")
                print("   Modelo está aprendendo DEVAGAR")
            else:
                print("✓  Loss teve redução razoável")
                
            if len(train_losses) >= 10:
                recent_std = np.std(train_losses[-10:])
                if recent_std < 0.1:
                    print("\n⚠️  Loss em PLATEAU (últimas 10 épocas)")
                    print("   Considerar aumentar LR ou continuar treinando")

if __name__ == "__main__":
    analyze_checkpoint()
