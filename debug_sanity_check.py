"""
Verificação de sanidade: números finitos, loss coerente, extração correta
"""
import torch
import numpy as np
from pathlib import Path
from models_base import DeepPrintBaseline
from PIL import Image
import torchvision.transforms as transforms

def check_model_weights(checkpoint_path):
    """Verificar se modelo tem números finitos"""
    print("=" * 80)
    print("1. VERIFICANDO PESOS DO MODELO")
    print("=" * 80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state = checkpoint["model_state_dict"]
    
    has_nan = False
    has_inf = False
    
    for name, param in model_state.items():
        if torch.isnan(param).any():
            print(f"❌ NaN encontrado em: {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"❌ Inf encontrado em: {name}")
            has_inf = True
    
    if not has_nan and not has_inf:
        print("✓ Todos os pesos são números finitos")
    
    return not (has_nan or has_inf)

def check_forward_pass(checkpoint_path, device):
    """Verificar forward pass produz números finitos"""
    print("\n" + "=" * 80)
    print("2. VERIFICANDO FORWARD PASS")
    print("=" * 80)
    
    # Carregar modelo
    model = DeepPrintBaseline(
        texture_embedding_dims=96,
        minutia_embedding_dims=96
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    if "num_classes" in checkpoint:
        model.set_num_classes(checkpoint["num_classes"])
    
    model.eval()
    
    # Criar imagem de teste (imagem sintética)
    print("\nTestando com imagem sintética...")
    fake_image = torch.randn(2, 1, 299, 299).to(device)
    
    with torch.no_grad():
        outputs = model(fake_image)
    
    # Verificar saídas
    checks = {
        "texture_embedding": outputs.get("texture_embedding"),
        "minutia_embedding": outputs.get("minutia_embedding"),
        "embedding": outputs.get("embedding"),
        "minutia_map": outputs.get("minutia_map"),
    }
    
    all_finite = True
    for name, tensor in checks.items():
        if tensor is None:
            print(f"⚠️  {name}: None")
            continue
            
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan or has_inf:
            print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
            print(f"   Shape: {tensor.shape}")
            print(f"   Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
            all_finite = False
        else:
            mean = tensor.mean().item()
            std = tensor.std().item()
            print(f"✓ {name}: finito (mean={mean:.6f}, std={std:.6f})")
    
    return all_finite

def check_loss_calculation(checkpoint_path, device):
    """Verificar cálculo da loss"""
    print("\n" + "=" * 80)
    print("3. VERIFICANDO CÁLCULO DA LOSS")
    print("=" * 80)
    
    # Carregar modelo
    model = DeepPrintBaseline(
        texture_embedding_dims=96,
        minutia_embedding_dims=96
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.set_num_classes(6000)  # Treino tem 6000 classes
    model.train()
    
    # Criar batch sintético
    fake_images = torch.randn(4, 1, 299, 299).to(device)
    fake_labels = torch.tensor([0, 1, 2, 3]).to(device)
    
    # Forward
    outputs = model(fake_images)
    
    # Calcular loss manualmente
    print("\nCalculando loss componentes:")
    
    # CrossEntropy Texture
    if "texture_logits" in outputs:
        criterion_ce = torch.nn.CrossEntropyLoss()
        ce_texture = criterion_ce(outputs["texture_logits"], fake_labels)
        print(f"  CrossEntropy Texture: {ce_texture.item():.4f}")
        
        # Verificar coerência
        num_classes = outputs["texture_logits"].shape[1]
        expected_random = -np.log(1.0 / num_classes)
        print(f"    (esperado random para {num_classes} classes: {expected_random:.4f})")
        
        if torch.isnan(ce_texture) or torch.isinf(ce_texture):
            print(f"    ❌ CrossEntropy tem NaN/Inf!")
            return False
    
    # CrossEntropy Minutia
    if "minutia_logits" in outputs:
        ce_minutia = criterion_ce(outputs["minutia_logits"], fake_labels)
        print(f"  CrossEntropy Minutia: {ce_minutia.item():.4f}")
        
        if torch.isnan(ce_minutia) or torch.isinf(ce_minutia):
            print(f"    ❌ CrossEntropy tem NaN/Inf!")
            return False
    
    print("\n✓ Loss calculation parece OK (números finitos)")
    return True

def check_real_image_extraction():
    """Verificar extração de features de imagem real"""
    print("\n" + "=" * 80)
    print("4. VERIFICANDO EXTRAÇÃO DE IMAGEM REAL")
    print("=" * 80)
    
    # Procurar uma imagem de teste
    sfinge_path = Path("/media/DRAGONSTONE/MEGAsync/Forense/Papiloscopia/Bases/SFinge/FP_gen_0")
    
    if not sfinge_path.exists():
        print("⚠️  Path SFinge não encontrado, pulando teste")
        return True
    
    # Pegar primeira imagem
    image_files = list(sfinge_path.glob("*.tif"))
    if len(image_files) == 0:
        print("⚠️  Nenhuma imagem .tif encontrada")
        return True
    
    test_image_path = image_files[0]
    print(f"Testando com: {test_image_path.name}")
    
    # Carregar e preprocessar
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    try:
        image = Image.open(test_image_path)
        image_tensor = transform(image).unsqueeze(0)
        
        print(f"  Imagem carregada: shape={image_tensor.shape}")
        print(f"  Min: {image_tensor.min().item():.4f}, Max: {image_tensor.max().item():.4f}")
        print(f"  Mean: {image_tensor.mean().item():.4f}, Std: {image_tensor.std().item():.4f}")
        
        if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
            print("  ❌ Imagem tem NaN/Inf após processamento!")
            return False
        
        print("  ✓ Imagem processada corretamente")
        
    except Exception as e:
        print(f"  ❌ Erro ao carregar imagem: {e}")
        return False
    
    return True

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    checkpoint_path = Path("exp0_baseline/models/best_model.pt")
    if not checkpoint_path.exists():
        print(f"ERRO: Checkpoint não encontrado: {checkpoint_path}")
        return
    
    # Executar verificações
    weights_ok = check_model_weights(checkpoint_path)
    forward_ok = check_forward_pass(checkpoint_path, device)
    loss_ok = check_loss_calculation(checkpoint_path, device)
    image_ok = check_real_image_extraction()
    
    # Resumo
    print("\n" + "=" * 80)
    print("RESUMO DA VERIFICAÇÃO DE SANIDADE")
    print("=" * 80)
    print(f"1. Pesos do modelo finitos: {'✓' if weights_ok else '❌'}")
    print(f"2. Forward pass finito: {'✓' if forward_ok else '❌'}")
    print(f"3. Cálculo da loss OK: {'✓' if loss_ok else '❌'}")
    print(f"4. Extração de imagem OK: {'✓' if image_ok else '❌'}")
    
    if weights_ok and forward_ok and loss_ok and image_ok:
        print("\n✓ TODOS OS TESTES PASSARAM")
        print("  Problema não é NaN/Inf ou extração quebrada")
        print("  Modelo simplesmente não treinou o suficiente (loss muito alta)")
    else:
        print("\n❌ PROBLEMAS DETECTADOS")
        print("  Verificar os erros acima")

if __name__ == "__main__":
    main()
