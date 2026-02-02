"""
Script de teste para validar implementação do ArcFace loss.

Testa:
1. Center Loss (modo atual) - garantir que não quebrou
2. ArcFace Loss - verificar que funciona corretamente

Uso:
    python test_arcface_implementation.py
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from losses import get_loss_function, CenterLoss, ArcFaceLoss
from config import LOSS_CONFIG


def test_center_loss():
    """Testar Center Loss (implementação original)."""
    print("=" * 80)
    print("TESTE 1: CENTER LOSS")
    print("=" * 80)

    num_classes = 10
    feat_dim = 192
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Criar loss
    center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, alpha=0.01)
    center_loss = center_loss.to(device)

    # Criar dados fake
    features = torch.randn(batch_size, feat_dim).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    # Forward pass
    loss = center_loss(features, labels)

    print(f"✅ Center Loss funcionou!")
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Centers shape: {center_loss.centers.shape}")
    print()

    return True


def test_arcface_loss():
    """Testar ArcFace Loss."""
    print("=" * 80)
    print("TESTE 2: ARCFACE LOSS")
    print("=" * 80)

    num_classes = 10
    feat_dim = 192
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Criar loss com parâmetros do config
    arcface_loss = ArcFaceLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        s=LOSS_CONFIG["arcface_scale"],
        m=LOSS_CONFIG["arcface_margin"],
        device=device
    )

    # Criar dados fake (features normalizados)
    features = torch.randn(batch_size, feat_dim).to(device)
    features = torch.nn.functional.normalize(features, p=2, dim=1)  # Normalizar
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    # Forward pass
    loss = arcface_loss(features, labels)

    print(f"✅ ArcFace Loss funcionou!")
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Weight matrix shape: {arcface_loss.weight.shape}")
    print(f"   Scale (s): {arcface_loss.s}")
    print(f"   Margin (m): {arcface_loss.m} rad (~{arcface_loss.m * 57.3:.1f}°)")
    print()

    return True


def test_combined_loss_center():
    """Testar CombinedLoss com Center Loss."""
    print("=" * 80)
    print("TESTE 3: COMBINED LOSS (CENTER)")
    print("=" * 80)

    num_classes = 10
    feat_dim = 192
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Criar loss combinada
    combined_loss = get_loss_function(
        loss_type="center",
        num_classes=num_classes,
        feat_dim=feat_dim,
        center_loss_weight=0.00125,
        device=device
    )
    combined_loss = combined_loss.to(device)

    # Criar dados fake
    features = torch.randn(batch_size, feat_dim).to(device)
    logits = torch.randn(batch_size, num_classes).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    # Forward pass
    result = combined_loss(features, logits, labels)

    print(f"✅ Combined Loss (Center) funcionou!")
    print(f"   Total loss: {result['total_loss'].item():.4f}")
    print(f"   Softmax loss: {result['softmax_loss'].item():.4f}")
    print(f"   Center loss: {result['center_loss'].item():.4f}")
    print()

    return True


def test_combined_loss_arcface():
    """Testar CombinedLoss com ArcFace."""
    print("=" * 80)
    print("TESTE 4: COMBINED LOSS (ARCFACE)")
    print("=" * 80)

    num_classes = 10
    feat_dim = 192
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Criar loss combinada
    combined_loss = get_loss_function(
        loss_type="arcface",
        num_classes=num_classes,
        feat_dim=feat_dim,
        arcface_margin=LOSS_CONFIG["arcface_margin"],
        arcface_scale=LOSS_CONFIG["arcface_scale"],
        device=device
    )
    combined_loss = combined_loss.to(device)

    # Criar dados fake (features normalizados)
    features = torch.randn(batch_size, feat_dim).to(device)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    # Forward pass (ArcFace não usa logits externos)
    result = combined_loss(features, logits=None, labels=labels)

    print(f"✅ Combined Loss (ArcFace) funcionou!")
    print(f"   Total loss: {result['total_loss'].item():.4f}")
    print(f"   ArcFace loss: {result['arcface_loss'].item():.4f}")
    print()

    return True


def test_backward_pass():
    """Testar backward pass para garantir que gradientes fluem corretamente."""
    print("=" * 80)
    print("TESTE 5: BACKWARD PASS (ARCFACE)")
    print("=" * 80)

    num_classes = 10
    feat_dim = 192
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Criar model fake (apenas embedding)
    embedding_layer = torch.nn.Linear(feat_dim, feat_dim).to(device)

    # Criar loss
    arcface_loss = ArcFaceLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        s=64.0,
        m=0.5,
        device=device
    )

    # Criar optimizer
    optimizer = torch.optim.SGD([
        {'params': embedding_layer.parameters()},
        {'params': arcface_loss.parameters()}
    ], lr=0.01)

    # Forward pass
    input_features = torch.randn(batch_size, feat_dim).to(device)
    embeddings = embedding_layer(input_features)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    loss = arcface_loss(embeddings, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Verificar gradientes
    has_gradients = False
    for name, param in arcface_loss.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            print(f"   ✅ {name} tem gradientes (norm: {param.grad.norm().item():.4f})")

    if has_gradients:
        print(f"\n✅ Backward pass funcionou! Gradientes estão fluindo corretamente.")
    else:
        print(f"\n❌ ERRO: Nenhum gradiente detectado!")
        return False

    print()
    return True


def main():
    """Executar todos os testes."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TESTE DE IMPLEMENTAÇÃO ARCFACE" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    tests = [
        ("Center Loss", test_center_loss),
        ("ArcFace Loss", test_arcface_loss),
        ("Combined Loss (Center)", test_combined_loss_center),
        ("Combined Loss (ArcFace)", test_combined_loss_arcface),
        ("Backward Pass", test_backward_pass),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"❌ ERRO no teste '{name}':")
            print(f"   {type(e).__name__}: {e}")
            print()
            results.append((name, False))

    # Resumo
    print("=" * 80)
    print("RESUMO DOS TESTES")
    print("=" * 80)
    for name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in results)
    print()
    if all_passed:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("\n✅ Implementação do ArcFace está funcionando corretamente.")
        print("✅ Próximo passo: Testar treinamento com debug_minimal")
    else:
        print("❌ ALGUNS TESTES FALHARAM!")
        print("   Verificar implementação antes de prosseguir.")

    print()
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
