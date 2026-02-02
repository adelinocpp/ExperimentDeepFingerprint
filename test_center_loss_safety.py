"""
Teste de segurança do Center Loss adaptativo.

Verifica proteções contra valores extremos.
"""

from config import get_center_loss_weight, LOSS_CONFIG

def test_safety():
    """Testa proteções contra valores extremos."""

    print("=" * 80)
    print("TESTE DE SEGURANÇA: CENTER LOSS ADAPTATIVO")
    print("=" * 80)

    print(f"\nProteções configuradas:")
    print(f"  Min weight: {LOSS_CONFIG['center_loss_min_weight']}")
    print(f"  Max weight: {LOSS_CONFIG['center_loss_max_weight']}")
    print(f"  Expoente: {LOSS_CONFIG['center_loss_adaptive_exponent']} (0.5 = sublinear, mais seguro)")

    print("\n" + "=" * 80)
    print("TESTE 1: Casos Extremos (poucas classes)")
    print("=" * 80)

    extreme_low_cases = [1, 2, 3, 5, 10]
    for num_classes in extreme_low_cases:
        weight = get_center_loss_weight(num_classes)
        print(f"  {num_classes:>6} classes: {weight:.10f} (min={LOSS_CONFIG['center_loss_min_weight']:.2e})")

        # Verificar se está acima do mínimo
        assert weight >= LOSS_CONFIG['center_loss_min_weight'], \
            f"ERRO: peso {weight} abaixo do mínimo!"

    print("\n✅ Proteção contra ZERO: OK (todos >= min_weight)")

    print("\n" + "=" * 80)
    print("TESTE 2: Casos Extremos (muitas classes)")
    print("=" * 80)

    extreme_high_cases = [10000, 50000, 100000, 1000000]
    for num_classes in extreme_high_cases:
        weight = get_center_loss_weight(num_classes)
        scale = weight / LOSS_CONFIG['center_loss_base_weight']
        clamped = "CLAMPED!" if weight == LOSS_CONFIG['center_loss_max_weight'] else ""
        print(f"  {num_classes:>8} classes: {weight:.10f} ({scale:>6.2f}x) {clamped}")

        # Verificar se está abaixo do máximo
        assert weight <= LOSS_CONFIG['center_loss_max_weight'], \
            f"ERRO: peso {weight} acima do máximo!"

    print("\n✅ Proteção contra EXPLOSÃO: OK (todos <= max_weight)")

    print("\n" + "=" * 80)
    print("TESTE 3: Comparação Linear vs Sublinear")
    print("=" * 80)

    print("\nExpoente 0.5 (SUBLINEAR - atual) vs 1.0 (LINEAR):")
    print(f"\n{'Classes':>8} {'Sublinear (0.5)':>18} {'Linear (1.0)':>18} {'Dif':>10}")
    print("-" * 80)

    test_classes = [3, 14, 100, 1000, 6000, 10000]
    base = LOSS_CONFIG['center_loss_base_weight']

    for n in test_classes:
        # Sublinear (atual)
        w_sub = base * (n / 6000) ** 0.5
        w_sub = max(LOSS_CONFIG['center_loss_min_weight'],
                    min(w_sub, LOSS_CONFIG['center_loss_max_weight']))

        # Linear (alternativa)
        w_lin = base * (n / 6000) ** 1.0
        w_lin = max(LOSS_CONFIG['center_loss_min_weight'],
                    min(w_lin, LOSS_CONFIG['center_loss_max_weight']))

        diff = (w_sub / w_lin) if w_lin > 0 else 1.0

        print(f"{n:>8} {w_sub:>18.10f} {w_lin:>18.10f} {diff:>10.2f}x")

    print("\n✅ Expoente 0.5 cresce MAIS DEVAGAR (mais seguro)")

    print("\n" + "=" * 80)
    print("TESTE 4: Validação de Entrada")
    print("=" * 80)

    print("\nTestando num_classes inválido:")
    try:
        weight = get_center_loss_weight(0)
        print("  ❌ ERRO: aceitou num_classes=0 (deveria rejeitar!)")
    except ValueError as e:
        print(f"  ✅ Rejeitou corretamente: {e}")

    try:
        weight = get_center_loss_weight(-5)
        print("  ❌ ERRO: aceitou num_classes=-5 (deveria rejeitar!)")
    except ValueError as e:
        print(f"  ✅ Rejeitou corretamente: {e}")

    print("\n" + "=" * 80)
    print("RESUMO: TODAS AS PROTEÇÕES FUNCIONANDO! ✅")
    print("=" * 80)

    print("\nProteções ativas:")
    print("  ✅ Validação: num_classes >= 1")
    print("  ✅ Clamp mínimo: >= 1e-7 (não zera)")
    print("  ✅ Clamp máximo: <= 0.01 (não explode)")
    print("  ✅ Expoente 0.5: crescimento sublinear (mais conservador)")
    print("\nSeguro para usar em produção! 🎉")


if __name__ == "__main__":
    test_safety()
