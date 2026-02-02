"""
Script para demonstrar o efeito do Center Loss adaptativo.

Compara os pesos calculados para diferentes números de classes.
"""

from config import get_center_loss_weight, LOSS_CONFIG

def test_adaptive_weights():
    """Mostra como o peso do Center Loss varia com o número de classes."""

    print("=" * 80)
    print("TESTE: CENTER LOSS ADAPTATIVO")
    print("=" * 80)

    print(f"\nConfiguração:")
    print(f"  Peso base (paper): {LOSS_CONFIG['center_loss_base_weight']}")
    print(f"  Classes referência (paper): {LOSS_CONFIG['center_loss_num_classes_reference']}")
    print(f"  Expoente: {LOSS_CONFIG['center_loss_adaptive_exponent']}")
    print(f"  Adaptativo ativado: {LOSS_CONFIG['center_loss_use_adaptive']}")

    print("\n" + "=" * 80)
    print("COMPARAÇÃO: Peso do Center Loss para diferentes números de classes")
    print("=" * 80)

    test_cases = [
        ("debug_minimal", 3),
        ("debug", 14),
        ("teste médio", 200),
        ("teste grande", 1000),
        ("paper (produção)", 6000),
    ]

    base_weight = LOSS_CONFIG['center_loss_base_weight']

    print(f"\n{'Cenário':<20} {'Classes':>8} {'Peso Adaptativo':>18} {'Fator':>10}")
    print("-" * 80)

    for name, num_classes in test_cases:
        adaptive_weight = get_center_loss_weight(num_classes)
        scale_factor = adaptive_weight / base_weight

        print(f"{name:<20} {num_classes:>8} {adaptive_weight:>18.10f} {scale_factor:>10.4f}x")

    print("\n" + "=" * 80)
    print("ANÁLISE")
    print("=" * 80)

    weight_3 = get_center_loss_weight(3)
    weight_14 = get_center_loss_weight(14)
    weight_6000 = get_center_loss_weight(6000)

    print(f"\n1. Debug minimal (3 classes):")
    print(f"   Peso: {weight_3:.10f}")
    print(f"   Redução: {base_weight / weight_3:.0f}x menor que paper")
    print(f"   Rationale: Com 3 classes, centros ficam MUITO próximos.")
    print(f"              Center Loss forte causaria colapso imediato!")

    print(f"\n2. Debug (14 classes):")
    print(f"   Peso: {weight_14:.10f}")
    print(f"   Redução: {base_weight / weight_14:.0f}x menor que paper")
    print(f"   Rationale: Com 14 classes, ainda precisa ser MUITO mais fraco")
    print(f"              para dar espaço aos embeddings se espalharem.")

    print(f"\n3. Produção (6000 classes):")
    print(f"   Peso: {weight_6000:.10f}")
    print(f"   Igual ao paper: ✅")
    print(f"   Rationale: Com 6000 classes, espaço de embeddings é grande.")
    print(f"              Center Loss pode ser forte sem causar colapso.")

    print("\n" + "=" * 80)
    print("EXPECTATIVA")
    print("=" * 80)
    print("\nCom Center Loss adaptativo:")
    print("  ✅ Debug (14 classes): EER deve melhorar vs peso fixo 0.00125")
    print("  ✅ Embeddings: separação genuínos/impostores deve aumentar")
    print("  ✅ Variância: embeddings devem se espalhar mais no espaço vetorial")
    print("  ✅ Produção (6000 classes): mantém performance do paper")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_adaptive_weights()
