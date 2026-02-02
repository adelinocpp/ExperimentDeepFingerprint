"""
MASTER DEBUG SCRIPT

Executa todos os scripts de debug em sequência para diagnóstico completo.
"""

import subprocess
import sys

def run_debug_script(script_name, description):
    """Executa um script de debug e mostra resultado."""

    print("\n" + "=" * 80)
    print(f"EXECUTANDO: {description}")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            print(f"\n⚠️  Script {script_name} terminou com código {result.returncode}")
        else:
            print(f"\n✅ Script {script_name} concluído")

    except Exception as e:
        print(f"\n❌ ERRO ao executar {script_name}: {e}")

    input("\n[Pressione ENTER para continuar para o próximo debug...]")


def main():
    """Executa todos os debugs."""

    print("=" * 80)
    print("DIAGNÓSTICO COMPLETO - DeepPrint Baseline")
    print("=" * 80)

    print("\nEste script executará 3 debugs em sequência:")
    print("  1. Loss Breakdown - Verificar se Center Loss está funcionando")
    print("  2. Gradient Flow - Verificar se gradientes estão fluindo")
    print("  3. Normalization - Verificar se L2 normalization está correta")

    input("\n[Pressione ENTER para começar...]")

    # Debug 1: Loss Breakdown
    run_debug_script(
        "debug_1_loss_breakdown.py",
        "DEBUG 1: Loss Breakdown"
    )

    # Debug 2: Gradient Flow
    run_debug_script(
        "debug_2_gradient_flow.py",
        "DEBUG 2: Gradient Flow"
    )

    # Debug 3: Normalization
    run_debug_script(
        "debug_3_normalization.py",
        "DEBUG 3: L2 Normalization"
    )

    # Resumo final
    print("\n" + "=" * 80)
    print("TODOS OS DEBUGS CONCLUÍDOS")
    print("=" * 80)

    print("\nResumo dos scripts executados:")
    print("  ✅ debug_1_loss_breakdown.py")
    print("  ✅ debug_2_gradient_flow.py")
    print("  ✅ debug_3_normalization.py")

    print("\nAnálise os outputs acima para identificar problemas.")
    print("\nProblemas comuns a procurar:")
    print("  - Center Loss contribuição < 0.1% da loss total")
    print("  - Gradientes de embedding 100x menores que classificador")
    print("  - Separação intra/inter classe < 1%")
    print("  - Normas L2 não são 1.0 após normalização")


if __name__ == "__main__":
    main()
