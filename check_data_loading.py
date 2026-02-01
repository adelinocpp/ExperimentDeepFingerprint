"""
Verificar se carregamento de imagens e minúcias está correto
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_image_loading():
    """Verificar se imagens estão sendo carregadas corretamente"""
    print("=" * 80)
    print("1. VERIFICANDO CARREGAMENTO DE IMAGENS")
    print("=" * 80)
    
    # Pegar uma imagem do SFinge
    sfinge_path = Path("/media/DRAGONSTONE/MEGAsync/Forense/Papiloscopia/Bases/SFinge/FP_gen_0")
    
    if not sfinge_path.exists():
        print("⚠️  Path SFinge não encontrado")
        return
    
    # Pegar primeira imagem
    image_files = list(sfinge_path.glob("*.png"))
    if len(image_files) == 0:
        print("⚠️  Nenhuma imagem encontrada")
        return
    
    test_image = image_files[0]
    print(f"Testando com: {test_image.name}")
    
    # Carregar com cv2 (como no código)
    img = cv2.imread(str(test_image), cv2.IMREAD_GRAYSCALE)
    
    print(f"\nImagem original:")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Min: {img.min()}, Max: {img.max()}")
    print(f"  Mean: {img.mean():.2f}")
    
    # Normalizar [0, 1]
    img_norm = img.astype(np.float32) / 255.0
    print(f"\nImagem normalizada:")
    print(f"  Min: {img_norm.min():.4f}, Max: {img_norm.max():.4f}")
    print(f"  Mean: {img_norm.mean():.4f}")
    
    # Verificar se não está invertida (imagem de digital deve ter fundo branco)
    if img_norm.mean() < 0.5:
        print("  ⚠️  ATENÇÃO: Imagem parece estar invertida (fundo escuro)")
    else:
        print("  ✓ OK: Fundo claro (esperado)")
    
    return img


def check_minutiae_loading():
    """Verificar se minúcias estão sendo carregadas corretamente"""
    print("\n" + "=" * 80)
    print("2. VERIFICANDO CARREGAMENTO DE MINÚCIAS")
    print("=" * 80)
    
    # Pegar arquivo .xyt
    xyt_path = Path("/media/DRAGONSTONE/MEGAsync/Forense/Papiloscopia/Bases/SFinge/FP_gen_0_minu/fingerprint_0001_v01.xyt")
    
    if not xyt_path.exists():
        print("⚠️  Arquivo .xyt não encontrado")
        return
    
    print(f"Testando com: {xyt_path.name}")
    
    # Ler arquivo .xyt
    locations = []
    orientations_deg = []
    
    with open(xyt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            x = int(parts[0])
            y = int(parts[1])
            theta = int(parts[2])
            
            locations.append([x, y])
            orientations_deg.append(theta)
    
    print(f"\nMinúcias encontradas: {len(locations)}")
    
    if len(locations) > 0:
        locations = np.array(locations)
        orientations_deg = np.array(orientations_deg)
        
        print(f"  Coordenadas X: min={locations[:, 0].min()}, max={locations[:, 0].max()}")
        print(f"  Coordenadas Y: min={locations[:, 1].min()}, max={locations[:, 1].max()}")
        print(f"  Orientações: min={orientations_deg.min()}°, max={orientations_deg.max()}°")
        
        # Verificar se precisa inverter Y
        print(f"\n  ⚠️  ATENÇÃO: Arquivo .xyt usa coordenadas BOTTOM-LEFT")
        print(f"     Código deve converter para TOP-LEFT: y_top = image_height - y_bottom")
        
        # Verificar orientações
        print(f"\n  Orientações em graus (amostra):")
        for i in range(min(5, len(orientations_deg))):
            rad = np.deg2rad(orientations_deg[i])
            print(f"    {orientations_deg[i]}° = {rad:.4f} rad")
        
        # Verificar se ângulos estão no range correto
        if orientations_deg.min() >= 0 and orientations_deg.max() <= 360:
            print(f"  ✓ OK: Ângulos no range [0, 360]")
        else:
            print(f"  ❌ ERRO: Ângulos fora do range esperado")
    
    return locations, orientations_deg


def check_minutia_map_generation():
    """Verificar se geração de mapa de minúcias está correta"""
    print("\n" + "=" * 80)
    print("3. VERIFICANDO GERAÇÃO DE MAPA DE MINÚCIAS")
    print("=" * 80)
    
    # Importar função
    try:
        from minutia_map_generator import read_xyt_file, load_minutia_map_from_xyt
        
        xyt_path = Path("/media/DRAGONSTONE/MEGAsync/Forense/Papiloscopia/Bases/SFinge/FP_gen_0_minu/fingerprint_0001_v01.xyt")
        
        if not xyt_path.exists():
            print("⚠️  Arquivo .xyt não encontrado")
            return
        
        # Ler com função do código
        image_height = 512  # Altura típica do SFinge
        locations, orientations = read_xyt_file(xyt_path, image_height=image_height)
        
        print(f"Minúcias carregadas: {len(locations)}")
        
        if len(locations) > 0:
            print(f"  Localizações (após conversão TOP-LEFT):")
            print(f"    X: min={locations[:, 0].min():.0f}, max={locations[:, 0].max():.0f}")
            print(f"    Y: min={locations[:, 1].min():.0f}, max={locations[:, 1].max():.0f}")
            
            print(f"  Orientações (radianos):")
            print(f"    Min: {orientations.min():.4f}, Max: {orientations.max():.4f}")
            
            # Verificar se conversão Y está correta
            if locations[:, 1].max() > image_height:
                print(f"  ❌ ERRO: Coordenadas Y > image_height (inversão incorreta)")
            else:
                print(f"  ✓ OK: Coordenadas Y dentro do range")
            
            # Verificar orientações
            if orientations.min() >= 0 and orientations.max() <= 2*np.pi:
                print(f"  ✓ OK: Orientações em [0, 2π]")
            else:
                print(f"  ⚠️  ATENÇÃO: Orientações fora de [0, 2π]")
        
        # Tentar gerar mapa
        print("\nGerando mapa de minúcias...")
        minu_map, weight = load_minutia_map_from_xyt(xyt_path, image_resolution=(512, 512))
        
        print(f"  Mapa shape: {minu_map.shape}")
        print(f"  Weight: {weight:.4f}")
        print(f"  Mapa min: {minu_map.min():.4f}, max: {minu_map.max():.4f}")
        
        if weight > 0:
            print(f"  ✓ OK: Mapa gerado com sucesso")
        else:
            print(f"  ❌ ERRO: Mapa vazio (weight=0)")
        
    except Exception as e:
        print(f"❌ Erro ao importar/usar funções: {e}")


if __name__ == "__main__":
    check_image_loading()
    check_minutiae_loading()
    check_minutia_map_generation()
    
    print("\n" + "=" * 80)
    print("RESUMO:")
    print("=" * 80)
    print("Verificar:")
    print("1. Imagens: fundo claro (normalização correta)")
    print("2. Minúcias: coordenadas Y invertidas (bottom-left → top-left)")
    print("3. Ângulos: convertidos para radianos [0, 2π]")
    print("4. Mapa: gerado com weight > 0")
