#!/usr/bin/env python3
"""
Script para visualizar minutiae extraídas pelo MindTCT.

Gera imagens com as minutiae marcadas visualmente para conferência manual.

Uso:
    python3 visualize_minutiae.py --data-dir /path/to/Bases_de_Dados --output-dir samples
    python3 visualize_minutiae.py --data-dir /path/to/Bases_de_Dados --num-samples 20
    python3 visualize_minutiae.py --data-dir /home/adelino/MegaSync/Forense/Papiloscopia/Compara_Metodos_Automaticos/Bases_de_Dados/FP_gen_0 --output-dir ./minucias_view --num-samples 20
"""

import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_xyt_file(xyt_path: Path, image_height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ler arquivo .xyt do MindTCT.
    
    IMPORTANTE: MindTCT (NIST Internal) usa coordenadas bottom-left.
    Convertemos para top-left (padrão OpenCV/imagens).
    
    Formato do arquivo .xyt:
        x y_bottom theta quality
        ...
    
    Args:
        xyt_path: Caminho para arquivo .xyt
        image_height: Altura da imagem original (para conversão de coordenadas)
    
    Returns:
        (locations, orientations, qualities): Arrays numpy
            - locations: (N, 2) coordenadas (x, y) top-left
            - orientations: (N,) orientações em radianos
            - qualities: (N,) qualidades (0-100)
    """
    if not xyt_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {xyt_path}")
    
    locations = []
    orientations = []
    qualities = []
    
    with open(xyt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            try:
                x = int(parts[0])
                y_bottom = int(parts[1])  # Coordenada bottom-left do MindTCT
                theta = int(parts[2])  # Em graus (0-359)
                quality = int(parts[3]) if len(parts) > 3 else 50
                
                # Converter bottom-left para top-left
                y_top = image_height - y_bottom
                
                locations.append([x, y_top])
                # Converter de graus para radianos
                orientations.append(np.deg2rad(theta))
                qualities.append(quality)
            except ValueError:
                continue
    
    return (
        np.array(locations, dtype=np.int32),
        np.array(orientations, dtype=np.float32),
        np.array(qualities, dtype=np.int32)
    )


def draw_minutiae_on_image(
    image: np.ndarray,
    locations: np.ndarray,
    orientations: np.ndarray,
    qualities: np.ndarray = None,
    line_length: int = 20,
    color_map: str = 'quality'
) -> np.ndarray:
    """
    Desenhar minutiae sobre a imagem.
    
    Args:
        image: Imagem BGR
        locations: Coordenadas (N, 2)
        orientations: Orientações em radianos (N,)
        qualities: Qualidades 0-100 (N,) ou None
        line_length: Comprimento da linha de orientação
        color_map: 'quality' (verde=boa, vermelho=ruim) ou 'uniform' (azul)
    
    Returns:
        Imagem com minutiae desenhadas
    """
    img_vis = image.copy()
    
    # Converter para BGR se grayscale
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    
    for idx, (loc, ori) in enumerate(zip(locations, orientations)):
        x, y = loc
        
        # Determinar cor baseada na qualidade
        if color_map == 'quality' and qualities is not None:
            quality = qualities[idx]
            # Verde (alta qualidade) -> Amarelo -> Vermelho (baixa qualidade)
            if quality >= 70:
                color = (0, 255, 0)  # Verde
            elif quality >= 40:
                color = (0, 255, 255)  # Amarelo
            else:
                color = (0, 0, 255)  # Vermelho
        else:
            color = (255, 0, 0)  # Azul
        
        # Desenhar círculo no ponto da minutia
        cv2.circle(img_vis, (x, y), 5, color, 2)
        
        # Desenhar linha indicando orientação
        end_x = int(x + line_length * np.cos(ori))
        end_y = int(y + line_length * np.sin(ori))
        cv2.line(img_vis, (x, y), (end_x, end_y), color, 2)
    
    return img_vis


def create_comparison_image(
    original: np.ndarray,
    with_minutiae: np.ndarray,
    image_name: str,
    num_minutiae: int
) -> np.ndarray:
    """
    Criar imagem lado a lado: original | com minutiae.
    
    Args:
        original: Imagem original
        with_minutiae: Imagem com minutiae desenhadas
        image_name: Nome da imagem
        num_minutiae: Número de minutiae detectadas
    
    Returns:
        Imagem concatenada com texto
    """
    # Garantir que ambas têm mesma altura
    h1, w1 = original.shape[:2]
    h2, w2 = with_minutiae.shape[:2]
    
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Concatenar horizontalmente
    comparison = np.hstack([original, with_minutiae])
    
    # Adicionar borda superior para texto
    text_height = 40
    comparison_with_text = np.zeros(
        (comparison.shape[0] + text_height, comparison.shape[1], 3),
        dtype=np.uint8
    )
    comparison_with_text.fill(255)
    comparison_with_text[text_height:, :] = comparison
    
    # Adicionar texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{image_name} - {num_minutiae} minutiae detectadas"
    cv2.putText(
        comparison_with_text, text, (10, 25),
        font, 0.6, (0, 0, 0), 2
    )
    
    return comparison_with_text


def process_sample_image(
    image_path: Path,
    xyt_path: Path,
    output_dir: Path
) -> bool:
    """
    Processar uma imagem de amostra.
    
    Args:
        image_path: Caminho da imagem PNG
        xyt_path: Caminho do arquivo .xyt
        output_dir: Diretório de saída
    
    Returns:
        True se processado com sucesso
    """
    try:
        # Ler imagem
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Não foi possível ler: {image_path}")
            return False
        
        image_height = image.shape[0]
        
        # Ler minutiae (com conversão bottom-left → top-left)
        locations, orientations, qualities = read_xyt_file(xyt_path, image_height)
        
        if len(locations) == 0:
            logger.warning(f"Nenhuma minutia encontrada em: {image_path.name}")
            return False
        
        # Desenhar minutiae
        img_with_minutiae = draw_minutiae_on_image(
            image, locations, orientations, qualities,
            color_map='quality'
        )
        
        # Criar comparação
        comparison = create_comparison_image(
            image, img_with_minutiae,
            image_path.name, len(locations)
        )
        
        # Salvar
        output_path = output_dir / f"minutiae_{image_path.stem}.png"
        cv2.imwrite(str(output_path), comparison)
        
        logger.info(
            f"✓ {image_path.name}: {len(locations)} minutiae "
            f"(qualidade média: {qualities.mean():.1f})"
        )
        
        return True
    
    except Exception as e:
        logger.error(f"Erro processando {image_path.name}: {e}")
        return False


def select_random_samples(
    dataset_dir: Path,
    num_samples: int
) -> List[Tuple[Path, Path]]:
    """
    Selecionar amostras aleatórias com arquivos .xyt.
    
    Args:
        dataset_dir: Diretório do dataset
        num_samples: Número de amostras a selecionar
    
    Returns:
        Lista de tuplas (image_path, xyt_path)
    """
    # Encontrar todos os pares (png, xyt)
    all_xyt = list(dataset_dir.glob("*.xyt"))
    
    valid_pairs = []
    for xyt_path in all_xyt:
        image_path = xyt_path.with_suffix('.png')
        if image_path.exists():
            valid_pairs.append((image_path, xyt_path))
    
    logger.info(f"Encontrados {len(valid_pairs)} pares (imagem, minutiae)")
    
    # Selecionar amostras aleatórias
    num_samples = min(num_samples, len(valid_pairs))
    samples = random.sample(valid_pairs, num_samples)
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Visualizar minutiae extraídas pelo MindTCT'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Diretório raiz dos datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['FP_gen_0', 'FP_gen_1'],
        default='FP_gen_0',
        help='Dataset a visualizar (default: FP_gen_0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='minutiae_samples',
        help='Diretório de saída para imagens (default: minutiae_samples)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Número de amostras a visualizar (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para seleção aleatória (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Configurar seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Verificar diretórios
    data_dir = Path(args.data_dir)
    dataset_dir = data_dir / args.dataset
    
    if not dataset_dir.exists():
        logger.error(f"Dataset não encontrado: {dataset_dir}")
        return
    
    # Criar diretório de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Visualizando minutiae: {args.dataset}")
    logger.info(f"{'='*60}")
    
    # Selecionar amostras
    samples = select_random_samples(dataset_dir, args.num_samples)
    
    if not samples:
        logger.error("Nenhum arquivo .xyt encontrado! Execute extract_minutiae.py primeiro.")
        return
    
    # Processar amostras
    success_count = 0
    for image_path, xyt_path in samples:
        if process_sample_image(image_path, xyt_path, output_dir):
            success_count += 1
    
    # Resumo
    logger.info(f"\n{'='*60}")
    logger.info(f"Processadas {success_count}/{len(samples)} amostras")
    logger.info(f"Imagens salvas em: {output_dir.absolute()}")
    logger.info(f"{'='*60}")
    logger.info("\nCódigo de cores:")
    logger.info("  🟢 Verde: Alta qualidade (≥70)")
    logger.info("  🟡 Amarelo: Média qualidade (40-69)")
    logger.info("  🔴 Vermelho: Baixa qualidade (<40)")


if __name__ == '__main__':
    main()
