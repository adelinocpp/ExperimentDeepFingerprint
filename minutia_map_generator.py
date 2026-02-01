"""
Gerador de mapas de minutiae para o DeepPrint.

Baseado na implementação original:
    flx/data/minutia_map.py
    flx/data/minutia_map_loader.py

Gera mapas de densidade gaussiana 128x128 com 6 camadas de orientação.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Constantes (mesmas do DeepPrint original)
MINUTIA_MAP_CHANNELS = 6
MINUTIA_MAP_SIZE = 128
INPUT_IMAGE_SIZE = 299  # Tamanho da imagem de entrada do DeepPrint


def read_xyt_file(
    xyt_path: Path,
    image_height: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ler arquivo .xyt gerado pelo MindTCT.
    
    IMPORTANTE: MindTCT (NIST Internal) usa coordenadas bottom-left.
    Convertemos para top-left (padrão OpenCV/imagens).
    
    Args:
        xyt_path: Caminho para arquivo .xyt
        image_height: Altura da imagem original (para conversão de coordenadas)
    
    Returns:
        (locations, orientations):
            - locations: (N, 2) coordenadas (x, y) top-left
            - orientations: (N,) orientações em radianos [0, 2π]
    """
    if not xyt_path.exists():
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    # Inferir altura da imagem a partir do nome do arquivo
    if image_height is None:
        image_path = xyt_path.with_suffix('.png')
        if image_path.exists():
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                image_height = img.shape[0]
            else:
                image_height = 900  # Fallback para SFinge
        else:
            image_height = 900  # Fallback para SFinge
    
    locations = []
    orientations = []
    
    try:
        with open(xyt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                x = int(parts[0])
                y_bottom = int(parts[1])  # Coordenada bottom-left
                theta_degrees = int(parts[2])  # Graus [0-360]
                
                # Converter bottom-left para top-left
                y_top = image_height - y_bottom
                
                # CRÍTICO: Ao inverter eixo Y (espelhamento vertical),
                # o ângulo também precisa ser espelhado
                # θ_novo = 360° - θ_original (ou 2π - θ em radianos)
                theta_flipped = (360 - theta_degrees) % 360
                
                locations.append([x, y_top])
                # Converter graus para radianos
                orientations.append(np.deg2rad(theta_flipped))
    
    except Exception as e:
        logger.warning(f"Erro lendo {xyt_path}: {e}")
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    return (
        np.array(locations, dtype=np.float32),
        np.array(orientations, dtype=np.float32)
    )


def _rescale_points(
    points: np.ndarray,
    image_resolution: Tuple[int, int],
    target_resolution: Tuple[int, int]
) -> np.ndarray:
    """
    Redimensionar coordenadas de minutiae para resolução alvo.
    
    Args:
        points: (N, 2) coordenadas (x, y)
        image_resolution: (width, height) da imagem original
        target_resolution: (width, height) alvo
    
    Returns:
        Coordenadas redimensionadas
    """
    scale_factor = min(
        target_resolution[0] / image_resolution[0],
        target_resolution[1] / image_resolution[1]
    )
    
    padding_x = (target_resolution[1] - (image_resolution[1] * scale_factor)) / 2
    padding_y = (target_resolution[0] - (image_resolution[0] * scale_factor)) / 2
    
    padding = np.array([padding_x, padding_y], dtype=np.float32)
    
    return points * scale_factor + padding


def _remove_points_outside_image(
    coords: np.ndarray,
    oris: np.ndarray,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Remover minutiae fora dos limites da imagem."""
    mask_x = coords[:, 0] >= width
    mask_y = coords[:, 1] >= height
    mask = np.logical_or(mask_x, mask_y)
    
    coords = np.delete(coords, np.where(mask), axis=0)
    oris = np.delete(oris, np.where(mask))
    
    return coords, oris


def _gaussian_mask(sigma: float, radius: int) -> np.ndarray:
    """Criar máscara gaussiana 2D."""
    x, y = np.meshgrid(
        np.linspace(-radius, radius, 2 * radius + 1),
        np.linspace(-radius, radius, 2 * radius + 1)
    )
    dst = x**2 + y**2
    return np.exp(-(dst / (2.0 * sigma**2)))


def _convert_orientations(orientations: np.ndarray) -> np.ndarray:
    """Normalizar orientações para [0, 2*pi]."""
    out = orientations.copy()
    two_pi_inv = 1 / (2 * np.pi)
    out *= two_pi_inv
    out = out - np.floor(out)
    out *= 2 * np.pi
    return out


def _layer_weights_softmax(orientations: np.ndarray, n_layers: int) -> np.ndarray:
    """
    Calcular pesos de cada layer baseado na orientação.
    
    Cada minutia contribui para múltiplos layers, com peso proporcional
    à similaridade entre sua orientação e a orientação do layer.
    
    Args:
        orientations: (N,) orientações em radianos
        n_layers: Número de layers (6 para DeepPrint)
    
    Returns:
        (N, n_layers) pesos para cada minutia em cada layer
    """
    if len(orientations) == 0:
        return np.zeros((0, n_layers), dtype=np.float32)
    
    layer_orientations = np.linspace(
        0, 2 * np.pi, num=n_layers, endpoint=False, dtype=np.float32
    )
    layer_orientations = np.tile(layer_orientations, (orientations.shape[0], 1))
    orientation_diffs = np.abs(layer_orientations - orientations[:, np.newaxis])
    
    # Se diferença > pi, usar 2*pi - diff (ângulo menor)
    mask = orientation_diffs > np.pi
    orientation_diffs[mask] *= -1
    orientation_diffs[mask] += 2 * np.pi
    
    # Softmax weights
    weights = np.exp(-orientation_diffs)
    norm = 1 / np.sum(weights, axis=1)
    return weights * norm[:, np.newaxis]


def create_minutia_map(
    minutia_locations: np.ndarray,
    minutia_orientations: np.ndarray,
    in_resolution: Tuple[int, int],
    out_resolution: Tuple[int, int] = (MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE),
    n_layers: int = MINUTIA_MAP_CHANNELS,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Criar mapa de minutiae como tensor 3D.
    
    Implementação baseada em:
        "End-to-End Latent Fingerprint Search" (Cao et al.)
        https://arxiv.org/abs/1812.10213v1
    
    Args:
        minutia_locations: (N, 2) coordenadas (x, y)
        minutia_orientations: (N,) orientações em radianos
        in_resolution: (width, height) da imagem original
        out_resolution: (width, height) do mapa de saída
        n_layers: Número de layers de orientação
        sigma: Desvio padrão da gaussiana
    
    Returns:
        np.ndarray shape (height, width, n_layers) dtype uint8 [0-255]
    """
    radius = int(np.ceil(2 * sigma))
    out_image = np.zeros(
        shape=(
            out_resolution[1] + 2 * radius,
            out_resolution[0] + 2 * radius,
            n_layers
        ),
        dtype=np.float32
    )
    
    if len(minutia_locations) == 0:
        return out_image[radius:-radius, radius:-radius].astype(dtype=np.uint8)
    
    # Redimensionar coordenadas se necessário
    if in_resolution != out_resolution:
        minutia_locations = _rescale_points(
            minutia_locations,
            image_resolution=in_resolution,
            target_resolution=out_resolution
        ).astype(np.int32)
    
    # Remover pontos fora da imagem
    minutia_locations, minutia_orientations = _remove_points_outside_image(
        minutia_locations, minutia_orientations,
        out_resolution[1], out_resolution[0]
    )
    
    if len(minutia_locations) == 0:
        return out_image[radius:-radius, radius:-radius].astype(dtype=np.uint8)
    
    # Calcular pesos por layer
    minu_layer_weights = _layer_weights_softmax(
        _convert_orientations(minutia_orientations), n_layers=n_layers
    )
    
    # Criar máscara gaussiana base
    mask = _gaussian_mask(sigma, radius)
    base_density = np.reshape(
        np.repeat(mask, n_layers), newshape=(mask.shape[0], mask.shape[0], n_layers)
    )
    
    # Adicionar cada minutia ao mapa
    for minu_idx, layer_weights in enumerate(minu_layer_weights):
        minu_density = base_density * layer_weights[np.newaxis, np.newaxis, :]
        
        out_loc = minutia_locations[minu_idx] + radius
        x_start = out_loc[0] - radius
        x_end = out_loc[0] + radius + 1
        y_start = out_loc[1] - radius
        y_end = out_loc[1] + radius + 1
        
        out_image[y_start:y_end, x_start:x_end, :] += minu_density
    
    # Normalizar e converter para uint8
    out_image = np.clip(out_image, 0, 1.0)
    out_image *= 255
    return out_image[radius:-radius, radius:-radius].astype(dtype=np.uint8)


def load_minutia_map_from_xyt(
    xyt_path: Path,
    image_resolution: Tuple[int, int] = (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
) -> Tuple[torch.Tensor, float]:
    """
    Carregar e gerar mapa de minutiae a partir de arquivo .xyt.
    
    Args:
        xyt_path: Caminho para arquivo .xyt do MindTCT
        image_resolution: Resolução da imagem original (width, height)
    
    Returns:
        (minutia_map, weight):
            - minutia_map: Tensor (6, 128, 128) float32 [0-1]
            - weight: Peso para a loss (1.0 se tem minutiae, 0.0 se vazio)
    """
    # Ler minutiae do arquivo (já converte bottom-left → top-left)
    locations, orientations = read_xyt_file(xyt_path, image_height=image_resolution[1])
    
    # Gerar mapa
    minu_map = create_minutia_map(
        locations,
        orientations,
        in_resolution=image_resolution,
        out_resolution=(MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE),
        n_layers=MINUTIA_MAP_CHANNELS,
        sigma=1.5
    )
    
    # Converter para tensor PyTorch
    # OpenCV/Numpy: (H, W, C) -> PyTorch: (C, H, W)
    minu_map_tensor = torch.from_numpy(minu_map).permute(2, 0, 1).float()
    
    # Normalizar para [0, 1]
    minu_map_tensor = minu_map_tensor / 255.0
    
    # Weight: 1.0 se tem minutiae, 0.0 se vazio (para não penalizar imagens sem .xyt)
    weight = 1.0 if len(locations) > 0 else 0.0
    
    return minu_map_tensor, weight


def get_minutia_map_path(image_path: Path) -> Path:
    """
    Obter caminho do arquivo .xyt correspondente à imagem.
    
    Args:
        image_path: Caminho da imagem (ex: fingerprint_0001_v01.png)
    
    Returns:
        Caminho do arquivo .xyt (ex: fingerprint_0001_v01.xyt)
    """
    return image_path.with_suffix('.xyt')


def batch_load_minutia_maps(
    image_paths: list,
    image_resolution: Tuple[int, int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Carregar mapas de minutiae para um batch de imagens.
    
    IMPORTANTE: Detecta automaticamente a resolução original das imagens
    para conversão correta de coordenadas bottom-left → top-left.
    
    Args:
        image_paths: Lista de caminhos das imagens originais
        image_resolution: Resolução das imagens originais (width, height).
                         Se None, detecta automaticamente da primeira imagem.
    
    Returns:
        (minutia_maps, weights):
            - minutia_maps: Tensor (B, 6, 128, 128)
            - weights: Tensor (B,) com pesos para a loss
    """
    batch_maps = []
    batch_weights = []
    
    # Detectar resolução original se não fornecida
    if image_resolution is None and len(image_paths) > 0:
        first_img_path = Path(image_paths[0])
        if first_img_path.exists():
            import cv2
            img = cv2.imread(str(first_img_path))
            if img is not None:
                image_resolution = (img.shape[1], img.shape[0])  # (width, height)
            else:
                image_resolution = (750, 900)  # Fallback SFinge
        else:
            image_resolution = (750, 900)  # Fallback SFinge
    elif image_resolution is None:
        image_resolution = (750, 900)  # Fallback SFinge
    
    for img_path in image_paths:
        xyt_path = get_minutia_map_path(Path(img_path))
        minu_map, weight = load_minutia_map_from_xyt(xyt_path, image_resolution)
        batch_maps.append(minu_map)
        batch_weights.append(weight)
    
    return (
        torch.stack(batch_maps),
        torch.tensor(batch_weights, dtype=torch.float32)
    )
