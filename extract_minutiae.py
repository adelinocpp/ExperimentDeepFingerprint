#!/usr/bin/env python3
"""
Script para extrair minutiae de todas as imagens do dataset SFinge usando MindTCT (NIST NBIS).

Processa FP_gen_0 e FP_gen_1, gerando arquivos .xyt com:
- Coordenadas (x, y) das minutiae
- Orientação (theta em graus)
- Qualidade

Uso:
    python3 extract_minutiae.py --data-dir /path/to/Bases_de_Dados
    python3 extract_minutiae.py --data-dir /path/to/Bases_de_Dados --dataset FP_gen_0
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
from tqdm import tqdm
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_mindtct_installed() -> bool:
    """Verificar se MindTCT está instalado."""
    try:
        result = subprocess.run(
            ['mindtct', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def extract_minutiae_from_image(image_path: Path) -> Tuple[bool, str]:
    """
    Extrair minutiae de uma imagem usando MindTCT.
    
    Args:
        image_path: Caminho para a imagem PNG
    
    Returns:
        (success, message): Tupla com sucesso e mensagem de erro/sucesso
    """
    try:
        # MindTCT gera arquivos: <basename>.xyt, <basename>.min, <basename>.brw, <basename>.dm
        output_base = image_path.with_suffix('')
        
        # Executar MindTCT
        # Formato: mindtct <image> <output_base>
        result = subprocess.run(
            ['mindtct', str(image_path), str(output_base)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return False, f"MindTCT failed: {result.stderr}"
        
        # Verificar se arquivo .xyt foi criado
        xyt_file = output_base.with_suffix('.xyt')
        if not xyt_file.exists():
            return False, "XYT file not created"
        
        # Remover arquivos intermediários (manter apenas .xyt)
        for ext in ['.min', '.brw', '.dm', '.lcf', '.lfm', '.hcm']:
            temp_file = output_base.with_suffix(ext)
            if temp_file.exists():
                temp_file.unlink()
        
        return True, "Success"
    
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def process_image_worker(image_path: Path) -> Tuple[Path, bool, str]:
    """Worker function para processar uma imagem."""
    success, message = extract_minutiae_from_image(image_path)
    return image_path, success, message


def extract_minutiae_parallel(
    image_paths: List[Path],
    num_workers: int = None
) -> Tuple[int, int]:
    """
    Extrair minutiae de múltiplas imagens em paralelo.
    
    Args:
        image_paths: Lista de caminhos das imagens
        num_workers: Número de workers paralelos (None = auto)
    
    Returns:
        (success_count, fail_count): Contadores de sucesso e falha
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Processando {len(image_paths)} imagens com {num_workers} workers")
    
    success_count = 0
    fail_count = 0
    failed_images = []
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_image_worker, image_paths),
            total=len(image_paths),
            desc="Extraindo minutiae"
        ))
    
    for image_path, success, message in results:
        if success:
            success_count += 1
        else:
            fail_count += 1
            failed_images.append((image_path.name, message))
    
    if failed_images:
        logger.warning(f"\n{fail_count} imagens falharam:")
        for img_name, msg in failed_images[:10]:
            logger.warning(f"  {img_name}: {msg}")
        if len(failed_images) > 10:
            logger.warning(f"  ... e mais {len(failed_images) - 10} imagens")
    
    return success_count, fail_count


def find_images_to_process(dataset_dir: Path) -> List[Path]:
    """
    Encontrar todas as imagens PNG que ainda não têm arquivo .xyt.
    
    Args:
        dataset_dir: Diretório do dataset (ex: FP_gen_0)
    
    Returns:
        Lista de caminhos das imagens a processar
    """
    all_images = sorted(dataset_dir.glob("*.png"))
    
    images_to_process = []
    for img_path in all_images:
        xyt_path = img_path.with_suffix('.xyt')
        if not xyt_path.exists():
            images_to_process.append(img_path)
    
    logger.info(
        f"{dataset_dir.name}: {len(all_images)} imagens totais, "
        f"{len(images_to_process)} a processar"
    )
    
    return images_to_process


def main():
    parser = argparse.ArgumentParser(
        description='Extrair minutiae de imagens de impressões digitais usando MindTCT'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Diretório raiz dos datasets (contém FP_gen_0, FP_gen_1, etc.)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['FP_gen_0', 'FP_gen_1', 'all'],
        default='all',
        help='Dataset a processar (default: all)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Número de workers paralelos (default: auto)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Apenas listar imagens a processar, sem executar'
    )
    
    args = parser.parse_args()
    
    # Verificar se MindTCT está instalado
    if not args.dry_run and not check_mindtct_installed():
        logger.error(
            "MindTCT não encontrado! Instale o NIST NBIS:\n"
            "  Ubuntu/Debian: sudo apt-get install nbis\n"
            "  Ou compile: https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis"
        )
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Diretório não encontrado: {data_dir}")
        sys.exit(1)
    
    # Determinar datasets a processar
    if args.dataset == 'all':
        datasets = ['FP_gen_0', 'FP_gen_1']
    else:
        datasets = [args.dataset]
    
    # Processar cada dataset
    total_success = 0
    total_fail = 0
    
    for dataset_name in datasets:
        dataset_dir = data_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset não encontrado: {dataset_dir}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Encontrar imagens a processar
        images_to_process = find_images_to_process(dataset_dir)
        
        if not images_to_process:
            logger.info(f"Nenhuma imagem a processar em {dataset_name}")
            continue
        
        if args.dry_run:
            logger.info(f"DRY RUN: {len(images_to_process)} imagens seriam processadas")
            for img_path in images_to_process[:5]:
                logger.info(f"  - {img_path.name}")
            if len(images_to_process) > 5:
                logger.info(f"  ... e mais {len(images_to_process) - 5} imagens")
            continue
        
        # Extrair minutiae
        success_count, fail_count = extract_minutiae_parallel(
            images_to_process,
            num_workers=args.num_workers
        )
        
        total_success += success_count
        total_fail += fail_count
        
        logger.info(
            f"\n{dataset_name}: {success_count} sucesso, {fail_count} falhas"
        )
    
    # Resumo final
    if not args.dry_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"RESUMO FINAL")
        logger.info(f"{'='*60}")
        logger.info(f"Total processado: {total_success + total_fail}")
        logger.info(f"Sucesso: {total_success}")
        logger.info(f"Falhas: {total_fail}")
        if total_success + total_fail > 0:
            success_rate = 100 * total_success / (total_success + total_fail)
            logger.info(f"Taxa de sucesso: {success_rate:.2f}%")


if __name__ == '__main__':
    main()
