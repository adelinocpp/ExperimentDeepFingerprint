#!/usr/bin/env python3
"""
Script de teste para verificar o carregamento das bases FVC.
Executa verificações para garantir que:
1. Todas as imagens são carregadas
2. Labels são atribuídos corretamente (por origem única)
3. A divisão train/val/test é reproduzível
4. Não há vazamento de dados entre splits
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos do projeto
from data_loader import FVCDatasetLoader, load_fvc_datasets
from config import DATA_DIR


def test_fvc_loader():
    """Testar o carregador FVC."""
    print("\n" + "=" * 80)
    print("TESTE DO CARREGADOR FVC")
    print("=" * 80)
    
    # 1. Verificar se os diretórios existem
    print("\n1. Verificando diretórios de dados...")
    datasets_found = []
    for ds_name in ["FVC2000", "FVC2002", "FVC2004"]:
        ds_path = DATA_DIR / ds_name
        if ds_path.exists():
            subdirs = [d for d in ds_path.iterdir() if d.is_dir()]
            print(f"   ✓ {ds_name}: {len(subdirs)} subdiretórios encontrados")
            datasets_found.append(ds_name)
        else:
            print(f"   ✗ {ds_name}: NÃO ENCONTRADO em {ds_path}")
    
    if not datasets_found:
        print("\n❌ ERRO: Nenhum dataset FVC encontrado!")
        return False
    
    # 2. Carregar datasets
    print("\n2. Carregando datasets...")
    loader = FVCDatasetLoader(
        datasets=datasets_found,
        random_state=42,
    )
    
    # 3. Verificar estatísticas
    print("\n3. Estatísticas do dataset:")
    stats = loader.get_statistics()
    print(f"   Total de imagens: {stats['total_images']}")
    print(f"   Total de origens únicas: {stats['total_origins']}")
    print(f"   Imagens por dataset: {stats['images_per_dataset']}")
    print(f"   Imagens por subdiretório: {stats['images_per_subdir']}")
    print(f"   Versões por origem: min={stats['versions_per_origin']['min']}, "
          f"max={stats['versions_per_origin']['max']}, "
          f"média={stats['versions_per_origin']['mean']:.1f}")
    
    # 4. Verificar labels únicos
    print("\n4. Verificando labels únicos...")
    # Agrupar por label
    label_to_paths = defaultdict(list)
    for path, label in zip(loader.image_paths, loader.labels):
        label_to_paths[label].append(path)
    
    # Verificar se cada label tem imagens do mesmo subdiretório e origem
    errors = []
    for label, paths in label_to_paths.items():
        label_name = loader.label_names[label]
        subdirs = set(p.parent.name for p in paths)
        datasets = set(p.parent.parent.name for p in paths)
        origens = set(p.stem.split('_')[0] for p in paths)
        
        if len(subdirs) > 1:
            errors.append(f"Label {label_name}: múltiplos subdiretórios {subdirs}")
        if len(datasets) > 1:
            errors.append(f"Label {label_name}: múltiplos datasets {datasets}")
        if len(origens) > 1:
            errors.append(f"Label {label_name}: múltiplas origens {origens}")
    
    if errors:
        print(f"   ❌ {len(errors)} erros encontrados:")
        for err in errors[:5]:
            print(f"      - {err}")
    else:
        print(f"   ✓ Todos os {len(label_to_paths)} labels estão corretos")
    
    # 5. Testar divisão train/val/test
    print("\n5. Testando divisão train/val/test...")
    train_idx, val_idx, test_idx = loader.get_split_indices(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    print(f"   Train: {len(train_idx)} imagens")
    print(f"   Val: {len(val_idx)} imagens")
    print(f"   Test: {len(test_idx)} imagens")
    print(f"   Total: {len(train_idx) + len(val_idx) + len(test_idx)} imagens")
    
    # 6. Verificar que não há vazamento de dados (mesma origem em splits diferentes)
    print("\n6. Verificando vazamento de dados entre splits...")
    train_labels = set(loader.labels[i] for i in train_idx)
    val_labels = set(loader.labels[i] for i in val_idx)
    test_labels = set(loader.labels[i] for i in test_idx)
    
    train_val_overlap = train_labels & val_labels
    train_test_overlap = train_labels & test_labels
    val_test_overlap = val_labels & test_labels
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"   ❌ VAZAMENTO DETECTADO!")
        print(f"      Train-Val overlap: {len(train_val_overlap)} labels")
        print(f"      Train-Test overlap: {len(train_test_overlap)} labels")
        print(f"      Val-Test overlap: {len(val_test_overlap)} labels")
        return False
    else:
        print(f"   ✓ Nenhum vazamento de dados detectado")
        print(f"      Train: {len(train_labels)} origens únicas")
        print(f"      Val: {len(val_labels)} origens únicas")
        print(f"      Test: {len(test_labels)} origens únicas")
    
    # 7. Testar reprodutibilidade
    print("\n7. Testando reprodutibilidade da divisão...")
    train_idx2, val_idx2, test_idx2 = loader.get_split_indices(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    if train_idx == train_idx2 and val_idx == val_idx2 and test_idx == test_idx2:
        print("   ✓ Divisão é reproduzível (mesma seed = mesmos índices)")
    else:
        print("   ❌ Divisão NÃO é reproduzível!")
        return False
    
    # 8. Criar datasets e verificar
    print("\n8. Criando datasets...")
    train_ds, val_ds, test_ds = loader.create_datasets(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment_train=False,
    )
    
    print(f"   Train dataset: {len(train_ds)} imagens")
    print(f"   Val dataset: {len(val_ds)} imagens")
    print(f"   Test dataset: {len(test_ds)} imagens")
    
    # 9. Testar carregamento de uma imagem
    print("\n9. Testando carregamento de imagem...")
    try:
        image, label = train_ds[0]
        print(f"   ✓ Imagem carregada: shape={image.shape}, dtype={image.dtype}, label={label}")
        print(f"      Valores: min={image.min():.3f}, max={image.max():.3f}")
    except Exception as e:
        print(f"   ❌ Erro ao carregar imagem: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ TODOS OS TESTES PASSARAM!")
    print("=" * 80)
    
    return True


def print_sample_labels(n=10):
    """Imprimir exemplos de labels para verificação manual."""
    print("\n" + "=" * 80)
    print("EXEMPLOS DE LABELS")
    print("=" * 80)
    
    loader = FVCDatasetLoader(
        datasets=["FVC2000", "FVC2002", "FVC2004"],
        random_state=42,
    )
    
    print(f"\nPrimeiros {n} arquivos e seus labels:")
    print("-" * 80)
    print(f"{'Arquivo':<60} {'Label':<6} {'Nome do Label'}")
    print("-" * 80)
    
    for i in range(min(n, len(loader.image_paths))):
        path = loader.image_paths[i]
        label = loader.labels[i]
        label_name = loader.label_names[label]
        rel_path = f"{path.parent.parent.name}/{path.parent.name}/{path.name}"
        print(f"{rel_path:<60} {label:<6} {label_name}")
    
    # Mostrar alguns de cada subdiretório
    print(f"\n\nExemplos de cada subdiretório (1 por DB):")
    print("-" * 80)
    
    seen_subdirs = set()
    for i, path in enumerate(loader.image_paths):
        subdir_key = f"{path.parent.parent.name}/{path.parent.name}"
        if subdir_key not in seen_subdirs:
            seen_subdirs.add(subdir_key)
            label = loader.labels[i]
            label_name = loader.label_names[label]
            print(f"{path.name:<20} em {subdir_key:<20} -> label={label} ({label_name})")


if __name__ == "__main__":
    success = test_fvc_loader()
    
    if success:
        print_sample_labels(20)
    
    sys.exit(0 if success else 1)
