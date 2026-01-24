"""
Script de teste para validar carregamento da base SD302
"""

import logging
from pathlib import Path
from data_loader import SD302DatasetLoader, load_datasets

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_sd302_loader_only():
    """Testar apenas o loader SD302"""
    logger.info("=" * 80)
    logger.info("TESTE 1: Carregamento apenas da base SD302")
    logger.info("=" * 80)
    
    try:
        loader = SD302DatasetLoader(random_state=42)
        
        # Estatísticas
        stats = loader.get_statistics()
        logger.info(f"Total de imagens: {stats['total_images']}")
        logger.info(f"Total de origens únicas: {stats['total_origins']}")
        logger.info(f"Imagens por device: {stats['images_per_device']}")
        logger.info(f"Versões por origem:")
        logger.info(f"  - Mínimo: {stats['versions_per_origin']['min']}")
        logger.info(f"  - Máximo: {stats['versions_per_origin']['max']}")
        logger.info(f"  - Média: {stats['versions_per_origin']['mean']:.2f}")
        
        # Criar datasets
        train_ds, val_ds, test_ds = loader.create_datasets()
        logger.info(f"\nDatasets criados:")
        logger.info(f"  - Train: {len(train_ds)} amostras")
        logger.info(f"  - Val: {len(val_ds)} amostras")
        logger.info(f"  - Test: {len(test_ds)} amostras")
        
        # Testar carregamento de uma imagem
        logger.info(f"\nTestando carregamento de imagem...")
        img, label = train_ds[0]
        logger.info(f"  - Shape: {img.shape}")
        logger.info(f"  - Label: {label}")
        logger.info(f"  - Min/Max: {img.min():.3f} / {img.max():.3f}")
        
        # Mostrar algumas origens
        logger.info(f"\nPrimeiras 10 origens únicas:")
        for i, name in enumerate(loader.label_names[:10]):
            logger.info(f"  {i}: {name}")
        
        logger.info("\n✓ TESTE 1 PASSOU\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TESTE 1 FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_loaders():
    """Testar carregamento combinado (FVC + SD302)"""
    logger.info("=" * 80)
    logger.info("TESTE 2: Carregamento combinado (FVC2000 + SD302)")
    logger.info("=" * 80)
    
    try:
        train_ds, val_ds, test_ds, loaders = load_datasets(
            datasets=["FVC2000", "SD302"],
            random_state=42,
        )
        
        # Estatísticas por loader
        total_images = 0
        total_origins = 0
        
        for loader_name, loader in loaders.items():
            stats = loader.get_statistics()
            logger.info(f"\n=== {loader_name} ===")
            logger.info(f"Imagens: {stats['total_images']}")
            logger.info(f"Origens: {stats['total_origins']}")
            
            total_images += stats['total_images']
            total_origins += stats['total_origins']
        
        logger.info(f"\n=== TOTAL COMBINADO ===")
        logger.info(f"Total de imagens: {total_images}")
        logger.info(f"Total de origens: {total_origins}")
        logger.info(f"\nDatasets criados:")
        logger.info(f"  - Train: {len(train_ds)} amostras")
        logger.info(f"  - Val: {len(val_ds)} amostras")
        logger.info(f"  - Test: {len(test_ds)} amostras")
        
        # Verificar que labels não se sobrepõem
        train_labels = set(train_ds.get_labels())
        val_labels = set(val_ds.get_labels())
        test_labels = set(test_ds.get_labels())
        
        logger.info(f"\nLabels únicos:")
        logger.info(f"  - Train: {len(train_labels)}")
        logger.info(f"  - Val: {len(val_labels)}")
        logger.info(f"  - Test: {len(test_labels)}")
        
        # Verificar split
        overlap_train_val = train_labels.intersection(val_labels)
        overlap_train_test = train_labels.intersection(test_labels)
        overlap_val_test = val_labels.intersection(test_labels)
        
        logger.info(f"\nVerificação de split:")
        logger.info(f"  - Train ∩ Val: {len(overlap_train_val)} (deve ser 0)")
        logger.info(f"  - Train ∩ Test: {len(overlap_train_test)} (deve ser 0)")
        logger.info(f"  - Val ∩ Test: {len(overlap_val_test)} (deve ser 0)")
        
        if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
            logger.info("  ✓ Split correto: sem sobreposição entre conjuntos")
        else:
            logger.warning("  ✗ ATENÇÃO: há sobreposição entre conjuntos!")
        
        # Testar carregamento
        logger.info(f"\nTestando carregamento de imagens...")
        img, label = train_ds[0]
        logger.info(f"  - Shape: {img.shape}")
        logger.info(f"  - Label: {label}")
        
        logger.info("\n✓ TESTE 2 PASSOU\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TESTE 2 FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_datasets():
    """Testar carregamento de todas as bases"""
    logger.info("=" * 80)
    logger.info("TESTE 3: Carregamento de todas as bases (FVC2000, FVC2002, FVC2004, SD302)")
    logger.info("=" * 80)
    
    try:
        train_ds, val_ds, test_ds, loaders = load_datasets(
            datasets=["FVC2000", "FVC2002", "FVC2004", "SD302"],
            random_state=42,
        )
        
        # Estatísticas por loader
        total_images = 0
        total_origins = 0
        
        for loader_name, loader in loaders.items():
            stats = loader.get_statistics()
            logger.info(f"\n=== {loader_name} ===")
            logger.info(f"Imagens: {stats['total_images']}")
            logger.info(f"Origens: {stats['total_origins']}")
            
            total_images += stats['total_images']
            total_origins += stats['total_origins']
        
        logger.info(f"\n=== TOTAL COMBINADO ===")
        logger.info(f"Total de imagens: {total_images}")
        logger.info(f"Total de origens: {total_origins}")
        logger.info(f"\nDatasets criados:")
        logger.info(f"  - Train: {len(train_ds)} amostras")
        logger.info(f"  - Val: {len(val_ds)} amostras")
        logger.info(f"  - Test: {len(test_ds)} amostras")
        
        logger.info("\n✓ TESTE 3 PASSOU\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TESTE 3 FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("INICIANDO TESTES DE CARREGAMENTO DE DADOS")
    logger.info("=" * 80 + "\n")
    
    results = []
    
    # Teste 1: SD302 apenas
    results.append(("SD302 apenas", test_sd302_loader_only()))
    
    # Teste 2: FVC2000 + SD302
    results.append(("FVC2000 + SD302", test_combined_loaders()))
    
    # Teste 3: Todas as bases
    results.append(("Todas as bases", test_all_datasets()))
    
    # Resumo
    logger.info("=" * 80)
    logger.info("RESUMO DOS TESTES")
    logger.info("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSOU" if passed else "✗ FALHOU"
        logger.info(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    logger.info(f"\nResultado: {total_passed}/{total_tests} testes passaram")
    
    if total_passed == total_tests:
        logger.info("✓ TODOS OS TESTES PASSARAM!")
    else:
        logger.info("✗ ALGUNS TESTES FALHARAM")
