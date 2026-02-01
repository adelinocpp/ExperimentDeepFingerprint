"""Módulo para carregamento de dados de impressões digitais

Estrutura esperada das bases FVC:
    FVC2000/
    ├── DB1_B/
    │   ├── 101_1.png  (origem 101, versão 1)
    │   ├── 101_2.png  (origem 101, versão 2)
    │   └── ...
    ├── DB2_B/
    ├── DB3_B/
    └── DB4_B/

O label único é formado por: {dataset}_{subdir}_{origem}
Exemplo: FVC2000_DB1_B_101
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import cv2
from PIL import Image
import logging
import re
from sklearn.model_selection import train_test_split

from config import DATA_DIR, AUGMENTATION_CONFIG, AGGRESSIVE_AUGMENTATION_CONFIG
import glob


class FingerprintDataset(Dataset):
    """Dataset para impressões digitais com suporte às bases FVC"""
    
    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        image_size: Tuple[int, int] = (299, 299),
        augment: bool = False,
        augmentation_config: Dict = None,
        random_state: int = 42,
        return_paths: bool = True,
    ):
        """
        Args:
            image_paths: lista de caminhos das imagens
            labels: lista de labels
            image_size: tamanho da imagem após resize
            augment: se True, aplica data augmentation
            augmentation_config: configurações de augmentation
            random_state: seed para reprodutibilidade
            return_paths: se True, retorna (image, label, path); caso contrário (image, label)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.random_state = random_state
        self.return_paths = return_paths
        np.random.seed(random_state)
        
        # Selecionar config de augmentation
        if self.augmentation_config.get("aggressive", False):
            self.aug_config = AGGRESSIVE_AUGMENTATION_CONFIG
        else:
            self.aug_config = AUGMENTATION_CONFIG
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Dataset criado: {len(self.image_paths)} imagens")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def get_file_paths(self) -> List[str]:
        """Retorna lista de caminhos dos arquivos."""
        return [str(p) for p in self.image_paths]
    
    def get_labels(self) -> List[int]:
        """Retorna lista de labels."""
        return list(self.labels)
    
    def __getitem__(self, idx: int):
        """
        Args:
            idx: índice
        
        Returns:
            Se return_paths=True: (image, label, path)
            Se return_paths=False: (image, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Carregar imagem
        image = self._load_image(image_path)
        
        # Redimensionar COM PADDING (igual ao DeepPrint original)
        # Isso mantém o aspect ratio e evita distorção
        image = self._pad_and_resize(image, self.image_size, fill=255)
        
        # Normalizar para [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Augmentation
        if self.augment:
            image = self._augment_image(image)
        
        # Garantir float32 (augmentation pode retornar float64)
        image = image.astype(np.float32)
        
        # Converter para tensor
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        
        if self.return_paths:
            return image, label, str(image_path)
        else:
            return image, label
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Carregar imagem"""
        if image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Não foi possível carregar imagem: {image_path}")
            return image
        else:
            raise ValueError(f"Formato de imagem não suportado: {image_path.suffix}")
    
    def _pad_and_resize(self, image: np.ndarray, target_size: Tuple[int, int], fill: int = 255) -> np.ndarray:
        """Pad para quadrado e resize mantendo aspect ratio (igual ao DeepPrint original)
        
        Args:
            image: imagem grayscale (H, W)
            target_size: tamanho alvo (H, W)
            fill: valor para padding (255=branco para impressões digitais)
        
        Returns:
            imagem redimensionada (target_H, target_W)
        """
        h, w = image.shape
        
        # Calcular padding para tornar quadrado
        if w >= h:
            pad_h = (w - h) // 2
            pad_w = 0
        else:
            pad_h = 0
            pad_w = (h - w) // 2
        
        # Aplicar padding
        padded = cv2.copyMakeBorder(
            image,
            top=pad_h,
            bottom=pad_h + (1 if (w - h) % 2 != 0 and w > h else 0),
            left=pad_w,
            right=pad_w + (1 if (h - w) % 2 != 0 and h > w else 0),
            borderType=cv2.BORDER_CONSTANT,
            value=fill
        )
        
        # Resize para o tamanho alvo
        resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Aplicar augmentation à imagem (usa self.aug_config)"""
        cfg = self.aug_config
        h, w = image.shape
        
        # Rotação aleatória
        angle = np.random.uniform(-cfg["rotation_range"], cfg["rotation_range"])
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Translação aleatória
        tx = np.random.uniform(-cfg["translation_range"], cfg["translation_range"])
        ty = np.random.uniform(-cfg["translation_range"], cfg["translation_range"])
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Ajuste de contraste e brilho
        if cfg.get("quality_augmentation", False):
            contrast = np.random.uniform(cfg["contrast_range"][0], cfg["contrast_range"][1])
            brightness = np.random.uniform(cfg["brightness_range"][0], cfg["brightness_range"][1])
            image = np.clip(image * contrast + (brightness - 1) * 0.5, 0, 1)
        
        # === Augmentations agressivos (apenas se configurado) ===
        
        # Elastic deformation
        if cfg.get("elastic_deformation", False):
            image = self._elastic_deformation(image, cfg["elastic_alpha"], cfg["elastic_sigma"])
        
        # Gaussian noise
        if cfg.get("gaussian_noise", False):
            noise = np.random.normal(0, cfg["noise_std"], image.shape)
            image = np.clip(image + noise, 0, 1)
        
        # Random erasing
        if cfg.get("random_erasing", False) and np.random.random() < cfg["erasing_prob"]:
            image = self._random_erasing(image, cfg["erasing_scale"])
        
        return image
    
    def _elastic_deformation(self, image: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
        """Aplicar deformação elástica à imagem"""
        h, w = image.shape
        
        # Gerar campos de deslocamento aleatórios
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        
        # Criar grid de coordenadas
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Aplicar remapeamento
        return cv2.remap(image.astype(np.float32), map_x, map_y, 
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def _random_erasing(self, image: np.ndarray, scale: Tuple[float, float]) -> np.ndarray:
        """Apagar região aleatória da imagem"""
        h, w = image.shape
        area = h * w
        
        # Calcular tamanho da região a apagar
        erase_area = np.random.uniform(scale[0], scale[1]) * area
        aspect_ratio = np.random.uniform(0.3, 3.0)
        
        eh = int(np.sqrt(erase_area * aspect_ratio))
        ew = int(np.sqrt(erase_area / aspect_ratio))
        
        if eh < h and ew < w:
            x = np.random.randint(0, w - ew)
            y = np.random.randint(0, h - eh)
            image[y:y+eh, x:x+ew] = np.random.uniform(0, 1)
        
        return image


class DummyFingerprintDataset(Dataset):
    """Dataset dummy para testes"""
    
    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 10,
        image_size: Tuple[int, int] = (299, 299),
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Gerar dados dummy
        self.images = np.random.randn(num_samples, *image_size).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, num_samples)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        label = int(self.labels[idx])
        return image, label


def create_dummy_dataset(
    num_samples: int = 100,
    num_classes: int = 10,
    image_size: Tuple[int, int] = (299, 299),
) -> DummyFingerprintDataset:
    """Criar dataset dummy para testes"""
    return DummyFingerprintDataset(num_samples, num_classes, image_size)


class FVCDatasetLoader:
    """
    Carregador de datasets FVC (FVC2000, FVC2002, FVC2004).
    
    Estrutura esperada:
        FVC2000/
        ├── DB1_B/
        │   ├── 101_1.png  (origem 101, versão 1)
        │   ├── 101_2.png  (origem 101, versão 2)
        │   └── ...
        ├── DB2_B/
        ├── DB3_B/
        └── DB4_B/
    
    O label único é formado por: {dataset}_{subdir}_{origem}
    Exemplo: FVC2000_DB1_B_101
    """
    
    # Subdiretórios padrão das bases FVC
    FVC_SUBDIRS = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]
    
    def __init__(
        self,
        datasets: List[str] = None,
        random_state: int = 42,
        image_size: Tuple[int, int] = (299, 299),
    ):
        """
        Args:
            datasets: lista de datasets a carregar (ex: ["FVC2000", "FVC2002", "FVC2004"])
                     Se None, carrega todos os disponíveis
            random_state: seed para reprodutibilidade
            image_size: tamanho das imagens
        """
        self.datasets = datasets or ["FVC2000", "FVC2002", "FVC2004"]
        self.random_state = random_state
        self.image_size = image_size
        
        self.logger = logging.getLogger(__name__)
        
        # Dados carregados
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.label_names: List[str] = []  # Nomes legíveis dos labels
        self.label_map: Dict[str, int] = {}  # Mapeamento nome -> int
        
        # Carregar dados
        self._load_all_datasets()
    
    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Extrair origem e versão do nome do arquivo.
        
        Args:
            filename: nome do arquivo (ex: "101_1.png")
        
        Returns:
            (origem, versão): tupla com origem e versão
        """
        match = re.match(r"(\d+)_(\d+)\.(png|jpg|jpeg)$", filename, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def _load_all_datasets(self):
        """Carregar todos os datasets especificados."""
        current_label = 0
        
        for dataset_name in self.datasets:
            dataset_dir = DATA_DIR / dataset_name
            
            if not dataset_dir.exists():
                self.logger.warning(f"Dataset não encontrado: {dataset_dir}")
                continue
            
            for subdir in self.FVC_SUBDIRS:
                subdir_path = dataset_dir / subdir
                
                if not subdir_path.exists():
                    self.logger.warning(f"Subdiretório não encontrado: {subdir_path}")
                    continue
                
                # Encontrar todas as imagens
                image_files = sorted(subdir_path.glob("*.png")) + sorted(subdir_path.glob("*.jpg"))
                
                for image_path in image_files:
                    origem, versao = self._parse_filename(image_path.name)
                    
                    if origem is None:
                        self.logger.warning(f"Arquivo com nome inválido: {image_path}")
                        continue
                    
                    # Criar label único: dataset_subdir_origem
                    label_name = f"{dataset_name}_{subdir}_{origem}"
                    
                    if label_name not in self.label_map:
                        self.label_map[label_name] = current_label
                        self.label_names.append(label_name)
                        current_label += 1
                    
                    self.image_paths.append(image_path)
                    self.labels.append(self.label_map[label_name])
        
        self.logger.info(
            f"Carregados {len(self.image_paths)} imagens de "
            f"{len(self.label_map)} origens únicas"
        )
    
    @property
    def num_classes(self) -> int:
        """Retorna o número total de classes (origens únicas) no dataset."""
        return len(self.label_map)
    
    def get_split_indices(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Dividir índices em train/val/test de forma estratificada e reproduzível.
        
        A divisão é feita por ORIGEM (label), não por imagem individual.
        Isso garante que todas as versões de uma mesma origem fiquem no mesmo split.
        
        Args:
            train_ratio: proporção para treino
            val_ratio: proporção para validação
            test_ratio: proporção para teste
        
        Returns:
            (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "As proporções devem somar 1.0"
        
        # Agrupar índices por label (origem)
        label_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        # Dividir labels em train/val/test
        unique_labels = list(label_to_indices.keys())
        
        # Primeira divisão: train vs (val+test)
        train_labels, temp_labels = train_test_split(
            unique_labels,
            train_size=train_ratio,
            random_state=self.random_state,
        )
        
        # Segunda divisão: val vs test
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_labels, test_labels = train_test_split(
            temp_labels,
            train_size=val_size_adjusted,
            random_state=self.random_state,
        )
        
        # Coletar índices de cada split
        train_indices = []
        val_indices = []
        test_indices = []
        
        for label in train_labels:
            train_indices.extend(label_to_indices[label])
        for label in val_labels:
            val_indices.extend(label_to_indices[label])
        for label in test_labels:
            test_indices.extend(label_to_indices[label])
        
        self.logger.info(
            f"Split: train={len(train_indices)} ({len(train_labels)} origens), "
            f"val={len(val_indices)} ({len(val_labels)} origens), "
            f"test={len(test_indices)} ({len(test_labels)} origens)"
        )
        
        return train_indices, val_indices, test_indices
    
    def create_datasets(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        augment_train: bool = True,
    ) -> Tuple[FingerprintDataset, FingerprintDataset, FingerprintDataset]:
        """
        Criar datasets de treino, validação e teste.
        
        Args:
            train_ratio: proporção para treino
            val_ratio: proporção para validação
            test_ratio: proporção para teste
            augment_train: aplicar augmentation no treino
        
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        train_idx, val_idx, test_idx = self.get_split_indices(
            train_ratio, val_ratio, test_ratio
        )
        
        train_dataset = FingerprintDataset(
            image_paths=[self.image_paths[i] for i in train_idx],
            labels=[self.labels[i] for i in train_idx],
            image_size=self.image_size,
            augment=augment_train,
        )
        
        val_dataset = FingerprintDataset(
            image_paths=[self.image_paths[i] for i in val_idx],
            labels=[self.labels[i] for i in val_idx],
            image_size=self.image_size,
            augment=False,
        )
        
        test_dataset = FingerprintDataset(
            image_paths=[self.image_paths[i] for i in test_idx],
            labels=[self.labels[i] for i in test_idx],
            image_size=self.image_size,
            augment=False,
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_full_dataset(self, augment: bool = False) -> FingerprintDataset:
        """
        Criar dataset com todas as imagens (sem divisão).
        
        Args:
            augment: aplicar augmentation
        
        Returns:
            dataset completo
        """
        return FingerprintDataset(
            image_paths=self.image_paths,
            labels=self.labels,
            image_size=self.image_size,
            augment=augment,
        )
    
    def get_statistics(self) -> Dict:
        """Retornar estatísticas do dataset."""
        # Contar imagens por dataset
        dataset_counts = {}
        for path in self.image_paths:
            dataset_name = path.parent.parent.name
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        
        # Contar imagens por subdiretório
        subdir_counts = {}
        for path in self.image_paths:
            subdir = path.parent.name
            subdir_counts[subdir] = subdir_counts.get(subdir, 0) + 1
        
        # Contar versões por origem
        versions_per_origin = {}
        for label in self.labels:
            versions_per_origin[label] = versions_per_origin.get(label, 0) + 1
        
        return {
            "total_images": len(self.image_paths),
            "total_origins": len(self.label_map),
            "images_per_dataset": dataset_counts,
            "images_per_subdir": subdir_counts,
            "versions_per_origin": {
                "min": min(versions_per_origin.values()) if versions_per_origin else 0,
                "max": max(versions_per_origin.values()) if versions_per_origin else 0,
                "mean": np.mean(list(versions_per_origin.values())) if versions_per_origin else 0,
            },
        }


class SFingeDatasetLoader:
    """
    Carregador de dataset derivado do SFinge (FP_gen_0 e FP_gen_1).
    
    Estrutura esperada:
        FP_gen_0/
        ├── fingerprint_0001_v01.png  (origem 0, versão 1)
        ├── fingerprint_0002_v02.png  (origem 0, versão 2)
        ├── ...
        FP_gen_1/
        ├── fingerprint_6001_v01.png
        └── ...
    
    Padrão: fingerprint_NNNN_vXX.png
    Origem = (NNNN - 1) // 10
    Versão = vXX (01-10)
    
    Carrega ambos FP_gen_0 e FP_gen_1 simultaneamente.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        image_size: Tuple[int, int] = (299, 299),
        use_both_gen: bool = True,
    ):
        self.random_state = random_state
        self.image_size = image_size
        self.use_both_gen = use_both_gen
        
        self.logger = logging.getLogger(__name__)
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.label_names: List[str] = []
        self.label_map: Dict[str, int] = {}
        
        self._load_dataset()
    
    def _parse_filename(self, filename: str) -> Tuple[int, int]:
        """
        Extrair origem e versão do nome do arquivo.
        
        Args:
            filename: nome do arquivo (ex: "fingerprint_0001_v01.png")
        
        Returns:
            (origem, versão): tupla com ID do dedo e versão
                - origem: ID do dedo (0-5999 para FP_gen_0, 6000-7999 para FP_gen_1)
                - versão: número da versão (1-10 ou 1-12)
        """
        match = re.match(r"fingerprint_(\d+)_v(\d+)\.png$", filename, re.IGNORECASE)
        if match:
            finger_id = int(match.group(1))  # ID do dedo (NNNN)
            version = int(match.group(2))     # Versão (XX)
            return finger_id, version
        return None, None
    
    def _load_dataset(self):
        """Carregar datasets FP_gen_0 e FP_gen_1."""
        datasets_to_load = ["FP_gen_0"]
        if self.use_both_gen:
            datasets_to_load.append("FP_gen_1")
        
        current_label = 0
        total_images = 0
        
        # Mapeamento dataset -> origem para split open-set
        self.dataset_source = []  # Rastrear qual dataset (FP_gen_0 ou FP_gen_1)
        
        for dataset_name in datasets_to_load:
            dataset_dir = DATA_DIR / dataset_name
            
            if not dataset_dir.exists():
                self.logger.warning(f"Dataset {dataset_name} não encontrado: {dataset_dir}")
                continue
            
            image_files = sorted(dataset_dir.glob("*.png"))
            self.logger.info(f"Encontradas {len(image_files)} imagens em {dataset_name}")
            
            for image_path in image_files:
                origin, version = self._parse_filename(image_path.name)
                
                if origin is None:
                    self.logger.warning(f"Arquivo com nome inválido: {image_path}")
                    continue
                
                label_name = f"SFinge_{origin:04d}"
                
                if label_name not in self.label_map:
                    self.label_map[label_name] = current_label
                    self.label_names.append(label_name)
                    current_label += 1
                
                self.image_paths.append(image_path)
                self.labels.append(self.label_map[label_name])
                self.dataset_source.append(dataset_name)  # Rastrear origem
                total_images += 1
        
        self.logger.info(
            f"SFinge (FP_gen_0 + FP_gen_1): {total_images} imagens, "
            f"{len(self.label_map)} origens únicas"
        )
    
    def get_split_indices(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        use_openset: bool = True,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Dividir dataset por origem.
        
        Args:
            train_ratio: Proporção de treino (usado apenas em closed-set)
            val_ratio: Proporção de validação (usado apenas em closed-set)
            test_ratio: Proporção de teste (usado apenas em closed-set)
            use_openset: Se True, usa FP_gen_0 para treino e FP_gen_1 para val/test
                        (igual ao paper original - OPEN-SET)
                        Se False, faz split dentro de todas as classes (CLOSED-SET)
        
        Returns:
            train_indices, val_indices, test_indices
        """
        if use_openset and self.use_both_gen:
            # OPEN-SET: FP_gen_0 (6000 classes) = treino
            #           FP_gen_1 (2000 classes) = validação + teste
            train_indices = []
            temp_indices = []
            
            for idx, source in enumerate(self.dataset_source):
                if source == "FP_gen_0":
                    train_indices.append(idx)
                else:  # FP_gen_1
                    temp_indices.append(idx)
            
            # Split FP_gen_1 em val e test (50/50)
            val_indices = temp_indices[:len(temp_indices)//2]
            test_indices = temp_indices[len(temp_indices)//2:]
            
            # Contar classes únicas em cada split
            train_classes = len(set(self.labels[i] for i in train_indices))
            val_classes = len(set(self.labels[i] for i in val_indices))
            test_classes = len(set(self.labels[i] for i in test_indices))
            
            self.logger.info(
                f"OPEN-SET Split: train={len(train_indices)} ({train_classes} classes únicas), "
                f"val={len(val_indices)} ({val_classes} classes únicas), "
                f"test={len(test_indices)} ({test_classes} classes únicas)"
            )
            self.logger.info(
                f"Treino usa FP_gen_0 (6000 classes), Val/Test usam FP_gen_1 (2000 classes DISJUNTAS)"
            )
            
            return train_indices, val_indices, test_indices
        
        else:
            # CLOSED-SET: Split tradicional dentro das mesmas classes
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
            
            label_to_indices: Dict[int, List[int]] = {}
            for idx, label in enumerate(self.labels):
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(idx)
            
            unique_labels = list(label_to_indices.keys())
            
            train_labels, temp_labels = train_test_split(
                unique_labels, train_size=train_ratio, random_state=self.random_state
            )
            
            val_size_adjusted = val_ratio / (val_ratio + test_ratio)
            val_labels, test_labels = train_test_split(
                temp_labels, train_size=val_size_adjusted, random_state=self.random_state
            )
            
            train_indices = []
            val_indices = []
            test_indices = []
            
            for label in train_labels:
                train_indices.extend(label_to_indices[label])
            for label in val_labels:
                val_indices.extend(label_to_indices[label])
            for label in test_labels:
                test_indices.extend(label_to_indices[label])
            
            self.logger.info(
                f"CLOSED-SET Split: train={len(train_indices)} ({len(train_labels)} classes), "
                f"val={len(val_indices)} ({len(val_labels)} classes), "
                f"test={len(test_indices)} ({len(test_labels)} classes)"
            )
            
            return train_indices, val_indices, test_indices
    
    def get_statistics(self) -> Dict:
        """Retornar estatísticas do dataset."""
        versions_per_origin = {}
        for label in self.labels:
            versions_per_origin[label] = versions_per_origin.get(label, 0) + 1
        
        return {
            "total_images": len(self.image_paths),
            "total_origins": len(self.label_map),
            "versions_per_origin": {
                "min": min(versions_per_origin.values()) if versions_per_origin else 0,
                "max": max(versions_per_origin.values()) if versions_per_origin else 0,
                "mean": np.mean(list(versions_per_origin.values())) if versions_per_origin else 0,
            },
        }


class SD302DatasetLoader:
    """
    Carregador de dataset NIST SD302.
    
    Estrutura esperada:
        SD302/
        ├── A/
        │   └── roll/
        │       └── png/
        │           ├── 00002502_A_roll_01.png  (subject 00002502, device A, finger 01)
        │           ├── 00002502_A_roll_02.png
        │           └── ...
        ├── B/
        ├── C/
        └── ...
    
    O label único é formado por: {subject}_{frgp}
    Exemplo: 00002502_01 (subject 00002502, dedo 01)
    
    Diferentes devices capturando o mesmo subject+finger são versões diferentes.
    """
    
    SD302_DEVICES = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    def __init__(
        self,
        random_state: int = 42,
        image_size: Tuple[int, int] = (299, 299),
    ):
        """
        Args:
            random_state: seed para reprodutibilidade
            image_size: tamanho das imagens
        """
        self.random_state = random_state
        self.image_size = image_size
        
        self.logger = logging.getLogger(__name__)
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.label_names: List[str] = []
        self.label_map: Dict[str, int] = {}
        
        self._load_dataset()
    
    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Extrair subject, device e frgp do nome do arquivo.
        
        Args:
            filename: nome do arquivo (ex: "00002502_A_roll_01.png")
        
        Returns:
            (subject, device, frgp): tupla com subject, device e frgp
        """
        match = re.match(r"(\d+)_([A-H])_roll_(\d+)\.(png|jpg|jpeg)$", filename, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None, None, None
    
    def _load_dataset(self):
        """Carregar dataset SD302."""
        sd302_dir = DATA_DIR / "SD302"
        
        if not sd302_dir.exists():
            self.logger.warning(f"Dataset SD302 não encontrado: {sd302_dir}")
            return
        
        current_label = 0
        
        for device in self.SD302_DEVICES:
            device_dir = sd302_dir / device / "roll" / "png"
            
            if not device_dir.exists():
                self.logger.debug(f"Device dir não encontrado: {device_dir}")
                continue
            
            image_files = sorted(device_dir.glob("*.png")) + sorted(device_dir.glob("*.jpg"))
            
            for image_path in image_files:
                subject, dev, frgp = self._parse_filename(image_path.name)
                
                if subject is None:
                    self.logger.warning(f"Arquivo com nome inválido: {image_path}")
                    continue
                
                label_name = f"SD302_{subject}_{frgp}"
                
                if label_name not in self.label_map:
                    self.label_map[label_name] = current_label
                    self.label_names.append(label_name)
                    current_label += 1
                
                self.image_paths.append(image_path)
                self.labels.append(self.label_map[label_name])
        
        self.logger.info(
            f"SD302: Carregados {len(self.image_paths)} imagens de "
            f"{len(self.label_map)} origens únicas"
        )
    
    def get_split_indices(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Dividir índices em train/val/test de forma estratificada.
        
        A divisão é feita por ORIGEM (label), não por imagem individual.
        
        Args:
            train_ratio: proporção para treino
            val_ratio: proporção para validação
            test_ratio: proporção para teste
        
        Returns:
            (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "As proporções devem somar 1.0"
        
        label_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        unique_labels = list(label_to_indices.keys())
        
        train_labels, temp_labels = train_test_split(
            unique_labels,
            train_size=train_ratio,
            random_state=self.random_state,
        )
        
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_labels, test_labels = train_test_split(
            temp_labels,
            train_size=val_size_adjusted,
            random_state=self.random_state,
        )
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for label in train_labels:
            train_indices.extend(label_to_indices[label])
        for label in val_labels:
            val_indices.extend(label_to_indices[label])
        for label in test_labels:
            test_indices.extend(label_to_indices[label])
        
        self.logger.info(
            f"SD302 Split: train={len(train_indices)} ({len(train_labels)} origens), "
            f"val={len(val_indices)} ({len(val_labels)} origens), "
            f"test={len(test_indices)} ({len(test_labels)} origens)"
        )
        
        return train_indices, val_indices, test_indices
    
    def create_datasets(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        augment_train: bool = True,
    ) -> Tuple[FingerprintDataset, FingerprintDataset, FingerprintDataset]:
        """
        Criar datasets de treino, validação e teste.
        
        Args:
            train_ratio: proporção para treino
            val_ratio: proporção para validação
            test_ratio: proporção para teste
            augment_train: aplicar augmentation no treino
        
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        train_idx, val_idx, test_idx = self.get_split_indices(
            train_ratio, val_ratio, test_ratio
        )
        
        train_dataset = FingerprintDataset(
            image_paths=[self.image_paths[i] for i in train_idx],
            labels=[self.labels[i] for i in train_idx],
            image_size=self.image_size,
            augment=augment_train,
        )
        
        val_dataset = FingerprintDataset(
            image_paths=[self.image_paths[i] for i in val_idx],
            labels=[self.labels[i] for i in val_idx],
            image_size=self.image_size,
            augment=False,
        )
        
        test_dataset = FingerprintDataset(
            image_paths=[self.image_paths[i] for i in test_idx],
            labels=[self.labels[i] for i in test_idx],
            image_size=self.image_size,
            augment=False,
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_statistics(self) -> Dict:
        """Retornar estatísticas do dataset."""
        device_counts = {}
        for path in self.image_paths:
            _, device, _ = self._parse_filename(path.name)
            if device:
                device_counts[device] = device_counts.get(device, 0) + 1
        
        versions_per_origin = {}
        for label in self.labels:
            versions_per_origin[label] = versions_per_origin.get(label, 0) + 1
        
        return {
            "total_images": len(self.image_paths),
            "total_origins": len(self.label_map),
            "images_per_device": device_counts,
            "versions_per_origin": {
                "min": min(versions_per_origin.values()) if versions_per_origin else 0,
                "max": max(versions_per_origin.values()) if versions_per_origin else 0,
                "mean": np.mean(list(versions_per_origin.values())) if versions_per_origin else 0,
            },
        }


def load_fvc_datasets(
    datasets: List[str] = None,
    random_state: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    augment_train: bool = True,
    image_size: Tuple[int, int] = (299, 299),
) -> Tuple[FingerprintDataset, FingerprintDataset, FingerprintDataset, FVCDatasetLoader]:
    """
    Função de conveniência para carregar datasets FVC.
    
    Args:
        datasets: lista de datasets (ex: ["FVC2000", "FVC2002", "FVC2004"])
        random_state: seed para reprodutibilidade
        train_ratio: proporção para treino
        val_ratio: proporção para validação
        test_ratio: proporção para teste
        augment_train: aplicar augmentation no treino
        image_size: tamanho das imagens
    
    Returns:
        (train_dataset, val_dataset, test_dataset, loader)
    """
    loader = FVCDatasetLoader(
        datasets=datasets,
        random_state=random_state,
        image_size=image_size,
    )
    
    train_ds, val_ds, test_ds = loader.create_datasets(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        augment_train=augment_train,
    )
    
    return train_ds, val_ds, test_ds, loader


def load_datasets(
    datasets: List[str] = None,
    random_state: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    augment_train: bool = True,
    aggressive_augment: bool = False,
    image_size: Tuple[int, int] = (299, 299),
):
    """
    Função unificada para carregar múltiplos datasets (FVC e SD302).
    
    Args:
        datasets: lista de datasets (ex: ["FVC2000", "FVC2002", "FVC2004", "SD302"])
        random_state: seed para reprodutibilidade
        train_ratio: proporção para treino
        val_ratio: proporção para validação
        test_ratio: proporção para teste
        augment_train: aplicar augmentation no treino
        aggressive_augment: aplicar augmentation agressivo (para datasets pequenos)
        image_size: tamanho das imagens
    
    Returns:
        (train_dataset, val_dataset, test_dataset, loaders_dict)
    """
    if datasets is None:
        datasets = ["FVC2000", "FVC2002", "FVC2004", "SD302"]
    
    logger = logging.getLogger(__name__)
    
    all_train_paths = []
    all_train_labels = []
    all_val_paths = []
    all_val_labels = []
    all_test_paths = []
    all_test_labels = []
    
    current_label_offset = 0
    loaders = {}
    
    fvc_datasets = [d for d in datasets if d.startswith("FVC")]
    include_sd302 = "SD302" in datasets
    include_sfinge = "SFinge" in datasets or "FP_gen_0" in datasets
    
    if fvc_datasets:
        logger.info(f"Carregando bases FVC: {fvc_datasets}")
        fvc_loader = FVCDatasetLoader(
            datasets=fvc_datasets,
            random_state=random_state,
            image_size=image_size,
        )
        
        train_idx, val_idx, test_idx = fvc_loader.get_split_indices(
            train_ratio, val_ratio, test_ratio
        )
        
        all_train_paths.extend([fvc_loader.image_paths[i] for i in train_idx])
        all_train_labels.extend([fvc_loader.labels[i] + current_label_offset for i in train_idx])
        
        all_val_paths.extend([fvc_loader.image_paths[i] for i in val_idx])
        all_val_labels.extend([fvc_loader.labels[i] + current_label_offset for i in val_idx])
        
        all_test_paths.extend([fvc_loader.image_paths[i] for i in test_idx])
        all_test_labels.extend([fvc_loader.labels[i] + current_label_offset for i in test_idx])
        
        current_label_offset += len(fvc_loader.label_map)
        loaders["FVC"] = fvc_loader
    
    if include_sd302:
        logger.info("Carregando base SD302")
        sd302_loader = SD302DatasetLoader(
            random_state=random_state,
            image_size=image_size,
        )
        
        train_idx, val_idx, test_idx = sd302_loader.get_split_indices(
            train_ratio, val_ratio, test_ratio
        )
        
        all_train_paths.extend([sd302_loader.image_paths[i] for i in train_idx])
        all_train_labels.extend([sd302_loader.labels[i] + current_label_offset for i in train_idx])
        
        all_val_paths.extend([sd302_loader.image_paths[i] for i in val_idx])
        all_val_labels.extend([sd302_loader.labels[i] + current_label_offset for i in val_idx])
        
        all_test_paths.extend([sd302_loader.image_paths[i] for i in test_idx])
        all_test_labels.extend([sd302_loader.labels[i] + current_label_offset for i in test_idx])
        
        loaders["SD302"] = sd302_loader
    
    if include_sfinge:
        logger.info("Carregando base SFinge (FP_gen_0 + FP_gen_1)")
        sfinge_loader = SFingeDatasetLoader(
            random_state=random_state,
            image_size=image_size,
            use_both_gen=True,
        )
        
        # OPEN-SET igual ao paper: FP_gen_0 treino, FP_gen_1 val/test
        train_idx, val_idx, test_idx = sfinge_loader.get_split_indices(
            train_ratio, val_ratio, test_ratio, use_openset=True
        )
        
        all_train_paths.extend([sfinge_loader.image_paths[i] for i in train_idx])
        all_train_labels.extend([sfinge_loader.labels[i] + current_label_offset for i in train_idx])
        
        all_val_paths.extend([sfinge_loader.image_paths[i] for i in val_idx])
        all_val_labels.extend([sfinge_loader.labels[i] + current_label_offset for i in val_idx])
        
        all_test_paths.extend([sfinge_loader.image_paths[i] for i in test_idx])
        all_test_labels.extend([sfinge_loader.labels[i] + current_label_offset for i in test_idx])
        
        current_label_offset += len(sfinge_loader.label_map)
        loaders["SFinge"] = sfinge_loader
    
    # Remapeamento de labels para OPEN-SET (igual ao DeepPrint original)
    # TREINO: apenas classes de FP_gen_0 (6000 classes) 
    # VAL/TEST: classes de FP_gen_1 (2000 classes DISJUNTAS)
    # Modelo é treinado APENAS com classes de treino!
    
    # Remapear APENAS labels de treino (contíguo 0 a num_train_classes-1)
    unique_train_labels = sorted(set(all_train_labels))
    train_label_remap = {old: new for new, old in enumerate(unique_train_labels)}
    all_train_labels_remapped = [train_label_remap[l] for l in all_train_labels]
    
    # Val/Test: remapear separadamente (não importa muito pois não usam classificador)
    # Mas mantemos consistência: remapear para 0 a num_val_classes-1
    unique_val_labels = sorted(set(all_val_labels))
    val_label_remap = {old: new for new, old in enumerate(unique_val_labels)}
    all_val_labels_remapped = [val_label_remap[l] for l in all_val_labels]
    
    unique_test_labels = sorted(set(all_test_labels))
    test_label_remap = {old: new for new, old in enumerate(unique_test_labels)}
    all_test_labels_remapped = [test_label_remap[l] for l in all_test_labels]
    
    # Número de classes: APENAS treino (modelo é treinado com estas)
    num_classes_global = len(unique_train_labels)
    
    # Log claro sobre open-set split
    logger.info("=" * 80)
    logger.info("OPEN-SET SPLIT (classes disjuntas treino/val/test):")
    logger.info(f"  Treino:     {len(unique_train_labels)} classes únicas (FP_gen_0)")
    logger.info(f"  Validação:  {len(unique_val_labels)} classes únicas (FP_gen_1)")
    logger.info(f"  Teste:      {len(unique_test_labels)} classes únicas (FP_gen_1)")
    logger.info(f"  MODELO SERÁ TREINADO COM: {num_classes_global} classes (apenas treino)")
    logger.info("=" * 80)
    
    train_dataset = FingerprintDataset(
        image_paths=all_train_paths,
        labels=all_train_labels_remapped,
        image_size=image_size,
        augment=augment_train,
        augmentation_config={"aggressive": aggressive_augment} if aggressive_augment else None,
    )
    
    val_dataset = FingerprintDataset(
        image_paths=all_val_paths,
        labels=all_val_labels_remapped,
        image_size=image_size,
        augment=False,
    )
    
    test_dataset = FingerprintDataset(
        image_paths=all_test_paths,
        labels=all_test_labels_remapped,
        image_size=image_size,
        augment=False,
    )
    
    # Armazenar número de classes GLOBAL em todos os datasets
    # (todos compartilham o mesmo espaço de classes)
    num_classes_train = num_classes_global
    num_classes_val = num_classes_global
    num_classes_test = num_classes_global
    
    logger.info(
        f"Datasets combinados: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    logger.info(
        f"Classes únicas: train={num_classes_train}, val={num_classes_val}, test={num_classes_test}"
    )
    
    # Adicionar num_classes aos datasets
    train_dataset.num_classes = num_classes_train
    val_dataset.num_classes = num_classes_val
    test_dataset.num_classes = num_classes_test
    
    return train_dataset, val_dataset, test_dataset, loaders
