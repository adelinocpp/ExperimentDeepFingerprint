"""
Modelos base para DeepPrint e suas variantes.

Baseado na implementação original:
https://github.com/tim-rohwedder/fixed-length-fingerprint-extractors

O baseline (exp0) usa a arquitetura original do DeepPrint.
Os demais experimentos são derivações com modificações específicas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

# Importar módulos do InceptionV4 original
import InceptionV4
import torchvision

DEEPPRINT_INPUT_SIZE = 299

# =============================================================================
# Localization Network (STN - Spatial Transformer Network)
# =============================================================================

class LocalizationNetwork(nn.Module):
    """Rede de localização para alinhamento automático (STN)"""
    
    def __init__(self):
        super().__init__()
        self.input_size = (128, 128)
        self.resize = torchvision.transforms.Resize(
            size=self.input_size, antialias=True
        )
        
        self.localization = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(8 * 8 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Inicializar com transformação identidade
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resized_x = self.resize(x)
        xs = self.localization(resized_x)
        xs = xs.view(-1, 8 * 8 * 64)
        theta_x_y = self.fc_loc(xs)
        theta_x_y = theta_x_y.view(-1, 3)
        theta = theta_x_y[:, 0]  # Ângulo de rotação
        
        # Construir matriz de rotação e translação
        m11 = torch.cos(theta)
        m12 = -torch.sin(theta)
        m13 = theta_x_y[:, 1]  # offset x
        m21 = torch.sin(theta)
        m22 = torch.cos(theta)
        m23 = theta_x_y[:, 2]  # offset y
        
        mat = torch.stack([m11, m12, m13, m21, m22, m23], dim=1)
        mat = mat.view(-1, 2, 3)
        grid = F.affine_grid(mat, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

# =============================================================================
# Componentes originais do DeepPrint
# =============================================================================

class _InceptionV4_Stem(nn.Module):
    """Stem do InceptionV4 - IGUAL AO ORIGINAL"""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            InceptionV4.BasicConv2d(1, 32, kernel_size=3, stride=2),
            InceptionV4.BasicConv2d(32, 32, kernel_size=3, stride=1),
            InceptionV4.BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            InceptionV4.Mixed_3a(),
            InceptionV4.Mixed_4a(),
            InceptionV4.Mixed_5a(),
        )

    def forward(self, x):
        return self.features(x)

class _Branch_TextureEmbedding(nn.Module):
    """Branch de textura - IGUAL AO ORIGINAL"""
    
    def __init__(self, texture_embedding_dims: int):
        super().__init__()
        self._0_block = nn.Sequential(
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Reduction_A(),
        )

        self._1_block = nn.Sequential(
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Reduction_B(),
        )

        self._2_block = nn.Sequential(
            InceptionV4.Inception_C(),
            InceptionV4.Inception_C(),
            InceptionV4.Inception_C(),
        )

        self._3_avg_pool2d = nn.AvgPool2d(kernel_size=8)
        self._4_flatten = nn.Flatten()
        self._5_dropout = nn.Dropout(p=0.2)
        self._6_linear = nn.Linear(1536, texture_embedding_dims)

    def forward(self, x):
        x = self._0_block(x)
        x = self._1_block(x)
        x = self._2_block(x)
        x = self._3_avg_pool2d(x)
        x = self._4_flatten(x)
        x = self._5_dropout(x)
        x = self._6_linear(x)
        x = F.normalize(x.squeeze(-1).squeeze(-1) if x.dim() > 2 else x, dim=1)
        return x

class _Branch_MinutiaStem(nn.Module):
    """Stem de minutiae - IGUAL AO ORIGINAL"""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
        )

    def forward(self, x):
        return self.features(x)

class _Branch_MinutiaEmbedding(nn.Module):
    """Branch de embedding de minutiae - IGUAL AO ORIGINAL"""
    
    def __init__(self, minutia_embedding_dims: int):
        super().__init__()
        self._0_block = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(768, 896, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(896, 1024, kernel_size=3, stride=2, padding=1),
        )
        self._1_max_pool2d = nn.MaxPool2d(kernel_size=9, stride=1)
        self._2_flatten = nn.Flatten()
        self._3_dropout = nn.Dropout(p=0.2)
        self._4_linear = nn.Linear(1024, minutia_embedding_dims)

    def forward(self, x):
        x = self._0_block(x)
        x = self._1_max_pool2d(x)
        x = self._2_flatten(x)
        x = self._3_dropout(x)
        x = self._4_linear(x)
        x = F.normalize(x, dim=1)
        return x

class _Branch_MinutiaMap(nn.Module):
    """Branch de mapa de minutiae - IGUAL AO ORIGINAL"""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=7, stride=1),
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2),
            nn.Conv2d(32, 6, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.features(x)[:, :, :-1, :-1]

# =============================================================================
# Módulo de Atenção Espacial (para experimentos derivados)
# =============================================================================

class SpatialAttention(nn.Module):
    """Módulo de atenção espacial para focar em regiões importantes"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = torch.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = torch.sigmoid(self.conv(spatial_out))
        x = x * spatial_out
        
        return x


class DeepPrintBaseline(nn.Module):
    """
    DeepPrint Baseline - ARQUITETURA ORIGINAL COMPLETA
    
    DeepPrint_LocTexMinu: Localization Network + Texture Branch + Minutiae Branch
    - Texture embedding: 96 dims
    - Minutiae embedding: 96 dims
    - Total: 192 dims (concatenados)
    
    Esta é a arquitetura EXATA do paper original.
    """
    
    def __init__(self, texture_embedding_dims: int = 96, minutia_embedding_dims: int = 96, num_classes: int = None):
        super().__init__()
        
        # Localization network (STN)
        self.localization = LocalizationNetwork()
        
        # Stem compartilhado
        self.stem = _InceptionV4_Stem()
        
        # Texture branch
        self.texture_branch = _Branch_TextureEmbedding(texture_embedding_dims)
        self.texture_embedding_dims = texture_embedding_dims
        
        # Minutiae branch
        self.minutia_stem = _Branch_MinutiaStem()
        self.minutia_map = _Branch_MinutiaMap()
        self.minutia_embedding = _Branch_MinutiaEmbedding(minutia_embedding_dims)
        self.minutia_embedding_dims = minutia_embedding_dims
        
        # Embedding total = texture + minutiae
        self.total_embedding_dims = texture_embedding_dims + minutia_embedding_dims
        
        # Classificadores para treinamento
        self.num_classes = num_classes
        self.texture_classifier = None
        self.minutia_classifier = None
        
        if num_classes is not None:
            self.texture_classifier = nn.Sequential(
                nn.Linear(texture_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
            self.minutia_classifier = nn.Sequential(
                nn.Linear(minutia_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
        
    def set_num_classes(self, num_classes: int):
        """Configura os classificadores com o número de classes"""
        if self.num_classes != num_classes:
            self.num_classes = num_classes
            self.texture_classifier = nn.Sequential(
                nn.Linear(self.texture_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
            self.minutia_classifier = nn.Sequential(
                nn.Linear(self.minutia_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
            if next(self.parameters()).is_cuda:
                self.texture_classifier = self.texture_classifier.cuda()
                self.minutia_classifier = self.minutia_classifier.cuda()
        
    def forward(self, x):
        # 1. Alinhamento automático (STN)
        x_aligned = self.localization(x)
        
        # 2. Stem compartilhado
        features = self.stem(x_aligned)
        
        # 3. Texture branch
        texture_embedding = self.texture_branch(features)
        
        # 4. Minutiae branch
        minutia_features = self.minutia_stem(features)
        minutia_embedding = self.minutia_embedding(minutia_features)
        minutia_map = self.minutia_map(minutia_features)
        
        # 5. Concatenar embeddings (texture + minutiae = 192 dims)
        combined_embedding = torch.cat([texture_embedding, minutia_embedding], dim=1)
        
        result = {
            "embedding": combined_embedding,
            "texture_embedding": texture_embedding,
            "minutia_embedding": minutia_embedding,
            "minutia_map": minutia_map,
        }
        
        # Logits para treinamento
        if self.texture_classifier is not None:
            result["texture_logits"] = self.texture_classifier(texture_embedding)
            result["minutia_logits"] = self.minutia_classifier(minutia_embedding)
        
        return result


class DeepPrintEnhancedRepresentation(nn.Module):
    """
    DeepPrint com Representação Aumentada (1024 dimensões)
    
    Derivação do baseline com embedding de maior dimensão e
    camada de refinamento adicional.
    """
    
    def __init__(self, texture_embedding_dims: int = 1024, num_classes: int = None):
        super().__init__()
        self.stem = _InceptionV4_Stem()
        self.texture_branch = _Branch_TextureEmbedding(texture_embedding_dims)
        self.texture_embedding_dims = texture_embedding_dims
        
        # Camada adicional de refinamento (diferencial deste experimento)
        self.refinement = nn.Sequential(
            nn.Linear(texture_embedding_dims, texture_embedding_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(texture_embedding_dims, texture_embedding_dims),
        )
        
        self.num_classes = num_classes
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(texture_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
    
    def set_num_classes(self, num_classes: int):
        if self.num_classes != num_classes:
            self.num_classes = num_classes
            self.classifier = nn.Sequential(
                nn.Linear(self.texture_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
            if next(self.parameters()).is_cuda:
                self.classifier = self.classifier.cuda()
        
    def forward(self, x):
        features = self.stem(x)
        texture_embedding = self.texture_branch(features)
        
        # Refinamento da representação (diferencial)
        texture_embedding = self.refinement(texture_embedding)
        texture_embedding = F.normalize(texture_embedding, p=2, dim=1)
        
        result = {"embedding": texture_embedding}
        
        if self.classifier is not None:
            result["logits"] = self.classifier(texture_embedding)
        
        return result


class DeepPrintSpatialAttention(nn.Module):
    """
    DeepPrint com Atenção Espacial
    
    Derivação do baseline com módulo de atenção espacial
    após o branch de textura para focar em regiões importantes.
    """
    
    def __init__(self, texture_embedding_dims: int = 512, num_classes: int = None):
        super().__init__()
        self.stem = _InceptionV4_Stem()
        self.texture_branch = _Branch_TextureEmbedding(texture_embedding_dims)
        self.texture_embedding_dims = texture_embedding_dims
        
        # Atenção espacial aplicada nas features (diferencial)
        self.spatial_attention = SpatialAttention(384)  # 384 = output do stem
        
        self.num_classes = num_classes
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(texture_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
    
    def set_num_classes(self, num_classes: int):
        if self.num_classes != num_classes:
            self.num_classes = num_classes
            self.classifier = nn.Sequential(
                nn.Linear(self.texture_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
            if next(self.parameters()).is_cuda:
                self.classifier = self.classifier.cuda()
        
    def forward(self, x):
        features = self.stem(x)
        
        # Aplicar atenção espacial nas features (diferencial)
        features = self.spatial_attention(features)
        
        texture_embedding = self.texture_branch(features)
        
        result = {"embedding": texture_embedding}
        
        if self.classifier is not None:
            result["logits"] = self.classifier(texture_embedding)
        
        return result


class DeepPrintWithReranking(nn.Module):
    """
    DeepPrint com Re-ranking
    
    Derivação do baseline com módulo de re-ranking para
    refinar scores de comparação entre pares.
    """
    
    def __init__(self, texture_embedding_dims: int = 512, num_classes: int = None):
        super().__init__()
        self.stem = _InceptionV4_Stem()
        self.texture_branch = _Branch_TextureEmbedding(texture_embedding_dims)
        self.texture_embedding_dims = texture_embedding_dims
        
        # Módulo de re-ranking (diferencial)
        self.reranking_network = nn.Sequential(
            nn.Linear(texture_embedding_dims * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        self.num_classes = num_classes
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(texture_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
    
    def set_num_classes(self, num_classes: int):
        if self.num_classes != num_classes:
            self.num_classes = num_classes
            self.classifier = nn.Sequential(
                nn.Linear(self.texture_embedding_dims, num_classes),
                nn.Dropout(p=0.2)
            )
            if next(self.parameters()).is_cuda:
                self.classifier = self.classifier.cuda()
        
    def forward(self, x):
        features = self.stem(x)
        texture_embedding = self.texture_branch(features)
        
        result = {"embedding": texture_embedding}
        
        if self.classifier is not None:
            result["logits"] = self.classifier(texture_embedding)
        
        return result
    
    def compute_reranking_score(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """Calcula score de re-ranking entre dois embeddings"""
        combined = torch.cat([embedding1, embedding2], dim=1)
        score = self.reranking_network(combined)
        return score


def create_model(model_type: str, texture_embedding_dims: int = 512) -> nn.Module:
    """Factory function para criar modelos"""
    
    models = {
        "baseline": DeepPrintBaseline,
        "enhanced_representation": DeepPrintEnhancedRepresentation,
        "spatial_attention": DeepPrintSpatialAttention,
        "reranking": DeepPrintWithReranking,
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Available: {list(models.keys())}")
    
    return models[model_type](texture_embedding_dims)
