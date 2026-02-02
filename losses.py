"""
Loss functions for fingerprint recognition.

Implements:
- CenterLoss: Original DeepPrint center loss
- ArcFaceLoss: Additive Angular Margin Loss (ArcFace)

References:
- Center Loss: "A Discriminative Feature Learning Approach for Deep Face Recognition" (Wen et al., 2016)
- ArcFace: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (Deng et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class CenterLoss(nn.Module):
    """
    Center Loss para aprendizado discriminativo de features.

    Formula: L_center = (1/(2*N)) * Σ ||x_i - c_yi||^2
    onde c_yi é o centro da classe yi.

    Args:
        num_classes: Número de classes
        feat_dim: Dimensão dos embeddings
        alpha: Learning rate para atualização dos centros (não usado nesta implementação)
        device: Device (cuda/cpu)
    """

    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.01, device: torch.device = torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha  # Para compatibilidade com código existente
        self.device = device

        # Centros de classe (learnable parameters)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feat_dim)
            labels: (batch_size,)

        Returns:
            Center loss value
        """
        batch_size = features.size(0)

        # Expandir centros e features para cálculo de distância
        centers_batch = self.centers.index_select(0, labels.long())

        # L2 distance: ||x_i - c_yi||^2
        loss = (features - centers_batch).pow(2).sum() / (2.0 * batch_size)

        return loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.

    Formula: L = -1/N * Σ log( e^(s*cos(θ_yi + m)) / (e^(s*cos(θ_yi + m)) + Σ_{j≠yi} e^(s*cos(θ_j))) )

    onde:
    - θ_yi é o ângulo entre feature x_i e weight W_yi
    - m é o angular margin (margem angular aditiva)
    - s é o feature scale

    ArcFace normaliza tanto features quanto weights para a hypersphere unitária,
    então a similaridade se torna puramente angular: cos(θ) = W^T * x

    Args:
        num_classes: Número de classes
        feat_dim: Dimensão dos embeddings
        s: Feature scale (paper usa 64)
        m: Angular margin em radianos (paper usa 0.5 ≈ 28.6°)
        easy_margin: Se True, usa cos(θ+m) apenas quando θ+m < π
        device: Device (cuda/cpu)

    References:
        ArcFace paper (CVPR 2019): https://arxiv.org/abs/1801.07698
        - Testado em LFW, CFP-FP, AgeDB-30, MegaFace, IJB-C
        - Escala até 85K classes (MS1MV2 dataset)
        - Superior a Center Loss, SphereFace, CosFace
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
        device: torch.device = torch.device('cpu')
    ):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.s = s  # Feature scale
        self.m = m  # Angular margin
        self.easy_margin = easy_margin
        self.device = device

        # Weight matrix (learnable, será normalizado)
        # Cada coluna é o weight vector de uma classe
        self.weight = Parameter(torch.FloatTensor(feat_dim, num_classes).to(device))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute cos(m) e sin(m) para eficiência
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Para easy_margin
        self.th = math.cos(math.pi - m)  # threshold = cos(π - m)
        self.mm = math.sin(math.pi - m) * m  # sin(π - m) * m

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feat_dim) - embeddings (serão normalizados internamente)
            labels: (batch_size,) - ground truth labels

        Returns:
            ArcFace loss value
        """
        # 1. Normalizar features: x = x / ||x||
        features_norm = F.normalize(features, p=2, dim=1)

        # 2. Normalizar weights: W = W / ||W||
        weight_norm = F.normalize(self.weight, p=2, dim=0)

        # 3. Calcular cos(θ) = W^T * x (produto escalar normalizado)
        # cosine shape: (batch_size, num_classes)
        cosine = F.linear(features_norm, weight_norm.t())
        cosine = cosine.clamp(-1, 1)  # Numérico: garantir [-1, 1]

        # 4. Calcular θ = arccos(W^T * x)
        theta = torch.acos(cosine)

        # 5. Calcular cos(θ + m) usando identidade trigonométrica:
        # cos(θ + m) = cos(θ)*cos(m) - sin(θ)*sin(m)
        # sin(θ) = sqrt(1 - cos²(θ))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        if self.easy_margin:
            # Easy margin: usar cos(θ + m) apenas se θ + m < π
            # Caso contrário, usar cos(θ) - m
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Hard margin: usar cos(θ + m) se θ < π - m, senão cos(θ) - mm
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 6. One-hot encoding das labels
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 7. Adicionar margin apenas para a classe correta
        # output = one_hot * phi + (1 - one_hot) * cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 8. Aplicar feature scale e calcular cross entropy
        output *= self.s

        # 9. Cross entropy loss
        loss = F.cross_entropy(output, labels.long())

        return loss


class CombinedLoss(nn.Module):
    """
    Loss combinada: Softmax + (Center Loss OU ArcFace).

    Para DeepPrint original:
        L_total = λ1 * L_softmax + λ2 * L_center + λ3 * L_minutia_map

    Com ArcFace (ArcFace já inclui softmax interno):
        L_total = L_arcface + λ3 * L_minutia_map

    Args:
        loss_type: "center" ou "arcface"
        num_classes: Número de classes
        feat_dim: Dimensão dos embeddings
        center_loss_weight: Peso do center loss (λ2)
        arcface_margin: Margin do ArcFace (m)
        arcface_scale: Scale do ArcFace (s)
        device: Device
    """

    def __init__(
        self,
        loss_type: str,
        num_classes: int,
        feat_dim: int,
        center_loss_weight: float = 0.00125,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        device: torch.device = torch.device('cpu')
    ):
        super(CombinedLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        if self.loss_type == "center":
            self.center_loss = CenterLoss(num_classes, feat_dim, device)
            self.center_loss_weight = center_loss_weight
            self.softmax_loss = nn.CrossEntropyLoss()

        elif self.loss_type == "arcface":
            self.arcface_loss = ArcFaceLoss(
                num_classes, feat_dim,
                s=arcface_scale,
                m=arcface_margin,
                device=device
            )

        else:
            raise ValueError(f"loss_type deve ser 'center' ou 'arcface', recebeu: {loss_type}")

    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Args:
            features: (batch_size, feat_dim) - embeddings normalizados
            logits: (batch_size, num_classes) - logits do classificador (usado apenas para center loss)
            labels: (batch_size,) - ground truth labels

        Returns:
            dict com:
                - total_loss: loss total
                - softmax_loss (center) ou arcface_loss
                - center_loss (apenas center)
        """
        if self.loss_type == "center":
            # DeepPrint original: softmax + center loss
            softmax_loss = self.softmax_loss(logits, labels.long())
            center_loss = self.center_loss(features, labels)
            total_loss = softmax_loss + self.center_loss_weight * center_loss

            return {
                'total_loss': total_loss,
                'softmax_loss': softmax_loss,
                'center_loss': center_loss,
            }

        else:  # arcface
            # ArcFace: já inclui softmax interno
            arcface_loss = self.arcface_loss(features, labels)

            return {
                'total_loss': arcface_loss,
                'arcface_loss': arcface_loss,
            }


def get_loss_function(
    loss_type: str,
    num_classes: int,
    feat_dim: int,
    center_loss_weight: float = 0.00125,
    arcface_margin: float = 0.5,
    arcface_scale: float = 64.0,
    device: torch.device = torch.device('cpu')
) -> CombinedLoss:
    """
    Factory function para criar loss function.

    Args:
        loss_type: "center" ou "arcface"
        num_classes: Número de classes
        feat_dim: Dimensão dos embeddings (192 para DeepPrint)
        center_loss_weight: Peso do center loss (paper: 0.00125)
        arcface_margin: Angular margin do ArcFace (paper: 0.5 ≈ 28.6°)
        arcface_scale: Feature scale do ArcFace (paper: 64)
        device: Device

    Returns:
        CombinedLoss configurada
    """
    return CombinedLoss(
        loss_type=loss_type,
        num_classes=num_classes,
        feat_dim=feat_dim,
        center_loss_weight=center_loss_weight,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
        device=device
    )
