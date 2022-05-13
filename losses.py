from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as func


def ohe(pred, target):
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.scatter_(1, target.data.unsqueeze(1), 1)
    return true_dist



def bce(pred, target):

    return dict(
            overall=func.binary_cross_entropy_with_logits(pred, ohe(pred, target))
        )


def multilabel_soft_margin_loss(pred, target):

    return dict(
            overall=func.multilabel_soft_margin_loss(pred, ohe(pred, target))
        )



def label_smooth(pred, target, claasses=1394, smoothing=0.1):
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(smoothing / (claasses - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - smoothing)
        return dict(
            overall=torch.mean(torch.sum(-true_dist * pred, dim=-1))
        )


def ce_loss(input, target, reduction='mean'):
        return  dict(overall=func.cross_entropy(input, target, reduction=reduction))


def focal_loss(input, target, alpha=4.0, gamma=2.0, reduction='mean'):
        ce_loss = func.cross_entropy(input, target, reduction=reduction, weight=torch.tensor(alpha, device=input.device))
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
        return dict(overall=focal_loss)


def distill_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    labels: Tensor,
    temperature: float = 1.0,
    alpha: float = 0.5
) -> Tuple[Tensor, Tensor, Tensor]:
    """Default distillation loss (KLD + CE).
    Args:
        teacher_logits: Logits from teacher model.
        student_logits: Logits from student model.
        labels: Targets.
        temperature: Temperature for softening distributions.
            Larger temperature -> softer distribution.
        alpha: Weight to KLD loss and 1 - alpha to CrossEntropy loss
    Returns:
        Tuple of KLD loss, CE loss and combined weighted loss tensors.
    """
    loss_kld = kld_loss(teacher_logits, student_logits, temperature)
    loss_ce = func.cross_entropy(student_logits, labels)
    overall = loss_kld * alpha + loss_ce * (1. - alpha)
    return dict(loss_kld=loss_kld, loss_ce=loss_ce, overall=overall)


def kld_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = None,
) -> torch.Tensor:
    """Kullback–Leibler divergence loss.
    Args:
        teacher_logits: Logits from teacher model.
        student_logits: Logits from student model.
        temperature: Temperature for softening distributions.
            Larger temperature -> softer distribution.
    Returns:
        Tensor of loss value.
    """
    soft_log_probs = func.log_softmax(student_logits / temperature, dim=-1)
    soft_targets = func.softmax(teacher_logits / temperature, dim=-1)
    distillation_loss = func.kl_div(
        input=soft_log_probs,
        target=soft_targets,
        reduction='batchmean'
    )
    distillation_loss_scaled = distillation_loss * temperature ** 2
    return distillation_loss_scaled


def cosine_loss(x1: Tensor, x2: Tensor) -> Tensor:
    """Cosine distance loss calculated on last dimension.
    Args:
        x1: First input.
        x2: Second input (of size matching x1).
    Returns:
        Averaged cosine distance between x1 and x2.
    """
    distance = 1 - func.cosine_similarity(x1, x2, dim=-1)
    mean_distance = distance.mean()
    return mean_distance