import torch
import torchvision
import torch.nn as nn

# https://programtalk.com/vs4/python/socom20/facebook-image-similarity-challenge-2021/ensemble_training_scripts/smp_test19/Facebook_model_v20.py/

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        return None

    def forward(self, inputs, targets, reduction='mean'):
        focal_loss = torchvision.ops.sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
        )
        
        return focal_loss


"""def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
):
    #if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        #_log_api_usage_once(sigmoid_focal_loss)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss"""