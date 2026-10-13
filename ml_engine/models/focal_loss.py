import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance without synthetic oversampling (SMOTE).
    Specifically tailored for transaction fraud detection on graphs.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Attributes:
        alpha (float or Tensor): Weight factor for the minority/illicit class (e.g., 0.80 - 0.85).
                                 For binary classification, class 1 gets alpha, class 0 gets (1 - alpha).
        gamma (float): Focusing parameter (typically 2.0) that smoothly down-weights easy examples
                       and focuses training on hard ambiguous transactions.
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss over optionally masked nodes.

        Args:
            logits: Predicted raw logits of shape [N, num_classes] (e.g. [N, 2]).
            targets: Ground-truth class labels of shape [N].
            mask: Optional boolean tensor of shape [N] indicating nodes to include in loss.
                  Nodes with targets == -1 (background/unlabeled) are always excluded.

        Returns:
            Computed scalar or tensor loss.
        """
        # Determine valid nodes: must satisfy mask (if provided) and target != -1
        valid_nodes = targets != -1
        if mask is not None:
            valid_nodes = valid_nodes & mask

        if not valid_nodes.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_nodes]
        valid_targets = targets[valid_nodes]

        # Log-softmax for numerical stability
        log_probs = F.log_softmax(valid_logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather probability of target class
        target_indices = valid_targets.unsqueeze(-1)
        log_pt = log_probs.gather(dim=-1, index=target_indices).squeeze(-1)
        pt = probs.gather(dim=-1, index=target_indices).squeeze(-1).clamp(min=1e-7, max=1.0)

        # Alpha weighting: alpha for class 1 (illicit), (1 - alpha) for class 0 (licit)
        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.where(
                valid_targets == 1,
                torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
                torch.tensor(1.0 - self.alpha, device=logits.device, dtype=logits.dtype),
            )
        elif isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(device=logits.device, dtype=logits.dtype)[valid_targets]
        else:
            alpha_t = torch.ones_like(pt)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - pt) ** self.gamma
        focal_loss = -alpha_t * modulating_factor * log_pt

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
