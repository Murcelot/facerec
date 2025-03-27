from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ArcFaceLoss(nn.Module):
    """
    ArcFace loss from `paper <https://arxiv.org/abs/1801.07698>`_ with possibility to use label smoothing.
    It contains projection size of ``num_features x num_classes`` inside itself. Please make sure that class labels
    started with 0 and ended as ``num_classes`` - 1.
    """

    criterion_name = "arcface"  # for better logging

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        m: float = 0.5,
        s: float = 64,
        smoothing_epsilon: float = 0,
        label2category: Optional[Dict[Any, Any]] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            in_features: Input feature size
            num_classes: Number of classes in train set
            m: Margin parameter for ArcFace loss. Usually you should use 0.3-0.5 values for it
            s: Scaling parameter for ArcFace loss. Usually you should use 30-64 values for it
            smoothing_epsilon: Label smoothing effect strength
            label2category: Optional, mapping from label to its category. If provided, label smoothing will redistribute
                 ``smoothing_epsilon`` only inside the category corresponding to the sample's ground truth label
            reduction: CrossEntropyLoss reduction
        """
        super(ArcFaceLoss, self).__init__()

        assert (
            smoothing_epsilon is None or 0 <= smoothing_epsilon < 1
        ), f"Choose another smoothing_epsilon parametrization, got {smoothing_epsilon}"

        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.num_classes = num_classes
        if label2category is not None:
            mapper = {l: i for i, l in enumerate(sorted(list(set(label2category.values()))))}
            label2category = {k: mapper[v] for k, v in label2category.items()}
            self.label2category = torch.arange(num_classes).apply_(label2category.get)
        else:
            self.label2category = None
        self.smoothing_epsilon = smoothing_epsilon
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.rescale = s
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = -self.cos_m
        self.mm = self.sin_m * m
        self._last_logs: Dict[str, float] = {}


    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, p=2), F.normalize(self.weight, p=2))

    def smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        if self.label2category is not None:
            self.label2category = self.label2category.to(self.weight.device)
        return label_smoothing(y, self.num_classes, self.smoothing_epsilon, self.label2category)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.all(y < self.num_classes), "You should provide labels between 0 and num_classes - 1."

        cos = self.fc(x)

        self._log_accuracy_on_batch(cos, y)

        sin = torch.sqrt(1.0 - torch.pow(cos, 2))

        cos_w_margin = cos * self.cos_m - sin * self.sin_m
        cos_w_margin = torch.where(cos > self.th, cos_w_margin, cos - self.mm)

        if y.dim() > 1:
            ohe = F.one_hot(torch.argmax(y, 1), self.num_classes)
        else:
            ohe = F.one_hot(y, self.num_classes)
        logit = torch.where(ohe.bool(), cos_w_margin, cos) * self.rescale

        if self.smoothing_epsilon:
            y = self.smooth_labels(y)

        return self.criterion(logit, y)

    @torch.no_grad()
    def _log_accuracy_on_batch(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1:
            self._last_logs["accuracy"] = torch.mean((torch.argmax(y, 1) == torch.argmax(logits, 1)).to(torch.float32))
        else:
            self._last_logs["accuracy"] = torch.mean((y == torch.argmax(logits, 1)).to(torch.float32))

    @property
    def last_logs(self) -> Dict[str, Any]:
        """
        Returns:
            Dictionary containing useful statistic calculated for the last batch.
        """
        return self._last_logs
    

@torch.no_grad()
def label_smoothing(
    y: torch.Tensor,
    num_classes: int,
    epsilon: float = 0.2,
    categories: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function is doing `label smoothing <https://arxiv.org/pdf/1512.00567v3.pdf>`_.
    You can also use modified version, where the label is smoothed only for the category corresponding to sample's
    ground truth label. To use this, you should provide the ``categories`` argument: vector, for which i-th entry
    is a corresponding category for label ``i``.

    Args:
        y: Ground truth labels with the size of batch_size where each element is from 0 (inclusive) to
            num_classes (exclusive).
        num_classes: Number of classes in total
        epsilon: Power of smoothing. The biggest value in OHE-vector will be
            ``1 - epsilon + 1 / num_classes`` after the transformation
        categories: Vector for which i-th entry is a corresponding category for label ``i``. Optional, used for
            category-based label smoothing. In that case the biggest value in OHE-vector will be
            ``1 - epsilon + 1 / num_classes_of_the_same_category``, labels outside of the category will not change
    """
    assert epsilon < 1, "`epsilon` must be less than 1."
    if y.dim() == 1:
        out = F.one_hot(y, num_classes).float()
        out *= 1 - epsilon
        out += epsilon / num_classes
    else:
        out = y * (1 - epsilon)
        out += epsilon / num_classes
    return out