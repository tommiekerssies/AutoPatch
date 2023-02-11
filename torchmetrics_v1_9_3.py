from typing import List, Optional, Sequence, Tuple, Union, Any
import torch
from torch import Tensor, tensor
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
import warnings
from torch.nn import functional as F


def _precision_recall_curve_update(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
) -> Tuple[Tensor, Tensor, int, Optional[int]]:
    """Updates and returns variables required to compute the precision-recall pairs for different thresholds.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range [0,num_classes-1]
    """

    if len(preds.shape) == len(target.shape):
        if pos_label is None:
            pos_label = 1
        if num_classes is not None and num_classes != 1:
            # multilabel problem
            if num_classes != preds.shape[1]:
                raise ValueError(
                    f"Argument `num_classes` was set to {num_classes} in"
                    f" metric `precision_recall_curve` but detected {preds.shape[1]}"
                    " number of classes from predictions"
                )
            preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1)
            target = target.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1)
        else:
            # binary problem
            preds = preds.flatten()
            target = target.flatten()
            num_classes = 1

    # multi class problem
    elif len(preds.shape) == len(target.shape) + 1:
        if pos_label is not None:
            rank_zero_warn(
                "Argument `pos_label` should be `None` when running"
                f" multiclass precision recall curve. Got {pos_label}"
            )
        if num_classes != preds.shape[1]:
            raise ValueError(
                f"Argument `num_classes` was set to {num_classes} in"
                f" metric `precision_recall_curve` but detected {preds.shape[1]}"
                " number of classes from predictions"
            )
        preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1)
        target = target.flatten()

    else:
        raise ValueError(
            "preds and target must have same number of dimensions, or one additional dimension for preds"
        )

    return preds, target, num_classes, pos_label


def _binary_clf_curve(
    preds: Tensor,
    target: Tensor,
    sample_weights: Optional[Sequence] = None,
    pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """adapted from https://github.com/scikit-learn/scikit- learn/blob/master/sklearn/metrics/_ranking.py."""
    if sample_weights is not None and not isinstance(sample_weights, Tensor):
        sample_weights = tensor(sample_weights, device=preds.device, dtype=torch.float)

    # remove class dimension if necessary
    if preds.ndim > target.ndim:
        preds = preds[:, 0]
    desc_score_indices = torch.argsort(preds, descending=True)

    preds = preds[desc_score_indices]
    target = target[desc_score_indices]

    weight = 1.0 if sample_weights is None else sample_weights[desc_score_indices]
    # pred typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
    threshold_idxs = F.pad(distinct_value_indices, [0, 1], value=target.size(0) - 1)
    target = (target == pos_label).to(torch.long)
    tps = torch.cumsum(target * weight, dim=0)[threshold_idxs]

    if sample_weights is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = torch.cumsum((1 - target) * weight, dim=0)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, preds[threshold_idxs]


def _precision_recall_curve_compute_single_class(
    preds: Tensor,
    target: Tensor,
    pos_label: int,
    sample_weights: Optional[Sequence] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes precision-recall pairs for single class inputs.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        pos_label: integer determining the positive class.
        sample_weights: sample weights for each data point
    """

    fps, tps, thresholds = _binary_clf_curve(
        preds=preds, target=target, sample_weights=sample_weights, pos_label=pos_label
    )
    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained and reverse the outputs so recall is decreasing
    last_ind = torch.where(tps == tps[-1])[0][0]
    sl = slice(0, last_ind.item() + 1)

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides that are not yet supported in pytorch
    precision = torch.cat(
        [
            reversed(precision[sl]),
            torch.ones(1, dtype=precision.dtype, device=precision.device),
        ]
    )

    recall = torch.cat(
        [reversed(recall[sl]), torch.zeros(1, dtype=recall.dtype, device=recall.device)]
    )

    thresholds = reversed(thresholds[sl]).detach().clone()  # type: ignore

    return precision, recall, thresholds


def precision_recall_curve(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    sample_weights: Optional[Sequence] = None,
) -> Union[
    Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]
]:
    """Computes precision-recall pairs for different thresholds.
    Args:
        preds: predictions from model (probabilities)
        target: ground truth labels
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems.
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        sample_weights: sample weights for each data point
    Returns:
        3-element tuple containing
        precision:
            tensor where element ``i`` is the precision of predictions with
            ``score >= thresholds[i]`` and the last element is 1.
            If multiclass, this is a list of such tensors, one for each class.
        recall:
            tensor where element ``i`` is the recall of predictions with
            ``score >= thresholds[i]`` and the last element is 0.
            If multiclass, this is a list of such tensors, one for each class.
        thresholds:
            Thresholds used for computing precision/recall scores
    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same number of dimensions,
            or one additional dimension for ``preds``.
        ValueError:
            If the number of classes deduced from ``preds`` is not the same as the ``num_classes`` provided.
    Example (binary case):
        >>> from torchmetrics.functional import precision_recall_curve
        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> precision, recall, thresholds = precision_recall_curve(pred, target, pos_label=1)
        >>> precision
        tensor([0.6667, 0.5000, 0.0000, 1.0000])
        >>> recall
        tensor([1.0000, 0.5000, 0.0000, 0.0000])
        >>> thresholds
        tensor([1, 2, 3])
    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> precision, recall, thresholds = precision_recall_curve(pred, target, num_classes=5)
        >>> precision
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.7500]), tensor([0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500])]
    """
    preds, target, num_classes, pos_label = _precision_recall_curve_update(
        preds, target, num_classes, pos_label
    )
    return _precision_recall_curve_compute(
        preds, target, num_classes, pos_label, sample_weights
    )


def _precision_recall_curve_compute_multi_class(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    sample_weights: Optional[Sequence] = None,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Computes precision-recall pairs for multi class inputs.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems.
        sample_weights: sample weights for each data point
    """

    # Recursively call per class
    precision, recall, thresholds = [], [], []
    for cls in range(num_classes):
        preds_cls = preds[:, cls]

        prc_args = dict(
            preds=preds_cls,
            target=target,
            num_classes=1,
            pos_label=cls,
            sample_weights=sample_weights,
        )
        if target.ndim > 1:
            prc_args |= dict(
                target=target[:, cls],
                pos_label=1,
            )
        res = precision_recall_curve(**prc_args)
        precision.append(res[0])
        recall.append(res[1])
        thresholds.append(res[2])

    return precision, recall, thresholds


def _precision_recall_curve_compute(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    pos_label: Optional[int] = None,
    sample_weights: Optional[Sequence] = None,
) -> Union[
    Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]
]:
    """Computes precision-recall pairs based on the number of classes.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range ``[0,num_classes-1]``
        sample_weights: sample weights for each data point

    Example:
        >>> # binary case
        >>> preds = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> pos_label = 1
        >>> preds, target, num_classes, pos_label = _precision_recall_curve_update(preds, target, pos_label=pos_label)
        >>> precision, recall, thresholds = _precision_recall_curve_compute(preds, target, num_classes, pos_label)
        >>> precision
        tensor([0.6667, 0.5000, 0.0000, 1.0000])
        >>> recall
        tensor([1.0000, 0.5000, 0.0000, 0.0000])
        >>> thresholds
        tensor([1, 2, 3])

        >>> # multiclass case
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> num_classes = 5
        >>> preds, target, num_classes, pos_label = _precision_recall_curve_update(preds, target, num_classes)
        >>> precision, recall, thresholds = _precision_recall_curve_compute(preds, target, num_classes)
        >>> precision
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.7500]), tensor([0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500])]
    """

    with torch.no_grad():
        if num_classes == 1:
            if pos_label is None:
                pos_label = 1
            return _precision_recall_curve_compute_single_class(
                preds, target, pos_label, sample_weights
            )
        return _precision_recall_curve_compute_multi_class(
            preds, target, num_classes, sample_weights
        )


def _average_precision_update(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Tuple[Tensor, Tensor, int, Optional[int]]:
    """Format the predictions and target based on the ``num_classes``, ``pos_label`` and ``average`` parameter.
    Args:
        preds: predictions from model (logits or probabilities)
        target: ground truth values
        num_classes: integer with number of classes.
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average: reduction method for multi-class or multi-label problems
    """
    preds, target, num_classes, pos_label = _precision_recall_curve_update(
        preds, target, num_classes, pos_label
    )
    if average == "micro" and preds.ndim != target.ndim:
        raise ValueError("Cannot use `micro` average with multi-class input")

    return preds, target, num_classes, pos_label


def _bincount(x: Tensor, minlength: Optional[int] = None) -> Tensor:
    """``torch.bincount`` currently does not support deterministic mode on GPU.
    This implementation fallback to a for-loop counting occurrences in that case.
    Args:
        x: tensor to count
        minlength: minimum length to count
    Returns:
        Number of occurrences for each unique element in x
    """
    if x.is_cuda and torch.are_deterministic_algorithms_enabled():
        if minlength is None:
            minlength = len(torch.unique(x))
        output = torch.zeros(minlength, device=x.device, dtype=torch.long)
        for i in range(minlength):
            output[i] = (x == i).sum()
        return output
    else:
        return torch.bincount(x, minlength=minlength)


def _average_precision_compute(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    sample_weights: Optional[Sequence] = None,
) -> Union[List[Tensor], Tensor]:
    """Computes the average precision score.
    Args:
        preds: predictions from model (logits or probabilities)
        target: ground truth values
        num_classes: integer with number of classes.
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems his argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average: reduction method for multi-class or multi-label problems
        sample_weights: sample weights for each data point
    Example:
        >>> # binary case
        >>> preds = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> pos_label = 1
        >>> preds, target, num_classes, pos_label = _average_precision_update(preds, target, pos_label=pos_label)
        >>> _average_precision_compute(preds, target, num_classes, pos_label)
        tensor(1.)
        >>> # multiclass case
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> num_classes = 5
        >>> preds, target, num_classes, pos_label = _average_precision_update(preds, target, num_classes)
        >>> _average_precision_compute(preds, target, num_classes, average=None)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """

    # todo: `sample_weights` is unused
    if average == "micro" and preds.ndim == target.ndim:
        preds = preds.flatten()
        target = target.flatten()
        num_classes = 1

    precision, recall, _ = _precision_recall_curve_compute(
        preds, target, num_classes, pos_label
    )
    if average == "weighted":
        if preds.ndim == target.ndim and target.ndim > 1:
            weights = target.sum(dim=0).float()
        else:
            weights = _bincount(target, minlength=num_classes).float()
        weights = weights / torch.sum(weights)
    else:
        weights = None
    return _average_precision_compute_with_precision_recall(
        precision, recall, num_classes, average, weights
    )


def _average_precision_compute_with_precision_recall(
    precision: Tensor,
    recall: Tensor,
    num_classes: int,
    average: Optional[str] = "macro",
    weights: Optional[Tensor] = None,
) -> Union[List[Tensor], Tensor]:
    """Computes the average precision score from precision and recall.
    Args:
        precision: precision values
        recall: recall values
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        average: reduction method for multi-class or multi-label problems
        weights: weights to use when average='weighted'
    Example:
        >>> # binary case
        >>> preds = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> pos_label = 1
        >>> preds, target, num_classes, pos_label = _average_precision_update(preds, target, pos_label=pos_label)
        >>> precision, recall, _ = _precision_recall_curve_compute(preds, target, num_classes, pos_label)
        >>> _average_precision_compute_with_precision_recall(precision, recall, num_classes, average=None)
        tensor(1.)
        >>> # multiclass case
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> num_classes = 5
        >>> preds, target, num_classes, pos_label = _average_precision_update(preds, target, num_classes)
        >>> precision, recall, _ = _precision_recall_curve_compute(preds, target, num_classes)
        >>> _average_precision_compute_with_precision_recall(precision, recall, num_classes, average=None)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """

    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    if num_classes == 1:
        return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])

    res = []
    for p, r in zip(precision, recall):
        res.append(-torch.sum((r[1:] - r[:-1]) * p[:-1]))

    # Reduce
    if average in ("macro", "weighted"):
        res = torch.stack(res)
        if torch.isnan(res).any():
            warnings.warn(
                "Average precision score for one or more classes was `nan`. Ignoring these classes in average",
                UserWarning,
            )
        if average == "macro":
            return res[~torch.isnan(res)].mean()
        weights = torch.ones_like(res) if weights is None else weights
        return (res * weights)[~torch.isnan(res)].sum()
    if average is None or average == "none":
        return res
    allowed_average = ("micro", "macro", "weighted", "none", None)
    raise ValueError(
        f"Expected argument `average` to be one of {allowed_average}"
        f" but got {average}"
    )


class AveragePrecision(Metric):
    """Computes the average precision score, which summarises the precision recall curve into one number. Works for
    both binary and multiclass problems. In the case of multiclass, the values will be calculated based on a one-
    vs-the-rest approach.
    Forward accepts
    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.
    - ``target`` (long tensor): ``(N, ...)`` with integer labels
    Args:
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average:
            defines the reduction that is applied in the case of multiclass and multilabel input.
            Should be one of the following:
            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes. Cannot be
              used with multiclass input.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support.
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Example (binary case):
        >>> from torchmetrics import AveragePrecision
        >>> pred = torch.tensor([0, 0.1, 0.8, 0.4])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> average_precision = AveragePrecision(pos_label=1)
        >>> average_precision(pred, target)
        tensor(1.)
    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> average_precision = AveragePrecision(num_classes=5, average=None)
        >>> average_precision(pred, target)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[str] = "macro",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.pos_label = pos_label
        allowed_average = ("micro", "macro", "weighted", "none", None)
        if average not in allowed_average:
            raise ValueError(
                f"Expected argument `average` to be one of {allowed_average}"
                f" but got {average}"
            )
        self.average = average

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AveragePrecision` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, num_classes, pos_label = _average_precision_update(
            preds, target, self.num_classes, self.pos_label, self.average
        )
        self.preds.append(preds)
        self.target.append(target)
        self.num_classes = num_classes
        self.pos_label = pos_label

    def compute(self) -> Union[Tensor, List[Tensor]]:
        """Compute the average precision score.
        Returns:
            tensor with average precision. If multiclass return list of such tensors, one for each class
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        if not self.num_classes:
            raise ValueError(
                f"`num_classes` bas to be positive number, but got {self.num_classes}"
            )
        return _average_precision_compute(
            preds, target, self.num_classes, self.pos_label, self.average
        )


class AveragePrecision(Metric):
    """Computes the average precision score, which summarises the precision recall curve into one number. Works for
    both binary and multiclass problems. In the case of multiclass, the values will be calculated based on a one-
    vs-the-rest approach.
    Forward accepts
    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.
    - ``target`` (long tensor): ``(N, ...)`` with integer labels
    Args:
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average:
            defines the reduction that is applied in the case of multiclass and multilabel input.
            Should be one of the following:
            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes. Cannot be
              used with multiclass input.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support.
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Example (binary case):
        >>> from torchmetrics import AveragePrecision
        >>> pred = torch.tensor([0, 0.1, 0.8, 0.4])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> average_precision = AveragePrecision(pos_label=1)
        >>> average_precision(pred, target)
        tensor(1.)
    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> average_precision = AveragePrecision(num_classes=5, average=None)
        >>> average_precision(pred, target)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[str] = "macro",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.pos_label = pos_label
        allowed_average = ("micro", "macro", "weighted", "none", None)
        if average not in allowed_average:
            raise ValueError(
                f"Expected argument `average` to be one of {allowed_average}"
                f" but got {average}"
            )
        self.average = average

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AveragePrecision` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, num_classes, pos_label = _average_precision_update(
            preds, target, self.num_classes, self.pos_label, self.average
        )
        self.preds.append(preds)
        self.target.append(target)
        self.num_classes = num_classes
        self.pos_label = pos_label

    def compute(self) -> Union[Tensor, List[Tensor]]:
        """Compute the average precision score.
        Returns:
            tensor with average precision. If multiclass return list of such tensors, one for each class
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        if not self.num_classes:
            raise ValueError(
                f"`num_classes` bas to be positive number, but got {self.num_classes}"
            )
        return _average_precision_compute(
            preds, target, self.num_classes, self.pos_label, self.average
        )
