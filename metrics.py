from typing import List, Optional, Sequence, Tuple, Union, Any, Callable
import torch
from torch import Tensor, tensor
from torch.nn import functional as F
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod
from torchmetrics.functional.classification.stat_scores import (
    _reduce_stat_scores,
    _stat_scores_update,
    _stat_scores_compute,
)


class StatScores(Metric):
    r"""Computes the number of true positives, false positives, true negatives, false negatives.
    Related to `Type I and Type II errors`_ and the `confusion matrix`_.

    The reduction method (how the statistics are aggregated) is controlled by the
    ``reduce`` parameter, and additionally by the ``mdmc_reduce`` parameter in the
    multi-dimensional multi-class case.

    Accepts all inputs listed in :ref:`pages/classification:input types`.

    Args:
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The default value (``None``) will be interpreted
            as 1 for these inputs. Should be left at default (``None``) for all other types of inputs.

        reduce:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Counts the statistics by summing over all ``[sample, class]``
              combinations (globally). Each statistic is represented by a single integer.
            - ``'macro'``: Counts the statistics for each class separately (over all samples).
              Each statistic is represented by a ``(C,)`` tensor. Requires ``num_classes`` to be set.
            - ``'samples'``: Counts the statistics for each sample separately (over all classes).
              Each statistic is represented by a ``(N, )`` 1d tensor.

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_reduce``.

        num_classes: Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.

        ignore_index:
            Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.

        mdmc_reduce: Defines how the multi-dimensional multi-class inputs are handeled. Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class (see :ref:`pages/classification:input types` for the definition of input types).

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then the outputs are concatenated together. In each
              sample the extra axes ``...`` are flattened to become the sub-sample axis, and
              statistics for each sample are computed by treating the sub-sample axis as the
              ``N`` axis for that sample.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs are
              flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``reduce`` parameter applies as usual.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``reduce`` is none of ``"micro"``, ``"macro"`` or ``"samples"``.
        ValueError:
            If ``mdmc_reduce`` is none of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``reduce`` is set to ``"macro"`` and ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``0`` <= ``ignore_index`` < ``num_classes``.

    Example:
        >>> from torchmetrics.classification import StatScores
        >>> preds  = torch.tensor([1, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> stat_scores = StatScores(reduce='macro', num_classes=3)
        >>> stat_scores(preds, target)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])
        >>> stat_scores = StatScores(reduce='micro')
        >>> stat_scores(preds, target)
        tensor([2, 2, 6, 2, 4])

    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    # TODO: canot be used because if scripting
    # tp: Union[Tensor, List[Tensor]]
    # fp: Union[Tensor, List[Tensor]]
    # tn: Union[Tensor, List[Tensor]]
    # fn: Union[Tensor, List[Tensor]]

    def __init__(
        self,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        reduce: str = "micro",
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        mdmc_reduce: Optional[str] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.reduce = reduce
        self.mdmc_reduce = mdmc_reduce
        self.num_classes = num_classes
        self.threshold = threshold
        self.multiclass = multiclass
        self.ignore_index = ignore_index
        self.top_k = top_k

        if reduce not in ["micro", "macro", "samples"]:
            raise ValueError(f"The `reduce` {reduce} is not valid.")

        if mdmc_reduce not in [None, "samplewise", "global"]:
            raise ValueError(f"The `mdmc_reduce` {mdmc_reduce} is not valid.")

        if reduce == "macro" and (not num_classes or num_classes < 1):
            raise ValueError(
                "When you set `reduce` as 'macro', you have to provide the number of classes."
            )

        if (
            num_classes
            and ignore_index is not None
            and (not ignore_index < num_classes or num_classes == 1)
        ):
            raise ValueError(
                f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes"
            )

        default: Callable = lambda: []
        reduce_fn: Optional[str] = "cat"
        if mdmc_reduce != "samplewise" and reduce != "samples":
            if reduce == "macro":
                zeros_shape = [num_classes]
            elif reduce == "micro":
                zeros_shape = []
            else:
                raise ValueError(f'Wrong reduce="{reduce}"')
            default = lambda: torch.zeros(zeros_shape, dtype=torch.long)
            reduce_fn = "sum"

        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default(), dist_reduce_fx=reduce_fn)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        See :ref:`pages/classification:input types` for more information on input types.

        Args:
            preds: Predictions from model (probabilities, logits or labels)
            target: Ground truth values
        """

        tp, fp, tn, fn = _stat_scores_update(
            preds,
            target,
            reduce=self.reduce,
            mdmc_reduce=self.mdmc_reduce,
            threshold=self.threshold,
            num_classes=self.num_classes,
            top_k=self.top_k,
            multiclass=self.multiclass,
            ignore_index=self.ignore_index,
        )

        # Update states
        if (
            self.reduce != AverageMethod.SAMPLES
            and self.mdmc_reduce != MDMCAverageMethod.SAMPLEWISE
        ):
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn
        else:
            self.tp.append(tp)
            self.fp.append(fp)
            self.tn.append(tn)
            self.fn.append(fn)

    def _get_final_stats(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Performs concatenation on the stat scores if neccesary, before passing them to a compute function."""
        tp = torch.cat(self.tp) if isinstance(self.tp, list) else self.tp
        fp = torch.cat(self.fp) if isinstance(self.fp, list) else self.fp
        tn = torch.cat(self.tn) if isinstance(self.tn, list) else self.tn
        fn = torch.cat(self.fn) if isinstance(self.fn, list) else self.fn
        return tp, fp, tn, fn

    def compute(self) -> Tensor:
        """Computes the stat scores based on inputs passed in to ``update`` previously.

        Return:
            The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
            to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The
            shape depends on the ``reduce`` and ``mdmc_reduce`` (in case of multi-dimensional
            multi-class data) parameters:

            - If the data is not multi-dimensional multi-class, then

              - If ``reduce='micro'``, the shape will be ``(5, )``
              - If ``reduce='macro'``, the shape will be ``(C, 5)``,
                where ``C`` stands for the number of classes
              - If ``reduce='samples'``, the shape will be ``(N, 5)``, where ``N`` stands for
                the number of samples

            - If the data is multi-dimensional multi-class and ``mdmc_reduce='global'``, then

              - If ``reduce='micro'``, the shape will be ``(5, )``
              - If ``reduce='macro'``, the shape will be ``(C, 5)``
              - If ``reduce='samples'``, the shape will be ``(N*X, 5)``, where ``X`` stands for
                the product of sizes of all "extra" dimensions of the data (i.e. all dimensions
                except for ``C`` and ``N``)

            - If the data is multi-dimensional multi-class and ``mdmc_reduce='samplewise'``, then

              - If ``reduce='micro'``, the shape will be ``(N, 5)``
              - If ``reduce='macro'``, the shape will be ``(N, C, 5)``
              - If ``reduce='samples'``, the shape will be ``(N, X, 5)``
        """
        tp, fp, tn, fn = self._get_final_stats()
        return _stat_scores_compute(tp, fp, tn, fn)


def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    """prevent zero division."""
    denom[denom == 0.0] = 1
    return num / denom


def _fbeta_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    beta: float,
    ignore_index: Optional[int],
    average: str,
    mdmc_average: Optional[str],
) -> Tensor:
    """Computes f_beta metric from stat scores: true positives, false positives, true negatives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        beta: The parameter `beta` (which determines the weight of recall in the combined score)
        ignore_index: Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method
        average: Defines the reduction that is applied
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter)

    Example:
        >>> from torchmetrics.functional.classification.stat_scores import _stat_scores_update
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> tp, fp, tn, fn = _stat_scores_update(
        ...                         preds,
        ...                         target,
        ...                         reduce='micro',
        ...                         num_classes=3,
        ...                     )
        >>> _fbeta_compute(tp, fp, tn, fn, beta=0.5, ignore_index=None, average='micro', mdmc_average=None)
        tensor(0.3333)
    """
    if average == AverageMethod.MICRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        mask = tp >= 0
        precision = _safe_divide(tp[mask].sum().float(), (tp[mask] + fp[mask]).sum())
        recall = _safe_divide(tp[mask].sum().float(), (tp[mask] + fn[mask]).sum())
    else:
        precision = _safe_divide(tp.float(), tp + fp)
        recall = _safe_divide(tp.float(), tp + fn)

    num = (1 + beta**2) * precision * recall
    denom = beta**2 * precision + recall
    denom[denom == 0.0] = 1.0  # avoid division by 0

    # if classes matter and a given class is not present in both the preds and the target,
    # computing the score for this class is meaningless, thus they should be ignored
    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = torch.nonzero((tp | fn | fp) == 0).cpu()
        if ignore_index is None:
            ignore_index = meaningless_indeces
        else:
            ignore_index = torch.unique(
                torch.cat((meaningless_indeces, torch.tensor([[ignore_index]])))
            )

    if ignore_index is not None and average not in (
        AverageMethod.MICRO,
        AverageMethod.SAMPLES,
    ):
        if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
            num[..., ignore_index] = -1
            denom[..., ignore_index] = -1
        else:
            num[ignore_index, ...] = -1
            denom[ignore_index, ...] = -1

    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = (tp + fp + fn == 0) | (tp + fp + fn == -3)
        num = num[~cond]
        denom = denom[~cond]

    return _reduce_stat_scores(
        numerator=num,
        denominator=denom,
        weights=None if average != AverageMethod.WEIGHTED else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def fbeta_score(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
) -> Tensor:
    r"""
    Computes f_beta metric.

    .. math::
        F_{\beta} = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}
        {(\beta^2 * \text{precision}) + \text{recall}}

    Works with binary, multiclass, and multilabel data.
    Accepts probabilities or logits from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label logits or probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`pages/classification:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth values
        beta: beta coefficient
        average:
            Defines the reduction that is applied. Should be one of the following:

                - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
                - ``'macro'``: Calculate the metric for each class separately, and average the
                  metrics across classes (with equal weights for each class).
                - ``'weighted'``: Calculate the metric for each class separately, and average the
                  metrics across classes, weighting each class by its support (``tp + fn``).
                - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
                  the metric for every class.
                - ``'samples'``: Calculate the metric for each sample, and average the metrics
                  across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

                - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
                  multi-class.
                - ``'samplewise'``: In this case, the statistics are computed separately for each
                  sample on the ``N`` axis, and then averaged over samples.
                  The computation for each sample is done by treating the flattened extra axes ``...``
                  (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
                  and computing the metric for the sample based on that.
                - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
                  (see :ref:`pages/classification:input types`)
                  are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
                  were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:
        >>> from torchmetrics.functional import fbeta_score
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> fbeta_score(preds, target, num_classes=3, beta=0.5)
        tensor(0.3333)

    """
    allowed_average = list(AverageMethod)
    if average not in allowed_average:
        raise ValueError(
            f"The `average` has to be one of {allowed_average}, got {average}."
        )

    if mdmc_average is not None and MDMCAverageMethod.from_str(mdmc_average) is None:
        raise ValueError(
            f"The `mdmc_average` has to be one of {list(MDMCAverageMethod)}, got {mdmc_average}."
        )

    if average in [
        AverageMethod.MACRO,
        AverageMethod.WEIGHTED,
        AverageMethod.NONE,
    ] and (not num_classes or num_classes < 1):
        raise ValueError(
            f"When you set `average` as {average}, you have to provide the number of classes."
        )

    if (
        num_classes
        and ignore_index is not None
        and (not 0 <= ignore_index < num_classes or num_classes == 1)
    ):
        raise ValueError(
            f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes"
        )

    reduce = (
        AverageMethod.MACRO
        if average in [AverageMethod.WEIGHTED, AverageMethod.NONE]
        else average
    )
    tp, fp, tn, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_average,
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _fbeta_compute(tp, fp, tn, fn, beta, ignore_index, average, mdmc_average)


def f1_score(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
) -> Tensor:
    """Computes F1 metric. F1 metrics correspond to a equally weighted average of the precision and recall scores.

    Works with binary, multiclass, and multilabel data.
    Accepts probabilities or logits from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities or logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`pages/classification:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth values
        beta: it is ignored
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:
        >>> from torchmetrics.functional import f1_score
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f1_score(preds, target, num_classes=3)
        tensor(0.3333)
    """
    return fbeta_score(
        preds,
        target,
        1.0,
        average,
        mdmc_average,
        ignore_index,
        num_classes,
        threshold,
        top_k,
        multiclass,
    )


class FBetaScore(StatScores):
    r"""Computes `F-score`_, specifically:

    .. math::
        F_\beta = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}
        {(\beta^2 * \text{precision}) + \text{recall}}

    Where :math:`\beta` is some positive real factor. Works with binary, multiclass, and multilabel data.
    Accepts logit scores or probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label logits and probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        beta: Beta coefficient in the F measure.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The default value (``None``) will be interpreted
            as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"``, ``None``.

    Example:
        >>> import torch
        >>> from torchmetrics import FBetaScore
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f_beta = FBetaScore(num_classes=3, beta=0.5)
        >>> f_beta(preds, target)
        tensor(0.3333)

    """
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: Optional[int] = None,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: Optional[str] = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.beta = beta
        allowed_average = list(AverageMethod)
        if average not in allowed_average:
            raise ValueError(
                f"The `average` has to be one of {allowed_average}, got {average}."
            )

        _reduce_options = (AverageMethod.WEIGHTED, AverageMethod.NONE, None)
        if "reduce" not in kwargs:
            kwargs["reduce"] = (
                AverageMethod.MACRO if average in _reduce_options else average
            )
        if "mdmc_reduce" not in kwargs:
            kwargs["mdmc_reduce"] = mdmc_average

        super().__init__(
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs,
        )

        self.average = average

    def compute(self) -> Tensor:
        """Computes f-beta over state."""
        tp, fp, tn, fn = self._get_final_stats()
        return _fbeta_compute(
            tp, fp, tn, fn, self.beta, self.ignore_index, self.average, self.mdmc_reduce
        )


class F1Score(FBetaScore):
    """Computes F1 metric.

    F1 metrics correspond to a harmonic mean of the precision and recall scores.
    Works with binary, multiclass, and multilabel data. Accepts logits or probabilities from a model
    output or integer class values in prediction. Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold for transforming probability or logit predictions to binary ``(0,1)`` predictions, in the case
            of binary or multi-label inputs. Default value of ``0.5`` corresponds to input being probabilities.
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.
            Should be left at default (``None``) for all other types of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.


    Example:
        >>> import torch
        >>> from torchmetrics import F1Score
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f1 = F1Score(num_classes=3)
        >>> f1(preds, target)
        tensor(0.3333)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: Optional[str] = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            beta=1.0,
            threshold=threshold,
            average=average,
            mdmc_average=mdmc_average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass,
            **kwargs,
        )


class PrecisionRecallCurve(Metric):
    """Computes precision-recall pairs for different thresholds. Works for both binary and multiclass problems. In
    the case of multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems this argument should not be set as we iteratively change it in the range
            ``[0, num_classes-1]``

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (binary case):
        >>> from torchmetrics import PrecisionRecallCurve
        >>> pred = torch.tensor([0, 0.1, 0.8, 0.4])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> pr_curve = PrecisionRecallCurve(pos_label=1)
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        tensor([0.6667, 0.5000, 1.0000, 1.0000])
        >>> recall
        tensor([1.0000, 0.5000, 0.5000, 0.0000])
        >>> thresholds
        tensor([0.1000, 0.4000, 0.8000])

    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> pr_curve = PrecisionRecallCurve(num_classes=5)
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor(0.7500), tensor(0.7500), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor(0.0500)]
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.pos_label = pos_label

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        # rank_zero_warn(
        #     "Metric `PrecisionRecallCurve` will save all targets and predictions in buffer."
        #     " For large datasets this may lead to large memory footprint."
        # )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, num_classes, pos_label = _precision_recall_curve_update(
            preds, target, self.num_classes, self.pos_label
        )
        self.preds.append(preds)
        self.target.append(target)
        self.num_classes = num_classes
        self.pos_label = pos_label

    def compute(
        self,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]
    ]:
        """Compute the precision-recall curve.

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
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        if not self.num_classes:
            raise ValueError(
                f"`num_classes` bas to be positive number, but got {self.num_classes}"
            )
        return _precision_recall_curve_compute(
            preds, target, self.num_classes, self.pos_label
        )


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
