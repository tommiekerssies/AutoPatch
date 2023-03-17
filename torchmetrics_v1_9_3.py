"""
The code in this file is borrowed from TorchMetrics v1.9.3 (https://github.com/Lightning-AI/metrics/tree/v0.9.3) (Apache-2.0 License)
"""

from typing import List, Optional, Sequence, Tuple, Union, Any, no_type_check, Dict
import torch
from torch import Tensor, tensor
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_9
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
import warnings
from torch.nn import functional as F
from enum import Enum


class EnumStr(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases.
    Example:
        >>> class MyEnum(EnumStr):
        ...     ABC = 'abc'
        >>> MyEnum.from_str('Abc')
        <MyEnum.ABC: 'abc'>
        >>> {MyEnum.ABC: 123}
        {<MyEnum.ABC: 'abc'>: 123}
    """

    @classmethod
    def from_str(cls, value: str) -> Optional["EnumStr"]:
        statuses = [status for status in dir(cls) if not status.startswith("_")]
        for st in statuses:
            if st.lower() == value.lower():
                return getattr(cls, st)
        return None

    def __eq__(self, other: Union[str, "EnumStr", None]) -> bool:  # type: ignore
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        # example: set(EnumStr)
        return hash(self.name)


class AverageMethod(EnumStr):
    """Enum to represent average method.
    >>> None in list(AverageMethod)
    True
    >>> AverageMethod.NONE == None
    True
    >>> AverageMethod.NONE == 'none'
    True
    """

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = None
    SAMPLES = "samples"


def to_onehot(
    label_tensor: Tensor,
    num_classes: Optional[int] = None,
) -> Tensor:
    """Converts a dense label tensor to one-hot format.
    Args:
        label_tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C
    Returns:
        A sparse label tensor with shape [N, C, d1, d2, ...]
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> to_onehot(x)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    """
    if num_classes is None:
        num_classes = int(label_tensor.max().detach().item() + 1)

    tensor_onehot = torch.zeros(
        label_tensor.shape[0],
        num_classes,
        *label_tensor.shape[1:],
        dtype=label_tensor.dtype,
        device=label_tensor.device,
    )
    index = label_tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def select_topk(prob_tensor: Tensor, topk: int = 1, dim: int = 1) -> Tensor:
    """Convert a probability tensor to binary by selecting top-k the highest entries.
    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``, where ``C`` is in the
            position defined by the ``dim`` argument
        topk: number of the highest entries to turn into 1s
        dim: dimension on which to compare entries
    Returns:
        A binary tensor of the same shape as the input tensor of type ``torch.int32``
    Example:
        >>> x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> select_topk(x, topk=2)
        tensor([[0, 1, 1],
                [1, 1, 0]], dtype=torch.int32)
    """
    zeros = torch.zeros_like(prob_tensor)
    if topk == 1:  # argmax has better performance than topk
        topk_tensor = zeros.scatter(dim, prob_tensor.argmax(dim=dim, keepdim=True), 1.0)
    else:
        topk_tensor = zeros.scatter(dim, prob_tensor.topk(k=topk, dim=dim).indices, 1.0)
    return topk_tensor.int()


class DataType(EnumStr):
    """Enum to represent data type.
    >>> "Binary" in list(DataType)
    True
    """

    BINARY = "binary"
    MULTILABEL = "multi-label"
    MULTICLASS = "multi-class"
    MULTIDIM_MULTICLASS = "multi-dim multi-class"


def _check_for_empty_tensors(preds: Tensor, target: Tensor) -> bool:
    if preds.numel() == target.numel() == 0:
        return True
    return False


def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(
            "Predictions and targets are expected to have the same shape"
        )


def _basic_input_validation(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    multiclass: Optional[bool],
    ignore_index: Optional[int],
) -> None:
    """Perform basic validation of inputs that does not require deducing any information of the type of inputs."""
    # Skip all other checks if both preds and target are empty tensors
    if _check_for_empty_tensors(preds, target):
        return

    if target.is_floating_point():
        raise ValueError("The `target` has to be an integer tensor.")

    if ignore_index is None and target.min() < 0:
        raise ValueError("The `target` has to be a non-negative tensor.")
    elif ignore_index is not None and ignore_index >= 0 and target.min() < 0:
        raise ValueError("The `target` has to be a non-negative tensor.")

    preds_float = preds.is_floating_point()
    if not preds_float and preds.min() < 0:
        raise ValueError("If `preds` are integers, they have to be non-negative.")

    if not preds.shape[0] == target.shape[0]:
        raise ValueError(
            "The `preds` and `target` should have the same first dimension."
        )

    if multiclass is False and target.max() > 1:
        raise ValueError(
            "If you set `multiclass=False`, then `target` should not exceed 1."
        )

    if multiclass is False and not preds_float and preds.max() > 1:
        raise ValueError(
            "If you set `multiclass=False` and `preds` are integers, then `preds` should not exceed 1."
        )


def _check_shape_and_type_consistency(
    preds: Tensor, target: Tensor
) -> Tuple[DataType, int]:
    """This checks that the shape and type of inputs are consistent with each other and fall into one of the
    allowed input types (see the documentation of docstring of ``_input_format_classification``). It does not check
    for consistency of number of classes, other functions take care of that.
    It returns the name of the case in which the inputs fall, and the implied number of classes (from the ``C`` dim for
    multi-class data, or extra dim(s) for multi-label data).
    """

    preds_float = preds.is_floating_point()

    if preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        if preds_float and target.numel() > 0 and target.max() > 1:
            raise ValueError(
                "If `preds` and `target` are of shape (N, ...) and `preds` are floats, `target` should be binary."
            )

        # Get the case
        if preds.ndim == 1 and preds_float:
            case = DataType.BINARY
        elif preds.ndim == 1 and not preds_float:
            case = DataType.MULTICLASS
        elif preds.ndim > 1 and preds_float:
            case = DataType.MULTILABEL
        else:
            case = DataType.MULTIDIM_MULTICLASS
        implied_classes = preds[0].numel() if preds.numel() > 0 else 0

    elif preds.ndim == target.ndim + 1:
        if not preds_float:
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds` should be a float tensor."
            )
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " (N, C, ...), and the shape of `target` should be (N, ...)."
            )

        implied_classes = preds.shape[1] if preds.numel() > 0 else 0

        if preds.ndim == 2:
            case = DataType.MULTICLASS
        else:
            case = DataType.MULTIDIM_MULTICLASS
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )

    return case, implied_classes


def _check_num_classes_binary(num_classes: int, multiclass: Optional[bool]) -> None:
    """This checks that the consistency of `num_classes` with the data and `multiclass` param for binary data."""

    if num_classes > 2:
        raise ValueError("Your data is binary, but `num_classes` is larger than 2.")
    if num_classes == 2 and not multiclass:
        raise ValueError(
            "Your data is binary and `num_classes=2`, but `multiclass` is not True."
            " Set it to True if you want to transform binary data to multi-class format."
        )
    if num_classes == 1 and multiclass:
        raise ValueError(
            "You have binary data and have set `multiclass=True`, but `num_classes` is 1."
            " Either set `multiclass=None`(default) or set `num_classes=2`"
            " to transform binary data to multi-class format."
        )


def _check_num_classes_mc(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    multiclass: Optional[bool],
    implied_classes: int,
) -> None:
    """This checks that the consistency of `num_classes` with the data and `multiclass` param for (multi-
    dimensional) multi-class data."""

    if num_classes == 1 and multiclass is not False:
        raise ValueError(
            "You have set `num_classes=1`, but predictions are integers."
            " If you want to convert (multi-dimensional) multi-class data with 2 classes"
            " to binary/multi-label, set `multiclass=False`."
        )
    if num_classes > 1:
        if multiclass is False and implied_classes != num_classes:
            raise ValueError(
                "You have set `multiclass=False`, but the implied number of classes "
                " (from shape of inputs) does not match `num_classes`. If you are trying to"
                " transform multi-dim multi-class data with 2 classes to multi-label, `num_classes`"
                " should be either None or the product of the size of extra dimensions (...)."
                " See Input Types in Metrics documentation."
            )
        if target.numel() > 0 and num_classes <= target.max():
            raise ValueError(
                "The highest label in `target` should be smaller than `num_classes`."
            )
        if preds.shape != target.shape and num_classes != implied_classes:
            raise ValueError(
                "The size of C dimension of `preds` does not match `num_classes`."
            )


def _check_num_classes_ml(
    num_classes: int, multiclass: Optional[bool], implied_classes: int
) -> None:
    """This checks that the consistency of ``num_classes`` with the data and ``multiclass`` param for multi-label
    data."""

    if multiclass and num_classes != 2:
        raise ValueError(
            "Your have set `multiclass=True`, but `num_classes` is not equal to 2."
            " If you are trying to transform multi-label data to 2 class multi-dimensional"
            " multi-class, you should set `num_classes` to either 2 or None."
        )
    if not multiclass and num_classes != implied_classes:
        raise ValueError(
            "The implied number of classes (from shape of inputs) does not match num_classes."
        )


def _check_top_k(
    top_k: int,
    case: str,
    implied_classes: int,
    multiclass: Optional[bool],
    preds_float: bool,
) -> None:
    if case == DataType.BINARY:
        raise ValueError("You can not use `top_k` parameter with binary data.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("The `top_k` has to be an integer larger than 0.")
    if not preds_float:
        raise ValueError(
            "You have set `top_k`, but you do not have probability predictions."
        )
    if multiclass is False:
        raise ValueError("If you set `multiclass=False`, you can not set `top_k`.")
    if case == DataType.MULTILABEL and multiclass:
        raise ValueError(
            "If you want to transform multi-label data to 2 class multi-dimensional"
            "multi-class data using `multiclass=True`, you can not use `top_k`."
        )
    if top_k >= implied_classes:
        raise ValueError(
            "The `top_k` has to be strictly smaller than the `C` dimension of `preds`."
        )


def _check_classification_inputs(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    num_classes: Optional[int],
    multiclass: Optional[bool],
    top_k: Optional[int],
    ignore_index: Optional[int] = None,
) -> DataType:
    """Performs error checking on inputs for classification.
    This ensures that preds and target take one of the shape/type combinations that are
    specified in ``_input_format_classification`` docstring. It also checks the cases of
    over-rides with ``multiclass`` by checking (for multi-class and multi-dim multi-class
    cases) that there are only up to 2 distinct labels.
    In case where preds are floats (probabilities), it is checked whether they are in ``[0,1]`` interval.
    When ``num_classes`` is given, it is checked that it is consistent with input cases (binary,
    multi-label, ...), and that, if available, the implied number of classes in the ``C``
    dimension is consistent with it (as well as that max label in target is smaller than it).
    When ``num_classes`` is not specified in these cases, consistency of the highest target
    value against ``C`` dimension is checked for (multi-dimensional) multi-class cases.
    If ``top_k`` is set (not None) for inputs that do not have probability predictions (and
    are not binary), an error is raised. Similarly, if ``top_k`` is set to a number that
    is higher than or equal to the ``C`` dimension of ``preds``, an error is raised.
    Preds and target tensors are expected to be squeezed already - all dimensions should be
    greater than 1, except perhaps the first one (``N``).
    Args:
        preds: Tensor with predictions (labels or probabilities)
        target: Tensor with ground truth labels, always integers (labels)
        threshold:
            Threshold value for transforming probability/logit predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be inferred
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of the highest probability entries for each sample to convert to 1s - relevant
            only for inputs with probability predictions. The default value (``None``) will be
            interpreted as 1 for these inputs. If this parameter is set for multi-label inputs,
            it will take precedence over threshold.
            Should be left unset (``None``) for inputs with label predictions.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/overview:using the multiclass parameter>`
            for a more detailed explanation and examples.
    Return:
        case: The case the inputs fall in, one of 'binary', 'multi-class', 'multi-label' or
            'multi-dim multi-class'
    """

    # Basic validation (that does not need case/type information)
    _basic_input_validation(preds, target, threshold, multiclass, ignore_index)

    # Check that shape/types fall into one of the cases
    case, implied_classes = _check_shape_and_type_consistency(preds, target)

    # Check consistency with the `C` dimension in case of multi-class data
    if preds.shape != target.shape:
        if multiclass is False and implied_classes != 2:
            raise ValueError(
                "You have set `multiclass=False`, but have more than 2 classes in your data,"
                " based on the C dimension of `preds`."
            )
        if target.max() >= implied_classes:
            raise ValueError(
                "The highest label in `target` should be smaller than the size of the `C` dimension of `preds`."
            )

    # Check that num_classes is consistent
    if num_classes:
        if case == DataType.BINARY:
            _check_num_classes_binary(num_classes, multiclass)
        elif case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS):
            _check_num_classes_mc(
                preds, target, num_classes, multiclass, implied_classes
            )
        elif case.MULTILABEL:
            _check_num_classes_ml(num_classes, multiclass, implied_classes)

    # Check that top_k is consistent
    if top_k is not None:
        _check_top_k(
            top_k, case, implied_classes, multiclass, preds.is_floating_point()
        )

    return case


def _input_squeeze(
    preds: Tensor,
    target: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Remove excess dimensions."""
    if preds.shape[0] == 1:
        preds, target = preds.squeeze().unsqueeze(0), target.squeeze().unsqueeze(0)
    else:
        preds, target = preds.squeeze(), target.squeeze()
    return preds, target


def _input_format_classification(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, DataType]:
    """Convert preds and target tensors into common format.
    Preds and targets are supposed to fall into one of these categories (and are
    validated to make sure this is the case):
    * Both preds and target are of shape ``(N,)``, and both are integers (multi-class)
    * Both preds and target are of shape ``(N,)``, and target is binary, while preds
      are a float (binary)
    * preds are of shape ``(N, C)`` and are floats, and target is of shape ``(N,)`` and
      is integer (multi-class)
    * preds and target are of shape ``(N, ...)``, target is binary and preds is a float
      (multi-label)
    * preds are of shape ``(N, C, ...)`` and are floats, target is of shape ``(N, ...)``
      and is integer (multi-dimensional multi-class)
    * preds and target are of shape ``(N, ...)`` both are integers (multi-dimensional
      multi-class)
    To avoid ambiguities, all dimensions of size 1, except the first one, are squeezed out.
    The returned output tensors will be binary tensors of the same shape, either ``(N, C)``
    of ``(N, C, X)``, the details for each case are described below. The function also returns
    a ``case`` string, which describes which of the above cases the inputs belonged to - regardless
    of whether this was "overridden" by other settings (like ``multiclass``).
    In binary case, targets are normally returned as ``(N,1)`` tensor, while preds are transformed
    into a binary tensor (elements become 1 if the probability is greater than or equal to
    ``threshold`` or 0 otherwise). If ``multiclass=True``, then both targets are preds
    become ``(N, 2)`` tensors by a one-hot transformation; with the thresholding being applied to
    preds first.
    In multi-class case, normally both preds and targets become ``(N, C)`` binary tensors; targets
    by a one-hot transformation and preds by selecting ``top_k`` largest entries (if their original
    shape was ``(N,C)``). However, if ``multiclass=False``, then targets and preds will be
    returned as ``(N,1)`` tensor.
    In multi-label case, normally targets and preds are returned as ``(N, C)`` binary tensors, with
    preds being binarized as in the binary case. Here the ``C`` dimension is obtained by flattening
    all dimensions after the first one. However, if ``multiclass=True``, then both are returned as
    ``(N, 2, C)``, by an equivalent transformation as in the binary case.
    In multi-dimensional multi-class case, normally both target and preds are returned as
    ``(N, C, X)`` tensors, with ``X`` resulting from flattening of all dimensions except ``N`` and
    ``C``. The transformations performed here are equivalent to the multi-class case. However, if
    ``multiclass=False`` (and there are up to two classes), then the data is returned as
    ``(N, X)`` binary tensors (multi-label).
    Note:
        Where a one-hot transformation needs to be performed and the number of classes
        is not implicitly given by a ``C`` dimension, the new ``C`` dimension will either be
        equal to ``num_classes``, if it is given, or the maximum label value in preds and
        target.
    Args:
        preds: Tensor with predictions (labels or probabilities)
        target: Tensor with ground truth labels, always integers (labels)
        threshold:
            Threshold value for transforming probability/logit predictions to binary
            (0 or 1) predictions, in the case of binary or multi-label inputs.
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be inferred
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of the highest probability entries for each sample to convert to 1s - relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interpreted as 1 for these inputs.
            Should be left unset (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/overview:using the multiclass parameter>`
            for a more detailed explanation and examples.
    Returns:
        preds: binary tensor of shape ``(N, C)`` or ``(N, C, X)``
        target: binary tensor of shape ``(N, C)`` or ``(N, C, X)``
        case: The case the inputs fall in, one of ``'binary'``, ``'multi-class'``, ``'multi-label'`` or
            ``'multi-dim multi-class'``
    """
    # Remove excess dimensions
    preds, target = _input_squeeze(preds, target)

    # Convert half precision tensors to full precision, as not all ops are supported
    # for example, min() is not supported
    if preds.dtype == torch.float16:
        preds = preds.float()

    case = _check_classification_inputs(
        preds,
        target,
        threshold=threshold,
        num_classes=num_classes,
        multiclass=multiclass,
        top_k=top_k,
        ignore_index=ignore_index,
    )

    if case in (DataType.BINARY, DataType.MULTILABEL) and not top_k:
        preds = (preds >= threshold).int()
        num_classes = num_classes if not multiclass else 2

    if case == DataType.MULTILABEL and top_k:
        preds = select_topk(preds, top_k)

    if case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS) or multiclass:
        if preds.is_floating_point():
            num_classes = preds.shape[1]
            preds = select_topk(preds, top_k or 1)
        else:
            num_classes = (
                num_classes if num_classes else max(preds.max(), target.max()) + 1
            )
            preds = to_onehot(preds, max(2, num_classes))

        target = to_onehot(target, max(2, num_classes))  # type: ignore

        if multiclass is False:
            preds, target = preds[:, 1, ...], target[:, 1, ...]

    if not _check_for_empty_tensors(preds, target):
        if (
            case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS)
            and multiclass is not False
        ) or multiclass:
            target = target.reshape(target.shape[0], target.shape[1], -1)
            preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
        else:
            target = target.reshape(target.shape[0], -1)
            preds = preds.reshape(preds.shape[0], -1)

    # Some operations above create an extra dimension for MC/binary case - this removes it
    if preds.ndim > 2:
        preds, target = preds.squeeze(-1), target.squeeze(-1)

    return preds.int(), target.int(), case


def _input_format_classification_one_hot(
    num_classes: int,
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multilabel: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Convert preds and target tensors into one hot spare label tensors.
    Args:
        num_classes: number of classes
        preds: either tensor with labels, tensor with probabilities/logits or multilabel tensor
        target: tensor with ground-true labels
        threshold: float used for thresholding multilabel input
        multilabel: boolean flag indicating if input is multilabel
    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same number of dimensions
            or one additional dimension for ``preds``.
    Returns:
        preds: one hot tensor of shape [num_classes, -1] with predicted labels
        target: one hot tensors of shape [num_classes, -1] with true labels
    """
    if preds.ndim not in (target.ndim, target.ndim + 1):
        raise ValueError(
            "preds and target must have same number of dimensions, or one additional dimension for preds"
        )

    if preds.ndim == target.ndim + 1:
        # multi class probabilities
        preds = torch.argmax(preds, dim=1)

    if (
        preds.ndim == target.ndim
        and preds.dtype in (torch.long, torch.int)
        and num_classes > 1
        and not multilabel
    ):
        # multi-class
        preds = to_onehot(preds, num_classes=num_classes)
        target = to_onehot(target, num_classes=num_classes)

    elif preds.ndim == target.ndim and preds.is_floating_point():
        # binary or multilabel probabilities
        preds = (preds >= threshold).long()

    # transpose class as first dim and reshape
    if preds.ndim > 1:
        preds = preds.transpose(1, 0)
        target = target.transpose(1, 0)

    return preds.reshape(num_classes, -1), target.reshape(num_classes, -1)


def _check_retrieval_functional_inputs(
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Check ``preds`` and ``target`` tensors are of the same shape and of the correct data type.
    Args:
        preds: either tensor with scores/logits
        target: tensor with ground true labels
        allow_non_binary_target: whether to allow target to contain non-binary values
    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty
            or not of the correct ``dtypes``.
    Returns:
        preds: as torch.float32
        target: as torch.long if not floating point else torch.float32
    """
    if preds.shape != target.shape:
        raise ValueError("`preds` and `target` must be of the same shape")

    if not preds.numel() or not preds.size():
        raise ValueError(
            "`preds` and `target` must be non-empty and non-scalar tensors"
        )

    return _check_retrieval_target_and_prediction_types(
        preds, target, allow_non_binary_target=allow_non_binary_target
    )


def _check_retrieval_inputs(
    indexes: Tensor,
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Check ``indexes``, ``preds`` and ``target`` tensors are of the same shape and of the correct data type.
    Args:
        indexes: tensor with queries indexes
        preds: tensor with scores/logits
        target: tensor with ground true labels
        ignore_index: ignore predictions where targets are equal to this number
    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty or not of the correct ``dtypes``.
    Returns:
        indexes: as ``torch.long``
        preds: as ``torch.float32``
        target: as ``torch.long``
    """
    if indexes.shape != preds.shape or preds.shape != target.shape:
        raise ValueError("`indexes`, `preds` and `target` must be of the same shape")

    if indexes.dtype is not torch.long:
        raise ValueError("`indexes` must be a tensor of long integers")

    # remove predictions where target is equal to `ignore_index`
    if ignore_index is not None:
        valid_positions = target != ignore_index
        indexes, preds, target = (
            indexes[valid_positions],
            preds[valid_positions],
            target[valid_positions],
        )

    if not indexes.numel() or not indexes.size():
        raise ValueError(
            "`indexes`, `preds` and `target` must be non-empty and non-scalar tensors",
        )

    preds, target = _check_retrieval_target_and_prediction_types(
        preds, target, allow_non_binary_target=allow_non_binary_target
    )

    return indexes.long().flatten(), preds, target


def _check_retrieval_target_and_prediction_types(
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Check ``preds`` and ``target`` tensors are of the same shape and of the correct data type.
    Args:
        preds: either tensor with scores/logits
        target: tensor with ground true labels
        allow_non_binary_target: whether to allow target to contain non-binary values
    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty or not of the correct ``dtypes``.
    """
    if target.dtype not in (
        torch.bool,
        torch.long,
        torch.int,
    ) and not torch.is_floating_point(target):
        raise ValueError("`target` must be a tensor of booleans, integers or floats")

    if not preds.is_floating_point():
        raise ValueError("`preds` must be a tensor of floats")

    if not allow_non_binary_target and (target.max() > 1 or target.min() < 0):
        raise ValueError("`target` must contain `binary` values")

    target = target.float() if target.is_floating_point() else target.long()
    preds = preds.float()

    return preds.flatten(), target.flatten()


def _allclose_recursive(res1: Any, res2: Any, atol: float = 1e-8) -> bool:
    """Utility function for recursively asserting that two results are within a certain tolerance."""
    # single output compare
    if isinstance(res1, Tensor):
        return torch.allclose(res1, res2, atol=atol)
    elif isinstance(res1, str):
        return res1 == res2
    elif isinstance(res1, Sequence):
        return all(_allclose_recursive(r1, r2) for r1, r2 in zip(res1, res2))
    elif isinstance(res1, Mapping):
        return all(_allclose_recursive(res1[k], res2[k]) for k in res1.keys())
    return res1 == res2


@no_type_check
def check_forward_full_state_property(
    metric_class,
    init_args: Dict[str, Any] = {},
    input_args: Dict[str, Any] = {},
    num_update_to_compare: Sequence[int] = [10, 100, 1000],
    reps: int = 5,
) -> bool:
    """Utility function for checking if the new ``full_state_update`` property can safely be set to ``False`` which
    will for most metrics results in a speedup when using ``forward``.
    Args:
        metric_class: metric class object that should be checked
        init_args: dict containing arguments for initializing the metric class
        input_args: dict containing arguments to pass to ``forward``
        num_update_to_compare: if we successfully detech that the flag is safe to set to ``False``
            we will run some speedup test. This arg should be a list of integers for how many
            steps to compare over.
        reps: number of repetitions of speedup test
    Example (states in ``update`` are independent, save to set ``full_state_update=False``)
        >>> from torchmetrics import ConfusionMatrix
        >>> check_forward_full_state_property(
        ...     ConfusionMatrix,
        ...     init_args = {'num_classes': 3},
        ...     input_args = {'preds': torch.randint(3, (10,)), 'target': torch.randint(3, (10,))},
        ... )  # doctest: +ELLIPSIS
        Full state for 10 steps took: ...
        Partial state for 10 steps took: ...
        Full state for 100 steps took: ...
        Partial state for 100 steps took: ...
        Full state for 1000 steps took: ...
        Partial state for 1000 steps took: ...
        Recommended setting `full_state_update=False`
    Example (states in ``update`` are dependend meaning that ``full_state_update=True``):
        >>> from torchmetrics import ConfusionMatrix
        >>> class MyMetric(ConfusionMatrix):
        ...     def update(self, preds, target):
        ...         super().update(preds, target)
        ...         # by construction make future states dependent on prior states
        ...         if self.confmat.sum() > 20:
        ...             self.reset()
        >>> check_forward_full_state_property(
        ...     MyMetric,
        ...     init_args = {'num_classes': 3},
        ...     input_args = {'preds': torch.randint(3, (10,)), 'target': torch.randint(3, (10,))},
        ... )
        Recommended setting `full_state_update=True`
    """

    class FullState(metric_class):
        full_state_update = True

    class PartState(metric_class):
        full_state_update = False

    fullstate = FullState(**init_args)
    partstate = PartState(**init_args)

    equal = True
    for _ in range(num_update_to_compare[0]):
        out1 = fullstate(**input_args)
        try:  # if it fails, the code most likely need access to the full state
            out2 = partstate(**input_args)
        except RuntimeError:
            equal = False
            break
        equal = equal & _allclose_recursive(out1, out2)

    res1 = fullstate.compute()
    try:  # if it fails, the code most likely need access to the full state
        res2 = partstate.compute()
    except RuntimeError:
        equal = False
    equal = equal & _allclose_recursive(res1, res2)

    if not equal:  # we can stop early because the results did not match
        print("Recommended setting `full_state_update=True`")
        return

    # Do timings
    res = torch.zeros(2, len(num_update_to_compare), reps)
    for i, metric in enumerate([fullstate, partstate]):
        for j, t in enumerate(num_update_to_compare):
            for r in range(reps):
                start = perf_counter()
                for _ in range(t):
                    _ = metric(**input_args)
                end = perf_counter()
                res[i, j, r] = end - start
                metric.reset()

    mean = torch.mean(res, -1)
    std = torch.std(res, -1)

    for t in range(len(num_update_to_compare)):
        print(
            f"Full state for {num_update_to_compare[t]} steps took: {mean[0, t]}+-{std[0, t]:0.3f}"
        )
        print(
            f"Partial state for {num_update_to_compare[t]} steps took: {mean[1, t]:0.3f}+-{std[1, t]:0.3f}"
        )

    faster = (
        mean[1, -1] < mean[0, -1]
    ).item()  # if faster on average, we recommend upgrading
    print(f"Recommended setting `full_state_update={not faster}`")


def is_overridden(method_name: str, instance: object, parent: object) -> bool:
    """Check if a method has been overridden by an instance compared to its parent class."""
    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    # `functools.wraps()` support
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    # `Mock(wraps=...)` support
    if isinstance(instance_attr, Mock):
        # access the wrapped function
        instance_attr = instance_attr._mock_wraps
    # `partial` support
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False

    parent_attr = getattr(parent, method_name, None)
    if parent_attr is None:
        raise ValueError("The parent should define the method")

    return instance_attr.__code__ != parent_attr.__code__


def _auc_update(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute area under the curve. Checks if the 2 input tenser have
    the same number of elements and if they are 1d.
    Args:
        x: x-coordinates
        y: y-coordinates
    """

    if x.ndim > 1:
        x = x.squeeze()

    if y.ndim > 1:
        y = y.squeeze()

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            f"Expected both `x` and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}"
        )
    if x.numel() != y.numel():
        raise ValueError(
            f"Expected the same number of elements in `x` and `y` tensor but received {x.numel()} and {y.numel()}"
        )
    return x, y


def _auc_compute_without_check(x: Tensor, y: Tensor, direction: float) -> Tensor:
    """Computes area under the curve using the trapezoidal rule. Assumes increasing or decreasing order of `x`.
    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        direction: 1 if increaing, -1 if decreasing
    Example:
        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> x, y = _auc_update(x, y)
        >>> _auc_compute_without_check(x, y, direction=1.0)
        tensor(4.)
    """

    with torch.no_grad():
        auc_: Tensor = torch.trapz(y, x) * direction
    return auc_


def _auc_compute(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Computes area under the curve using the trapezoidal rule. Checks for increasing or decreasing order of `x`.
    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing
    Example:
        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> x, y = _auc_update(x, y)
        >>> _auc_compute(x, y)
        tensor(4.)
        >>> _auc_compute(x, y, reorder=True)
        tensor(4.)
    """

    with torch.no_grad():
        if reorder:
            # TODO: include stable=True arg when pytorch v1.9 is released
            x, x_idx = torch.sort(x)
            y = y[x_idx]

        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx <= 0).all():
                direction = -1.0
            else:
                raise ValueError(
                    "The `x` tensor is neither increasing or decreasing. Try setting the reorder argument to `True`."
                )
        else:
            direction = 1.0
        return _auc_compute_without_check(x, y, direction)


def auc(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Computes Area Under the Curve (AUC) using the trapezoidal rule.
    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing
    Return:
        Tensor containing AUC score
    Raises:
        ValueError:
            If both ``x`` and ``y`` tensors are not ``1d``.
        ValueError:
            If both ``x`` and ``y`` don't have the same numnber of elements.
        ValueError:
            If ``x`` tesnsor is neither increasing nor decreasing.
    Example:
        >>> from torchmetrics.functional import auc
        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> auc(x, y)
        tensor(4.)
        >>> auc(x, y, reorder=True)
        tensor(4.)
    """
    x, y = _auc_update(x, y)
    return _auc_compute(x, y, reorder=reorder)


def _roc_update(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
) -> Tuple[Tensor, Tensor, int, Optional[int]]:
    """Updates and returns variables required to compute the Receiver Operating Characteristic.
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

    return _precision_recall_curve_update(preds, target, num_classes, pos_label)


def _roc_compute_single_class(
    preds: Tensor,
    target: Tensor,
    pos_label: int,
    sample_weights: Optional[Sequence] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes Receiver Operating Characteristic for single class inputs. Returns tensor with false positive
    rates, tensor with true positive rates, tensor with thresholds used for computing false- and true-postive
    rates.
    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        sample_weights: sample weights for each data point
    """

    fps, tps, thresholds = _binary_clf_curve(
        preds=preds, target=target, sample_weights=sample_weights, pos_label=pos_label
    )
    # Add an extra threshold position to make sure that the curve starts at (0, 0)
    tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
    fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
    thresholds = torch.cat([thresholds[0][None] + 1, thresholds])

    if fps[-1] <= 0:
        rank_zero_warn(
            "No negative samples in targets, false positive value should be meaningless."
            " Returning zero tensor in false positive score",
            UserWarning,
        )
        fpr = torch.zeros_like(thresholds)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        rank_zero_warn(
            "No positive samples in targets, true positive value should be meaningless."
            " Returning zero tensor in true positive score",
            UserWarning,
        )
        tpr = torch.zeros_like(thresholds)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def _roc_compute_multi_class(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    sample_weights: Optional[Sequence] = None,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Computes Receiver Operating Characteristic for multi class inputs. Returns tensor with false positive rates,
    tensor with true positive rates, tensor with thresholds used for computing false- and true-postive rates.
    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_classes: number of classes
        sample_weights: sample weights for each data point
    """

    fpr, tpr, thresholds = [], [], []
    for cls in range(num_classes):
        if preds.shape == target.shape:
            target_cls = target[:, cls]
            pos_label = 1
        else:
            target_cls = target
            pos_label = cls
        res = roc(
            preds=preds[:, cls],
            target=target_cls,
            num_classes=1,
            pos_label=pos_label,
            sample_weights=sample_weights,
        )
        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])

    return fpr, tpr, thresholds


def _roc_compute(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    pos_label: Optional[int] = None,
    sample_weights: Optional[Sequence] = None,
) -> Union[
    Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]
]:
    """Computes Receiver Operating Characteristic based on the number of classes.
    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems.
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        sample_weights: sample weights for each data point
    Example:
        >>> # binary case
        >>> preds = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> pos_label = 1
        >>> preds, target, num_classes, pos_label = _roc_update(preds, target, pos_label=pos_label)
        >>> fpr, tpr, thresholds = _roc_compute(preds, target, num_classes, pos_label)
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([4, 3, 2, 1, 0])
        >>> # multiclass case
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> num_classes = 4
        >>> preds, target, num_classes, pos_label = _roc_update(preds, target, num_classes)
        >>> fpr, tpr, thresholds = _roc_compute(preds, target, num_classes)
        >>> fpr
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]), tensor([0.0000, 0.3333, 1.0000])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.])]
        >>> thresholds
        [tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500])]
    """

    with torch.no_grad():
        if num_classes == 1 and preds.ndim == 1:  # binary
            if pos_label is None:
                pos_label = 1
            return _roc_compute_single_class(preds, target, pos_label, sample_weights)
        return _roc_compute_multi_class(preds, target, num_classes, sample_weights)


def roc(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    sample_weights: Optional[Sequence] = None,
) -> Union[
    Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]
]:
    """Computes the Receiver Operating Characteristic (ROC). Works with both binary, multiclass and multilabel
    input.
    .. note::
        If either the positive class or negative class is completly missing in the target tensor,
        the roc values are not well-defined in this case and a tensor of zeros will be returned (either fpr
        or tpr depending on what class is missing) together with a warning.
    Args:
        preds: predictions from model (logits or probabilities)
        target: ground truth values
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems.
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        sample_weights: sample weights for each data point
    Returns:
        3-element tuple containing
        fpr: tensor with false positive rates.
            If multiclass or multilabel, this is a list of such tensors, one for each class/label.
        tpr: tensor with true positive rates.
            If multiclass or multilabel, this is a list of such tensors, one for each class/label.
        thresholds: tensor with thresholds used for computing false- and true postive rates
            If multiclass or multilabel, this is a list of such tensors, one for each class/label.
    Example (binary case):
        >>> from torchmetrics.functional import roc
        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> fpr, tpr, thresholds = roc(pred, target, pos_label=1)
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([4, 3, 2, 1, 0])
    Example (multiclass case):
        >>> from torchmetrics.functional import roc
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> fpr, tpr, thresholds = roc(pred, target, num_classes=4)
        >>> fpr
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]), tensor([0.0000, 0.3333, 1.0000])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.])]
        >>> thresholds
        [tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500])]
    Example (multilabel case):
        >>> from torchmetrics.functional import roc
        >>> pred = torch.tensor([[0.8191, 0.3680, 0.1138],
        ...                      [0.3584, 0.7576, 0.1183],
        ...                      [0.2286, 0.3468, 0.1338],
        ...                      [0.8603, 0.0745, 0.1837]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> fpr, tpr, thresholds = roc(pred, target, num_classes=3, pos_label=1)
        >>> fpr
        [tensor([0.0000, 0.3333, 0.3333, 0.6667, 1.0000]),
         tensor([0., 0., 0., 1., 1.]),
         tensor([0.0000, 0.0000, 0.3333, 0.6667, 1.0000])]
        >>> tpr
        [tensor([0., 0., 1., 1., 1.]), tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000]), tensor([0., 1., 1., 1., 1.])]
        >>> thresholds
        [tensor([1.8603, 0.8603, 0.8191, 0.3584, 0.2286]),
         tensor([1.7576, 0.7576, 0.3680, 0.3468, 0.0745]),
         tensor([1.1837, 0.1837, 0.1338, 0.1183, 0.1138])]
    """
    preds, target, num_classes, pos_label = _roc_update(
        preds, target, num_classes, pos_label
    )
    return _roc_compute(preds, target, num_classes, pos_label, sample_weights)


def _auroc_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, DataType]:
    """Updates and returns variables required to compute Area Under the Receiver Operating Characteristic Curve.
    Validates the inputs and returns the mode of the inputs.
    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    # use _input_format_classification for validating the input and get the mode of data
    _, _, mode = _input_format_classification(preds, target)

    if mode == "multi class multi dim":
        n_classes = preds.shape[1]
        preds = preds.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)
        target = target.flatten()
    if mode == "multi-label" and preds.ndim > 2:
        n_classes = preds.shape[1]
        preds = preds.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)
        target = target.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)

    return preds, target, mode


def _auroc_compute(
    preds: Tensor,
    target: Tensor,
    mode: DataType,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    max_fpr: Optional[float] = None,
    sample_weights: Optional[Sequence] = None,
) -> Tensor:
    """Computes Area Under the Receiver Operating Characteristic Curve.
    Args:
        preds: predictions from model (logits or probabilities)
        target: Ground truth labels
        mode: 'multi class multi dim' or 'multi-label' or 'binary'
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class.
            Should be set to ``None`` for binary problems
        average: Defines the reduction that is applied to the output:
        max_fpr: If not ``None``, calculates standardized partial AUC over the
            range ``[0, max_fpr]``. Should be a float between 0 and 1.
        sample_weights: sample weights for each data point
    Example:
        >>> # binary case
        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> preds, target, mode = _auroc_update(preds, target)
        >>> _auroc_compute(preds, target, mode, pos_label=1)
        tensor(0.5000)
        >>> # multiclass case
        >>> preds = torch.tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = torch.tensor([0, 1, 1, 2, 2])
        >>> preds, target, mode = _auroc_update(preds, target)
        >>> _auroc_compute(preds, target, mode, num_classes=3)
        tensor(0.7778)
    """

    # binary mode override num_classes
    if mode == DataType.BINARY:
        num_classes = 1

    # check max_fpr parameter
    if max_fpr is not None:
        if not isinstance(max_fpr, float) and 0 < max_fpr <= 1:
            raise ValueError(
                f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}"
            )

        if not _TORCH_GREATER_EQUAL_1_9:
            raise RuntimeError()
        # if _TORCH_LOWER_1_6:
        #     raise RuntimeError(
        #         "`max_fpr` argument requires `torch.bucketize` which"
        #         " is not available below PyTorch version 1.6"
        #     )

        # max_fpr parameter is only support for binary
        if mode != DataType.BINARY:
            raise ValueError(
                "Partial AUC computation not available in multilabel/multiclass setting,"
                f" 'max_fpr' must be set to `None`, received `{max_fpr}`."
            )

    # calculate fpr, tpr
    if mode == DataType.MULTILABEL:
        if average == AverageMethod.MICRO:
            fpr, tpr, _ = roc(
                preds.flatten(), target.flatten(), 1, pos_label, sample_weights
            )
        elif num_classes:
            # for multilabel we iteratively evaluate roc in a binary fashion
            output = [
                roc(
                    preds[:, i],
                    target[:, i],
                    num_classes=1,
                    pos_label=1,
                    sample_weights=sample_weights,
                )
                for i in range(num_classes)
            ]
            fpr = [o[0] for o in output]
            tpr = [o[1] for o in output]
        else:
            raise ValueError(
                "Detected input to be `multilabel` but you did not provide `num_classes` argument"
            )
    else:
        if mode != DataType.BINARY:
            if num_classes is None:
                raise ValueError(
                    "Detected input to `multiclass` but you did not provide `num_classes` argument"
                )
            if (
                average == AverageMethod.WEIGHTED
                and len(torch.unique(target)) < num_classes
            ):
                # If one or more classes has 0 observations, we should exclude them, as its weight will be 0
                target_bool_mat = torch.zeros(
                    (len(target), num_classes), dtype=bool, device=target.device
                )
                target_bool_mat[torch.arange(len(target)), target.long()] = 1
                class_observed = target_bool_mat.sum(axis=0) > 0
                for c in range(num_classes):
                    if not class_observed[c]:
                        warnings.warn(
                            f"Class {c} had 0 observations, omitted from AUROC calculation",
                            UserWarning,
                        )
                preds = preds[:, class_observed]
                target = target_bool_mat[:, class_observed]
                target = torch.where(target)[1]
                num_classes = class_observed.sum()
                if num_classes == 1:
                    raise ValueError(
                        "Found 1 non-empty class in `multiclass` AUROC calculation"
                    )
        fpr, tpr, _ = roc(preds, target, num_classes, pos_label, sample_weights)

    # calculate standard roc auc score
    if max_fpr is None or max_fpr == 1:
        if mode == DataType.MULTILABEL and average == AverageMethod.MICRO:
            pass
        elif num_classes != 1:
            # calculate auc scores per class
            auc_scores = [
                _auc_compute_without_check(x, y, 1.0) for x, y in zip(fpr, tpr)
            ]

            # calculate average
            if average == AverageMethod.NONE:
                return tensor(auc_scores)
            if average == AverageMethod.MACRO:
                return torch.mean(torch.stack(auc_scores))
            if average == AverageMethod.WEIGHTED:
                if mode == DataType.MULTILABEL:
                    support = torch.sum(target, dim=0)
                else:
                    support = _bincount(target.flatten(), minlength=num_classes)
                return torch.sum(torch.stack(auc_scores) * support / support.sum())

            allowed_average = (
                AverageMethod.NONE.value,
                AverageMethod.MACRO.value,
                AverageMethod.WEIGHTED.value,
            )
            raise ValueError(
                f"Argument `average` expected to be one of the following: {allowed_average} but got {average}"
            )

        return _auc_compute_without_check(fpr, tpr, 1.0)

    _device = fpr.device if isinstance(fpr, Tensor) else fpr[0].device
    max_area: Tensor = tensor(max_fpr, device=_device)
    # Add a single point at max_fpr and interpolate its tpr value
    stop = torch.bucketize(max_area, fpr, out_int32=True, right=True)
    weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
    interp_tpr: Tensor = torch.lerp(tpr[stop - 1], tpr[stop], weight)
    tpr = torch.cat([tpr[:stop], interp_tpr.view(1)])
    fpr = torch.cat([fpr[:stop], max_area.view(1)])

    # Compute partial AUC
    partial_auc = _auc_compute_without_check(fpr, tpr, 1.0)

    # McClish correction: standardize result to be 0.5 if non-discriminant and 1 if maximal
    min_area: Tensor = 0.5 * max_area**2
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


class AUROC(Metric):
    r"""Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_).
    Works for both binary, multilabel and multiclass problems. In the case of
    multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    For non-binary input, if the ``preds`` and ``target`` tensor have the same
    size the input will be interpretated as multilabel and if ``preds`` have one
    dimension more than the ``target`` tensor the input will be interpretated as
    multiclass.

    .. note::
        If either the positive class or negative class is completly missing in the target tensor,
        the auroc score is meaningless in this case and a score of 0 will be returned together
        with an warning.

    Args:
        num_classes: integer with number of classes for multi-label and multiclass problems.

            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average:
            - ``'micro'`` computes metric globally. Only works for multilabel problems
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``None`` computes and returns the metric per class
        max_fpr:
            If not ``None``, calculates standardized partial AUC over the
            range ``[0, max_fpr]``. Should be a float between 0 and 1.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``average`` is none of ``None``, ``"macro"`` or ``"weighted"``.
        ValueError:
            If ``max_fpr`` is not a ``float`` in the range ``(0, 1]``.
        RuntimeError:
            If ``PyTorch version`` is ``below 1.6`` since ``max_fpr`` requires ``torch.bucketize``
            which is not available below 1.6.
        ValueError:
            If the mode of data (binary, multi-label, multi-class) changes between batches.

    Example (binary case):
        >>> from torchmetrics import AUROC
        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> auroc = AUROC(pos_label=1)
        >>> auroc(preds, target)
        tensor(0.5000)

    Example (multiclass case):
        >>> preds = torch.tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = torch.tensor([0, 1, 1, 2, 2])
        >>> auroc = AUROC(num_classes=3)
        >>> auroc(preds, target)
        tensor(0.7778)

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[str] = "macro",
        max_fpr: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.pos_label = pos_label
        self.average = average
        self.max_fpr = max_fpr

        allowed_average = (None, "macro", "weighted", "micro")
        if self.average not in allowed_average:
            raise ValueError(
                f"Argument `average` expected to be one of the following: {allowed_average} but got {average}"
            )

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) or not 0 < max_fpr <= 1:
                raise ValueError(
                    f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}"
                )

            if _TORCH_LOWER_1_6:
                raise RuntimeError(
                    "`max_fpr` argument requires `torch.bucketize` which is not available below PyTorch version 1.6"
                )

        self.mode: DataType = None  # type: ignore
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AUROC` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        preds, target, mode = _auroc_update(preds, target)

        self.preds.append(preds)
        self.target.append(target)

        if self.mode and self.mode != mode:
            raise ValueError(
                "The mode of data (binary, multi-label, multi-class) should be constant, but changed"
                f" between batches from {self.mode} to {mode}"
            )
        self.mode = mode

    def compute(self) -> Tensor:
        """Computes AUROC based on inputs passed in to ``update`` previously."""
        if not self.mode:
            raise RuntimeError("You have to have determined mode.")
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _auroc_compute(
            preds,
            target,
            self.mode,
            self.num_classes,
            self.pos_label,
            self.average,
            self.max_fpr,
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

        rank_zero_warn(
            "Metric `PrecisionRecallCurve` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

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
