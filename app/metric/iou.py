from torch import tensor
from torchmetrics import Metric


class IoUMetric(Metric):
    full_state_update = False
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state(
            "intersection",
            default=tensor(0, device=self.device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "union", default=tensor(0, device=self.device), dist_reduce_fx="sum"
        )

    def update(self, class_output, class_mask, ignore_mask):
        preds = class_output > 0
        non_ignore_mask = ignore_mask == 0
        non_ignore_mask = non_ignore_mask.to(self.device)

        intersection = preds.logical_and(class_mask).to(self.device)
        intersection_masked = intersection.logical_and(non_ignore_mask).to(self.device)
        self.intersection += intersection_masked.sum()

        union = preds.logical_or(class_mask).to(self.device)
        union_masked = union.logical_and(non_ignore_mask).to(self.device)
        self.union += union_masked.sum()

        return self

    def compute(self):
        return self.intersection.float() / self.union  # type: ignore
