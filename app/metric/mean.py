from torchmetrics import Metric
from torch import nanmean, stack


class MeanMetric(Metric):
    full_state_update = False

    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def update(self):
        return self

    def compute(self):
        values = [metric.compute() for metric in self.metrics]
        return nanmean(stack(values))
