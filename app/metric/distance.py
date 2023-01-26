from torchmetrics import Metric
from torch import tensor, sum, bmm
from torch.nn.functional import normalize, softmax, log_softmax


class Distance(Metric):
    full_state_update = False
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0), dist_reduce_fx="sum")

    def update(self, result_q, result_k):
        self.total = self.total.to(self.device)
        self.count = self.count.to(self.device)

        for q, k in zip(result_q, result_k):
            q, k = normalize(q, dim=1), normalize(k, dim=1)

            q = q.view(q.size(0), q.size(1), -1)  # [N,C,H*W]
            q = bmm(q.transpose(1, 2), q)  # [N,H*W,H*W]
            q = q.view(-1, q.size(2))  # [N*HW, H*W]

            k = k.view(k.size(0), k.size(1), -1)
            k = bmm(k.transpose(1, 2), k)
            k = k.view(-1, k.size(2))

            self.total += sum(
                -sum(
                    softmax(k / 0.2, dim=1) * log_softmax(q / 0.2, dim=1),
                    dim=1,
                )
            )
            self.count += q.size(0)

        return self

    def compute(self):
        return self.total / self.count
