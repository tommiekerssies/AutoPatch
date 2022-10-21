from torchmetrics import Metric
from torch import tensor, sum, einsum, bmm
from torch.nn.functional import normalize


class Distance(Metric):
    full_state_update = False
    higher_is_better = False

    def __init__(self, dense_distance):
        super().__init__()
        self.dense_distance = dense_distance

        self.add_state("distance", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0), dist_reduce_fx="sum")

    def update(self, q, k):
        self.distance = self.distance.to(self.device)
        self.count = self.count.to(self.device)

        q, k = q * 100, k * 100  # scale up for avoiding overflow
        q, k = normalize(q, dim=1), normalize(k, dim=1)

        # For dense, resolution of tensor from early stage may cause memory out.
        # Consider cropping it.
        if self.dense_distance:
            q = q.view(q.size(0), q.size(1), -1)  # [N,C,H*W]
            q = bmm(q.transpose(1, 2), q)  # [N,H*W,H*W]
            q = q.view(-1, q.size(2))  # [N*HW, H*W]

            k = k.view(k.size(0), k.size(1), -1)
            k = bmm(k.transpose(1, 2), k)
            k = k.view(-1, k.size(2))

        self.distance += sum(einsum("nc,nc->n", [q, k]))
        self.count += q.size(0)

        return self

    def compute(self):
        return self.distance / self.count
