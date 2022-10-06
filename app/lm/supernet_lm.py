from pytorch_lightning import LightningModule
from mmselfsup.models import build_algorithm
from app.lm.base_lm import BaseLM
import lib.gaia.dynamic_moco
import lib.gaia.dynamic_resnet
import lib.gaia.dynamic_conv
import lib.gaia.dynamic_bn
import lib.gaia.dynamic_nonlinear_neck
from torchmetrics import Metric
from torch import tensor, sum, einsum, bmm
from torch.nn.functional import normalize


class Distance(Metric):
  full_state_update = False
  higher_is_better = False

  def __init__(self, dense_distance):
    self.dense_distance = dense_distance

    self.add_state("distance", default=tensor(0.), dist_reduce_fx="sum")
    self.add_state("count", default=tensor(0), dist_reduce_fx="sum")

  def update(self, q, k):
    self.distance = self.distance.to(self.device)
    self.count = self.count.to(self.device)

    q, k = q * 100, k * 100  # scale up for avoiding overflow
    q, k = normalize(q, dim=1), normalize(k, dim=1)

    # For dense, resolution of tensor from early stage may cause memory out. Consider cropping it.
    if self.dense_distance:
      q = q.view(q.size(0), q.size(1), -1)  # [N,C,H*W]
      q = bmm(q.transpose(1, 2), q)  # [N,H*W,H*W]
      q = q.view(-1, q.size(2))  # [N*HW, H*W]

      k = k.view(k.size(0), k.size(1), -1)
      k = bmm(k.transpose(1, 2), k)
      k = k.view(-1, k.size(2))

    self.distance += sum(einsum('nc,nc->n', [q, k]))
    self.count += q.size(0)

    return self

  def compute(self):
    return self.distance / self.count


class SuperNetLM(BaseLM):
  def __init__(self, **kwargs):
    # TODO make this generic such that it can be MoCo or DenseCL
    super().__init__(
        model_cfg=dict(
            type='DynamicMOCO',
            queue_len=65536,
            feat_dim=128,
            momentum=0.999,
            backbone=dict(
                type='DynamicResNet',
                in_channels=3,
                stem_width=64,
                body_depth=[4, 6, 29, 4],
                body_width=[80, 160, 320, 640],
                num_stages=4,
                out_indices=[3],  # 0: conv-1, x: stage-x
                conv_cfg=dict(type='DynConv2d'),
                norm_cfg=dict(type='DynBN', requires_grad=True),
                style='pytorch',),
            neck=dict(
                type='DynamicNonLinearNeckV1',
                in_channels=2560,
                hid_channels=2048,
                out_channels=128,
                with_avg_pool=True),
            head=dict(type='ContrastiveHead', temperature=0.2)),
        **kwargs)

    self.distance = Distance(self.hparams.dense_distance)

  def predict_step(self, batch, batch_idx):
    # TODO: add return?
    if self.distance.dense_distance:
      mode = 'extract'
    else:
      mode = 'get_embedding'

    result_q = self.model(batch[0], mode=mode)
    result_k = self.model(batch[0], mode=mode, extract_from='encoder_k')

    self.distance(result_q, result_k)

  def on_predict_start(self):
    self.distance.reset()

  def on_predict_end(self):
    self.distance.compute()

  def forward(self, x):
    return self.model(x)

  @staticmethod
  def add_argparse_args(parent_parser):
    parent_parser = super(SuperNetLM, SuperNetLM) \
        .add_argparse_args(parent_parser)
    parser = parent_parser.add_argument_group("SuperNetLM")
    parser.add_argument("--dense_distance", action='store_true')
    return parent_parser
