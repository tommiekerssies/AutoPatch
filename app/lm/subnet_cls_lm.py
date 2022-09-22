from pytorch_lightning import LightningModule
from mmselfsup.models import build_algorithm
from app.lm.base_cls_lm import BaseClsLM
import lib.gaia.dynamic_resnet
import lib.gaia.dynamic_conv
import lib.gaia.dynamic_bn
from torch.nn.functional import cross_entropy


class SubnetClsLM(BaseClsLM):
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)
    
    self.model_cfg = dict(
      type='Classification',
      backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=self.hparams.stem_width,
        body_depth=self.hparams.body_depth,
        body_width=self.hparams.body_width,
        num_stages=4,
        out_indices=[3],  # 0: conv-1, x: stage-x
        conv_cfg=dict(type='DynConv2d'),
        norm_cfg=dict(type='DynBN', requires_grad=True),
        style='pytorch',),
      head=dict(type='ClsHead', with_avg_pool=True, 
                in_channels=4 * self.hparams.body_width[-1],
                num_classes=self.hparams.num_classes))
    
    self.model = build_algorithm(self.model_cfg)
    
  @staticmethod
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("DynamicClassifierLM")
    parser.add_argument("--stem_width", type=int)
    parser.add_argument("--body_width", nargs="+", type=int)
    parser.add_argument("--body_depth", nargs="+", type=int)
    parser.add_argument("--num_classes", type=int)
    return parent_parser