from mmselfsup.models import build_algorithm
from app.lm.base_cls_lm import BaseClsLM


class ResNetClsLM(BaseClsLM):
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)
    
    model_cfg = dict(
      type='Classification',
      backbone=dict(
        type='ResNet',
        depth=self.hparams.depth,
        out_indices=[4],  # 4: stage-4
        norm_cfg=dict(type='BN')),
      head=dict(
        type='ClsHead', with_avg_pool=True, 
        in_channels=2048, num_classes=self.hparams.num_classes))
    
    self.model = build_algorithm(model_cfg)

  @staticmethod
  def add_argparse_args(parent_parser):
    parent_parser = super(ResNetClsLM, ResNetClsLM) \
      .add_argparse_args(parent_parser)
    parser = parent_parser.add_argument_group("DynamicClassifierLM")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--num_classes", type=int)
    return parent_parser