from app.lm.base_cls_lm import BaseClsLM
from torchvision.models import resnet50


class ResNetClsLM(BaseClsLM):
  def __init__(self, depth, num_classes, **kwargs):
    super().__init__(model_cfg=dict(
      type='Classification',
      backbone=dict(
        type='ResNet',
        depth=depth),
      head=dict(
        type='ClsHead',
        with_avg_pool=True,
        num_classes=num_classes)),
      )
    
    self.save_hyperparameters()

  @staticmethod
  def add_argparse_args(parent_parser):
    parent_parser = super(ResNetClsLM, ResNetClsLM) \
      .add_argparse_args(parent_parser)
    parser = parent_parser.add_argument_group("ResNetClsLM")
    parser.add_argument("--depth", type=int)
    return parent_parser
