from os import environ
from pytorch_lightning import Trainer
from app.lm.base_cls_lm import BaseClsLM
from app.lm.supernet_lm import SuperNetLM
import lib.gaia.dynamic_resnet
import lib.gaia.dynamic_conv
import lib.gaia.dynamic_bn


class SubnetClsLM(BaseClsLM):
  def __init__(self, stem_width, body_depth, 
               body_width, num_classes, **kwargs):
    super().__init__(
        model_cfg=dict(
            type='Classification',
            backbone=dict(
                type='DynamicResNet',
                in_channels=3,
                stem_width=stem_width,
                body_depth=body_depth,
                body_width=body_width,
                num_stages=4,
                out_indices=[3],  # 0: conv-1, x: stage-x
                conv_cfg=dict(type='DynConv2d'),
                norm_cfg=dict(type='DynBN', requires_grad=True),
                style='pytorch',),
            head=dict(type='ClsHead', with_avg_pool=True,
                      in_channels=4 * body_width[-1],
                      num_classes=num_classes)))
    
    self.save_hyperparameters()
    
  def load_weights(self, obj, ldm, prefix_old, prefix_new, **kwargs):
    supernet_lm = SuperNetLM().load_state_dict_from_obj(obj,
                                                        prefix_old, 
                                                        prefix_new)

    # Doing a forward pass with supernet in deploy mode should
    # remove unused layers and channels from its state dict.
    supernet_lm.model.deploy()
    supernet_lm.model.manipulate_arch(dict(
        encoder_q=dict(
          stem=dict(width=self.hparams.stem_width),
          body=dict(width=self.hparams.body_width, 
                    depth=self.hparams.body_depth))))
    
    # to prevent error when starting training later set this env
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    
    Trainer(strategy='ddp', accelerator='auto', fast_dev_run=True)\
      .predict(supernet_lm, datamodule=ldm)
    
    return self.load_state_dict_from_obj(supernet_lm)

  @staticmethod
  def add_argparse_args(parent_parser):
    parent_parser = super(SubnetClsLM, SubnetClsLM) \
      .add_argparse_args(parent_parser)
    parser = parent_parser.add_argument_group("SubnetClsLM")
    parser.add_argument("--stem_width", type=int)
    parser.add_argument("--body_width", nargs="+", type=int)
    parser.add_argument("--body_depth", nargs="+", type=int)
    return parent_parser
