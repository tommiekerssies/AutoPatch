from typing import Any, Union
from pytorch_lightning import LightningModule
from os.path import join
import lib.gaia.dynamic_resnet
import lib.gaia.dynamic_conv
import lib.gaia.dynamic_sync_bn
from mmcv.cnn import MODELS as MMCV_MODELS


class Base(LightningModule):
    @classmethod
    def create(cls, resume_run_id, work_dir, project_name=None, ldm=None, **kwargs):
        ckpt_path = None

        if resume_run_id and project_name:
            ckpt_path = join(
                work_dir, project_name, resume_run_id, "checkpoints", "last.ckpt"
            )
            lm = cls.load_from_checkpoint(ckpt_path, **kwargs)

        else:
            lm = cls(ldm=ldm, resume_run_id=resume_run_id, work_dir=work_dir, **kwargs)  # type: ignore

        lm.ckpt_path = ckpt_path

        return lm

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--stem_width", type=int)
        parser.add_argument("--body_width", nargs="+", type=int)
        parser.add_argument("--body_depth", nargs="+", type=int)

    def __init__(self, model_cfg):
        super().__init__()
        self.hparams: Any
        self.ckpt_path: Union[str, None] = None
        self.log_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True)

        if "backbone" not in model_cfg:
            model_cfg["backbone"] = {}

        num_stages = len(self.hparams.body_depth)
        dilations = (1, 1, 1, 1)
        strides = (1, 2, 2, 2)

        model_cfg["backbone"] |= dict(
            type="mmselfsup.DynamicResNet",
            conv_cfg=dict(type="DynConv2d"),
            contract_dilation=True,
            in_channels=3,
            stem_width=self.hparams.stem_width,
            body_depth=self.hparams.body_depth,
            body_width=self.hparams.body_width,
            num_stages=num_stages,
            dilations=dilations[:num_stages],
            strides=strides[:num_stages],
            out_indices=[num_stages - 1],
        )

        self.model = MMCV_MODELS.build(cfg=model_cfg)
