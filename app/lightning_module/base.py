from typing import Any, Union
from pytorch_lightning import LightningModule
from os.path import join
import lib.gaia.dynamic_resnet
import lib.gaia.dynamic_conv
import lib.gaia.dynamic_sync_bn
from mmcv.cnn import MODELS as MMCV_MODELS


class Base(LightningModule):
    @classmethod
    def create(cls, ldm, resume_run_id, work_dir, project_name, **kwargs):
        ckpt_path = None

        if resume_run_id:
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
        parser.add_argument("--lr", type=float)
        parser.add_argument("--warmup_steps", type=int, default=0)
        parser.add_argument("--stem_width", type=int)
        parser.add_argument("--body_width", nargs="+", type=int)
        parser.add_argument("--body_depth", nargs="+", type=int)

    def __init__(self, model_cfg):
        super().__init__()
        self.hparams: Any
        self.ckpt_path: Union[str, None] = None

        if "backbone" not in model_cfg:
            model_cfg["backbone"] = {}

        model_cfg["backbone"] |= dict(
            type="mmselfsup.DynamicResNet",
            in_channels=3,
            stem_width=self.hparams.stem_width,
            body_depth=self.hparams.body_depth,
            body_width=self.hparams.body_width,
            conv_cfg=dict(type="DynConv2d"),
            dilations=(1, 1, 2, 2),
            strides=(1, 2, 1, 1),
        )

        self.model = MMCV_MODELS.build(cfg=model_cfg)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.step(closure=optimizer_closure)

        # learning rate warm-up
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            for i, pg in enumerate(optimizer.param_groups):  # type: ignore
                pg["lr"] = lr_scale * self.hparams.lr
                print(
                    f"Step {self.trainer.global_step}: increasing learning rate"
                    f" of group {i} to {pg['lr']}."
                )
