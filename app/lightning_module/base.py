from typing import Any, Union
from pytorch_lightning import LightningModule
from os.path import join
import lib.gaia.dynamic_resnet
import lib.gaia.dynamic_conv
import lib.gaia.dynamic_bn
from mmcv.cnn import MODELS as MMCV_MODELS
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch import load


class Base(LightningModule):
    @classmethod
    def create(cls, resume_run_id, work_dir, project_name=None, **kwargs):
        ckpt_path = None

        if resume_run_id:
            if not project_name:
                raise ValueError("Must provide project name to resume run.")
            ckpt_path = join(
                work_dir, project_name, resume_run_id, "checkpoints", "last.ckpt"
            )
            lm = cls.load_from_checkpoint(ckpt_path, **kwargs)

        else:
            lm = cls(work_dir=work_dir, resume_run_id=None, **kwargs)  # type: ignore

        lm.ckpt_path = ckpt_path

        return lm

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weights_prefix", type=str, default="")

    def __init__(self):
        super().__init__()
        self.hparams: Any
        self.model_cfg: dict
        self.ckpt_path: Union[str, None] = None
        self.log_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True)

        if "backbone" not in self.model_cfg:
            self.model_cfg["backbone"] = {}

        self.model_cfg["backbone"] |= dict(
            type="mmselfsup.DynamicResNet",
            conv_cfg=dict(type="DynConv2d"),
            norm_cfg=dict(type="DynBN"),
            in_channels=3,
            stem_width=self.hparams.stem_width,
            body_width=self.hparams.body_width,
            body_depth=self.hparams.body_depth,
            dilations=self.hparams.dilations[:self.hparams.num_stages],
            strides=self.hparams.strides[:self.hparams.num_stages],
            out_indices=[0, 1, 2, 3][:self.hparams.num_stages],
            num_stages=self.hparams.num_stages,
            contract_dilation=True,
            block=self.block,
        )

        self.model = MMCV_MODELS.build(cfg=self.model_cfg)

        if not self.hparams.resume_run_id and self.hparams.weights_file:
            self.load_weights_from_file()

    def load_weights_from_file(self):
        obj = load(
            join(self.hparams.work_dir, self.hparams.weights_file),
            map_location=self.device,
        )

        if getattr(obj, "state_dict", None):
            state_dict = obj.state_dict()

        elif isinstance(obj, dict):
            state_dict = obj["state_dict"] if "state_dict" in obj else obj
        else:
            raise ValueError("Object must contain a state dict.")

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = self.hparams.model_prefix + key.replace(
                self.hparams.weights_prefix, ""
            )
            new_state_dict[new_key] = value

        return self.load_state_dict_verbose(new_state_dict)

    def load_state_dict_verbose(self, state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        imported_keys = state_dict.keys() - unexpected_keys

        # rank_zero_info(f"Imported keys: {imported_keys}")
        rank_zero_info(f"Missing keys: {missing_keys}")
        rank_zero_info(f"Unexpected keys: {unexpected_keys}")

        if len(imported_keys) == 0:
            raise ValueError("No keys were imported.")

        return self
