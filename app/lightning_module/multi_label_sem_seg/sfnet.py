from app.lightning_module.multi_label_sem_seg_subnet.base import Base
import lib.sf_neck


class SFNet(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--sf_neck_channels", type=int, default=128)
        parser.add_argument("--aux_weight", type=float, default=1.0)

    def __init__(self, **kwargs):
        self.save_hyperparameters()
        head_cfg = dict(
            in_channels=self.hparams.sf_neck_channels,
            channels=self.hparams.sf_neck_channels,
            num_convs=1,
            kernel_size=3,
        )
        self.model_cfg = dict(
            type="mmseg.EncoderDecoder",
            neck=dict(
                type="SFNeck",
                in_channels=self.hparams.body_width,
                channels=self.hparams.sf_neck_channels,
                norm_cfg=self.head_cfg["norm_cfg"],
            ),
            decode_head=dict(
                in_index=4,
                **head_cfg,
                **self.head_cfg,
            ),
            auxiliary_head=[
                dict(
                    in_index=1,
                    **head_cfg,
                    **self.head_cfg,
                ),
                dict(
                    in_index=2,
                    **head_cfg,
                    **self.head_cfg,
                ),
                dict(
                    in_index=3,
                    **head_cfg,
                    **self.head_cfg,
                ),
            ],
        )
        super().__init__()
