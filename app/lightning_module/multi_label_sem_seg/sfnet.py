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
        self.model_cfg = dict(
            type="mmseg.EncoderDecoder",
            neck=dict(
                type="SFNeck",
                in_channels=self.hparams.body_width,
                channels=self.hparams.sf_neck_channels,
                norm_cfg=self.decoder_norm_cfg,
                align_corners=self.hparams.align_corners,
            ),
            decode_head=dict(
                type="FCNHead",
                in_channels=self.hparams.sf_neck_channels,
                in_index=4,
                channels=self.hparams.sf_neck_channels,
                num_convs=1,
                kernel_size=3,
                concat_input=False,
                num_classes=self.hparams.num_classes,
                norm_cfg=self.decoder_norm_cfg,
            ),
            auxiliary_head=[
                dict(
                    type="FCNHead",
                    in_channels=self.hparams.sf_neck_channels,
                    in_index=1,
                    channels=self.hparams.sf_neck_channels,
                    num_convs=1,
                    kernel_size=3,
                    concat_input=False,
                    num_classes=self.hparams.num_classes,
                    norm_cfg=self.decoder_norm_cfg,
                ),
                dict(
                    type="FCNHead",
                    in_channels=self.hparams.sf_neck_channels,
                    in_index=2,
                    channels=self.hparams.sf_neck_channels,
                    num_convs=1,
                    kernel_size=3,
                    concat_input=False,
                    num_classes=self.hparams.num_classes,
                    norm_cfg=self.decoder_norm_cfg,
                ),
                dict(
                    type="FCNHead",
                    in_channels=self.hparams.sf_neck_channels,
                    in_index=3,
                    channels=self.hparams.sf_neck_channels,
                    num_convs=1,
                    kernel_size=3,
                    concat_input=False,
                    num_classes=self.hparams.num_classes,
                    norm_cfg=self.decoder_norm_cfg,
                ),
            ],
        )
        super().__init__()
