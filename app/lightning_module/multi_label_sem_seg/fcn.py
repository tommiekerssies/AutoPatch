from app.lightning_module.multi_label_sem_seg.base import Base


class FCN(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--aux_weight", type=float, default=0.4)

    def __init__(self, **kwargs):
        self.save_hyperparameters()
        head = dict(
            type="FCNHead",
            num_convs=0,
            num_classes=self.hparams.num_classes,
            norm_cfg=self.decoder_norm_cfg,
            concat_input=False,
            threshold=0.0,  # not used but here to prevent log warning
        )
        self.model_cfg = dict(
            type="mmseg.EncoderDecoder",
            init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnet18v1c"),
            decode_head=dict(
                in_channels=self.hparams.body_width[-1],
                channels=self.hparams.body_width[-1],
                **head,
            ),
            auxiliary_head=[
                dict(
                    in_channels=self.hparams.body_width[-2],
                    channels=self.hparams.body_width[-2],
                    in_index=-2,
                    **head,
                )
            ],
        )
        super().__init__()
