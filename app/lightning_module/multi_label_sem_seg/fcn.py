from app.lightning_module.multi_label_sem_seg.base import Base


class FCN(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--aux_weight", type=float, default=0.4)

    def __init__(self, **kwargs):
        self.save_hyperparameters()
        self.model_cfg = dict(
            type="mmseg.EncoderDecoder",
            decode_head=dict(
                in_channels=self.hparams.body_width[-1],
                channels=self.hparams.body_width[-1],
                num_convs=0,
                **self.head_cfg,
            ),
            auxiliary_head=[
                dict(
                    in_channels=self.hparams.body_width[-2],
                    channels=self.hparams.body_width[-2],
                    in_index=-2,
                    num_convs=0,
                    **self.head_cfg,
                )
            ],
        )
        super().__init__()
