from app.lightning_module.base import Base
from lib.gaia.dynamic_moco import DynamicMOCO
from lib.gaia.dynamic_nonlinear_neck import DynamicNonLinearNeckV1
from lib.gaia.dynamic_res_blocks import DynamicBottleneck
from itertools import product
from lib.gaia.flops_counter import (
    flops_to_string,
    get_model_complexity_info,
    params_to_string,
)


class SuperNet(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--dilations", nargs="+", type=int, default=[1, 1, 1, 1])
        parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 2, 2])
        parser.add_argument("--stem_width", type=int, default=64)
        parser.add_argument(
            "--body_width", nargs="+", type=int, default=[80, 160, 320, 640]
        )
        parser.add_argument("--body_depth", nargs="+", type=int, default=[4, 6, 29, 4])
        parser.add_argument("--model_prefix", type=str, default="model.")
        parser.add_argument("--weights_file", type=str, default="epoch_200.pth")
        parser.add_argument("--out_layers", nargs="+", type=int, default=[1])
        parser.add_argument("--stem_widths", nargs="+", type=int, default=[32, 48, 64])
        parser.add_argument(
            "--body_width_min", nargs="+", type=int, default=[48, 96, 192]
        )
        parser.add_argument(
            "--body_width_max", nargs="+", type=int, default=[80, 160, 320]
        )
        parser.add_argument(
            "--body_width_step", nargs="+", type=int, default=[16, 32, 64]
        )
        parser.add_argument("--body_depth_min", nargs="+", type=int, default=[2, 2, 5])
        parser.add_argument("--body_depth_max", nargs="+", type=int, default=[4, 6, 29])
        parser.add_argument("--body_depth_step", nargs="+", type=int, default=[1, 2, 2])
        parser.add_argument("--num_stages", type=int, default=2)

    def __init__(self, **kwargs):
        self.save_hyperparameters()
        self.block = DynamicBottleneck
        self.model_cfg = dict(
            type="mmselfsup.DynamicMOCO",
            queue_len=65536,
            feat_dim=128,
            momentum=0.999,
            neck=dict(
                type="mmselfsup.DynamicNonLinearNeckV1",
                in_channels=self.hparams.body_width[-1] * self.block.expansion,
                hid_channels=2048,
                out_channels=128,
                with_avg_pool=True,
            ),
            head=dict(type="mmselfsup.ContrastiveHead", temperature=0.2),
        )
        super().__init__()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outs_q = self.model.encoder_q[0](batch["img"])
        return [outs_q[i] for i in self.hparams.out_layers]

    def get_subnets(self):
        body_widths = [
            range(
                self.hparams.body_width_min[i],
                self.hparams.body_width_max[i] + self.hparams.body_width_step[i],
                self.hparams.body_width_step[i],
            )
            if self.hparams.body_width_step[i] > 0
            else [self.hparams.body_width_min[i]]
            for i in range(self.hparams.num_stages)
        ]

        body_depths = [
            range(
                self.hparams.body_depth_min[i],
                self.hparams.body_depth_max[i] + self.hparams.body_depth_step[i],
                self.hparams.body_depth_step[i],
            )
            for i in range(self.hparams.num_stages)
        ]

        return [
            dict(
                arch=dict(
                    encoder_q=dict(
                        stem=dict(width=params[0]),
                        body=dict(
                            width=list(params[1 : self.hparams.num_stages + 1]),
                            depth=list(params[self.hparams.num_stages + 1 :]),
                        ),
                    ),
                ),
                overhead={},
                overhead_as_strings={},
            )
            for params in product(self.hparams.stem_widths, *body_widths, *body_depths)
        ]

    def get_backbone_complexity(self):
        flops, params = get_model_complexity_info(
            self.model.backbone,
            (3, self.hparams.img_size, self.hparams.img_size),
            print_per_layer_stat=False,
            as_strings=False,
        )
        return flops, params, flops_to_string(flops), params_to_string(params)
