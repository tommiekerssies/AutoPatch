from os import environ
from torch.distributed import init_process_group
from app.metric.distance import Distance
from app.lightning_module.base import Base
from lib.gaia.dynamic_moco import DynamicMOCO
from lib.gaia.dynamic_nonlinear_neck import DynamicNonLinearNeckV1
from itertools import product
from typing import Any, List
from lib.gaia.flops_counter import get_model_complexity_info


class SuperNet(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--dense_distance", action="store_true")

        parser.add_argument("--stem_width_min", type=int)
        parser.add_argument("--stem_width_max", type=int)
        parser.add_argument("--stem_width_step", type=int)

        parser.add_argument("--body_width_min", nargs="+", type=int)
        parser.add_argument("--body_width_max", nargs="+", type=int)
        parser.add_argument("--body_width_step", nargs="+", type=int)

        parser.add_argument("--body_depth_min", nargs="+", type=int)
        parser.add_argument("--body_depth_max", nargs="+", type=int)
        parser.add_argument("--body_depth_step", nargs="+", type=int)

    def __init__(self, dense_distance=None, **kwargs):
        self.save_hyperparameters()

        super().__init__(
            dict(
                type="mmselfsup.DynamicMOCO",
                queue_len=65536,
                feat_dim=128,
                momentum=0.999,
                neck=dict(
                    type="mmselfsup.DynamicNonLinearNeckV1",
                    in_channels=4 * self.hparams.body_width[-1],
                    hid_channels=2048,
                    out_channels=128,
                    with_avg_pool=True,
                ),
                head=dict(type="mmselfsup.ContrastiveHead", temperature=0.2),
            ),
        )

        self.distance = Distance(dense_distance)

    def predict_step(self, batch, batch_idx):
        # TODO: add return?
        mode = "extract" if self.distance.dense_distance else "get_embedding"
        result_q = self(batch[0], mode=mode, extract_from="encoder_q")
        result_k = self(batch[0], mode=mode, extract_from="encoder_k")

        self.distance(result_q, result_k)

    def on_predict_start(self):
        self.distance.reset()

    def on_predict_end(self):
        self.distance.compute()

    def forward(self, img):
        return self.model(img)

    def get_complexities(self):
        num_stages = len(self.hparams.body_width_step)

        body_widths_per_stage = [
            range(
                self.hparams.body_width_min[i],
                self.hparams.body_width_max[i] + self.hparams.body_width_step[i],
                self.hparams.body_width_step[i],
            )
            for i in range(num_stages)
        ]

        body_depths_per_stage = [
            range(
                self.hparams.body_depth_min[i],
                self.hparams.body_depth_max[i] + self.hparams.body_depth_step[i],
                self.hparams.body_depth_step[i],
            )
            for i in range(num_stages)
        ]

        stem_widths = range(
            self.hparams.stem_width_min,
            self.hparams.stem_width_max + self.hparams.stem_width_step,
            self.hparams.stem_width_step,
        )

        architectures: List[Any] = [
            dict(
                arch=dict(
                    encoder_q=dict(
                        stem=dict(width=params[0]),
                        body=dict(
                            width=params[1 : num_stages + 1],
                            depth=params[num_stages + 1 :],
                        ),
                    ),
                ),
                overhead={},
            )
            for params in product(
                stem_widths, *body_widths_per_stage, *body_depths_per_stage
            )
        ]

        self.cuda()
        init_process_group(backend="nccl")
        for arch in architectures:
            self.model.manipulate_arch(arch["arch"])
            (  # TODO: make input shape a parameter
                arch["overhead"]["flops"],
                arch["overhead"]["params"],
            ) = get_model_complexity_info(
                self.model.backbone,
                (3, 224, 224),
                print_per_layer_stat=False,
                as_strings=False,
            )
            print(arch)
