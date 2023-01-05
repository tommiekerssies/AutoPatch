from app.metric.distance import Distance
from app.lightning_module.base import Base
from lib.gaia.dynamic_moco import DynamicMOCO
from lib.gaia.dynamic_nonlinear_neck import DynamicNonLinearNeckV1
from lib.gaia.dynamic_res_blocks import DynamicBottleneck


class SuperNet(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--dilations", nargs="+", type=int, default=[1, 1, 1, 1])
        parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 2, 2])
        parser.add_argument("--body_width", nargs="+", type=int, default=[80, 160, 320, 640])
        parser.add_argument("--body_depth", nargs="+", type=int, default=[4, 6, 29, 4])
        parser.add_argument("--model_prefix", type=str, default="model.")
        parser.add_argument("--weights_file", type=str, default="epoch_200.pth")
        parser.add_argument("--out_indices", nargs="+", type=int, default=[0, 1, 2])

    def __init__(self, dense_distance=None, **kwargs):
        self.save_hyperparameters()
        self.model_cfg = dict(
            type="mmselfsup.DynamicMOCO",
            queue_len=65536,
            feat_dim=128,
            momentum=0.999,
            neck=dict(
                type="mmselfsup.DynamicNonLinearNeckV1",
                in_channels=self.hparams.body_width[-1] * 4,
                hid_channels=2048,
                out_channels=128,
                with_avg_pool=True,
            ),
            head=dict(type="mmselfsup.ContrastiveHead", temperature=0.2),
            backbone=dict(
                block=DynamicBottleneck,
            )
        )
        super().__init__()
        self.distance = Distance()

    def predict_step(self, batch, batch_idx):
        # TODO: add return?
        result_q = self.model.encoder_q[0](batch["image"].float())
        result_k = self.model.encoder_k[0](batch["image"].float())
        self.distance(result_q, result_k)

    def on_predict_start(self):
        self.distance.reset()

    def on_predict_end(self):
        self.distance.compute()