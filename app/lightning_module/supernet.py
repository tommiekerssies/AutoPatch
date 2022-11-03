from torch.distributed import init_process_group
from app.metric.distance import Distance
from app.lightning_module.base import Base
from lib.gaia.dynamic_moco import DynamicMOCO
from lib.gaia.dynamic_nonlinear_neck import DynamicNonLinearNeckV1


class SuperNet(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--dense_distance", action="store_true")

    def __init__(self, dense_distance=None, **kwargs):
        self.save_hyperparameters()
        self.model_cfg = dict(
            type="mmselfsup.DynamicMOCO",
            queue_len=65536,
            feat_dim=128,
            momentum=0.999,
            neck=dict(
                type="mmselfsup.DynamicNonLinearNeckV1",
                in_channels=self.hparams.body_width[-1],
                hid_channels=2048,
                out_channels=128,
                with_avg_pool=True,
            ),
            head=dict(type="mmselfsup.ContrastiveHead", temperature=0.2),
        )
        super().__init__()

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
        # TODO: do forward on batch to be consistent with other lm's.
        return self.model(img)
