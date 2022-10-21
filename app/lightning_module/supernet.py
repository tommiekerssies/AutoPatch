from app.metric.distance import Distance
from app.lightning_module.base import Base


class SuperNet(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--dense_distance", action="store_true")

    def __init__(self, dense_distance=None, **kwargs):
        self.save_hyperparameters()

        super().__init__(
            dict(
                type="DynamicMOCO",
                queue_len=65536,
                feat_dim=128,
                momentum=0.999,
                neck=dict(
                    type="DynamicNonLinearNeckV1",
                    in_channels=2560,
                    hid_channels=2048,
                    out_channels=128,
                    with_avg_pool=True,
                ),
                head=dict(type="ContrastiveHead", temperature=0.2),
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
