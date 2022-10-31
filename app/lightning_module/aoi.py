from os import environ
from pytorch_lightning import Trainer
from torch import mean, stack, tensor
from torch.nn import ModuleList
from torch.nn.functional import binary_cross_entropy_with_logits
from app.lightning_module.supernet import SuperNet
from app.metric.iou import IoUMetric
from app.metric.mean import MeanMetric
from app.lightning_module.base import Base
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import mmseg.models.segmentors.encoder_decoder
from mmseg.ops import resize


class AOI(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--supernet_run_id", type=str)
        parser.add_argument("--num_classes", type=int)

    def __init__(self, ldm=None, **kwargs):
        self.save_hyperparameters(ignore="ldm")

        super().__init__(
            dict(
                type="mmseg.EncoderDecoder",
                decode_head=dict(
                    type="FCNHead",
                    num_convs=0,
                    concat_input=False,
                    channels=4 * self.hparams.body_width[-1],
                    in_channels=4 * self.hparams.body_width[-1],
                    num_classes=self.hparams.num_classes,
                ),
            )
        )

        if self.hparams.supernet_run_id and not self.hparams.resume_run_id:
            self.load_weights_from_supernet(ldm)

        self.train_IoUs = ModuleList()
        for _ in range(self.hparams.num_classes):
            self.train_IoUs.append(IoUMetric())
        self.train_mIoU = MeanMetric(self.train_IoUs)

        self.val_IoUs = ModuleList()
        for _ in range(self.hparams.num_classes):
            self.val_IoUs.append(IoUMetric())
        self.val_mIoU = MeanMetric(self.val_IoUs)

    def load_weights_from_supernet(self, ldm):
        # TODO: remove requirement of ldm

        supernet_lm = SuperNet.create(
            resume_run_id=self.hparams.supernet_run_id, **self.hparams
        )

        # Doing a forward pass with supernet in deploy mode should
        # remove unused layers and channels from its state dict.
        supernet_lm.model.deploy()
        supernet_lm.model.manipulate_arch(
            dict(
                encoder_q=dict(
                    stem=dict(width=self.hparams.stem_width),
                    body=dict(
                        width=self.hparams.body_width,
                        depth=self.hparams.body_depth,
                    ),
                )
            )
        )

        # to prevent error when starting training later set this env
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        Trainer(strategy="ddp", accelerator="auto", fast_dev_run=True).predict(
            supernet_lm, datamodule=ldm
        )

        return self.load_state_dict_from_obj(supernet_lm)

    def load_state_dict_from_obj(self, obj):
        if getattr(obj, "state_dict", None):
            state_dict = obj.state_dict()

        elif isinstance(obj, dict):
            state_dict = obj["state_dict"] if "state_dict" in obj else obj
        else:
            raise ValueError("Object must contain a state dict.")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        imported_keys = state_dict.keys() - unexpected_keys

        rank_zero_info(f"Imported keys: {imported_keys}")
        rank_zero_info(f"Missing keys: {missing_keys}")
        rank_zero_info(f"Unexpected keys: {unexpected_keys}")

        if len(imported_keys) == 0:
            raise ValueError("No keys were imported.")

        return self

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)

        return dict(
            optimizer=optimizer,
        )

    def forward(self, img):
        x = self.model.extract_feat(img)
        out = self.model.decode_head(x)

        return resize(
            input=out,
            size=img.shape[2:],
            mode="bilinear",
        )

    def step(self, batch, metrics, mean_metric, prefix):
        out = self(batch["image"].float())

        for i in range(self.hparams.num_classes):
            metrics[i](out[:, i, :, :], batch["masks"][i], batch["ignore_mask"])
            self.log(f"{prefix}_IoU_{i}", metrics[i], **self.log_kwargs)

        self.log(f"{prefix}_mIoU", mean_metric, **self.log_kwargs)

        loss = self._get_loss(out, batch)
        self.log(f"{prefix}_loss", loss, **self.log_kwargs)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, self.train_IoUs, self.train_mIoU, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, self.val_IoUs, self.val_mIoU, "val")

    def _get_loss(
        self, out, batch
    ):  # sourcery skip: for-append-to-extend, list-comprehension
        class_losses = []
        for i in range(self.hparams.num_classes):
            class_losses.append(
                self._get_class_loss(
                    out[:, i, :, :], batch["masks"][i], batch["ignore_mask"]
                )
            )

        return mean(stack(class_losses))

    def _get_class_loss(self, out, class_mask, ignore_mask):
        loss = binary_cross_entropy_with_logits(
            out, class_mask.float(), reduction="none"
        )
        non_ignore_mask = ignore_mask == 0
        loss_masked = loss.where(non_ignore_mask, tensor(0.0, device=self.device))

        return loss_masked.sum() / non_ignore_mask.sum()
