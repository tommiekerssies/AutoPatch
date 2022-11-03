from torch import mean, stack, tensor
from torch.nn import ModuleList
from torch.nn.functional import binary_cross_entropy_with_logits
from app.metric.iou import IoUMetric
from app.metric.mean import MeanMetric
from app.lightning_module.base import Base as BaseLM
from torch.optim import Adam
from mmseg.ops import resize


class Base(BaseLM):
    @staticmethod
    def add_argparse_args(parser):
        BaseLM.add_argparse_args(parser)
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--align_corners", action="store_true")
        parser.add_argument("--monitor", type=str, default="val_mIoU")
        parser.add_argument("--monitor_mode", type=str, default="max")

    def __init__(self):
        super().__init__()

        self.train_IoUs = ModuleList()
        for _ in range(self.hparams.num_classes):
            self.train_IoUs.append(IoUMetric())
        self.train_mIoU = MeanMetric(self.train_IoUs)

        self.val_IoUs = ModuleList()
        for _ in range(self.hparams.num_classes):
            self.val_IoUs.append(IoUMetric())
        self.val_mIoU = MeanMetric(self.val_IoUs)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)

        return dict(
            optimizer=optimizer,
        )

    def training_step(self, batch, batch_idx):
        return self.step(batch, self.train_IoUs, self.train_mIoU, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, self.val_IoUs, self.val_mIoU, "val")

    def step(self, batch, metrics, mean_metric, prefix):
        out, loss = self(batch)

        bs = batch["image"].shape[0]

        for i in range(self.hparams.num_classes):
            metrics[i](out[:, i, :, :], batch["masks"][i], batch["ignore_mask"])
            self.log(f"{prefix}_IoU_{i}", metrics[i], batch_size=bs, **self.log_kwargs)
        self.log(f"{prefix}_mIoU", mean_metric, batch_size=bs, **self.log_kwargs)

        self.log(f"{prefix}_loss", loss, batch_size=bs, **self.log_kwargs)

        return loss

    def forward(self, batch):
        x = self.model.extract_feat(batch["image"].float())

        img_size = batch["image"].shape[2:]
        resize_mode = "bilinear"
        out = resize(
            input=self.model.decode_head(x),
            size=img_size,
            mode=resize_mode,
            align_corners=self.hparams.align_corners,
        )

        if self.training:
            aux_outs = [
                resize(
                    input=aux_head(x),
                    size=img_size,
                    mode=resize_mode,
                    align_corners=self.hparams.align_corners,
                )
                for aux_head in self.model.auxiliary_head
            ]

        class_losses = []
        for i in range(self.hparams.num_classes):
            class_loss = self._get_class_loss(
                out[:, i, :, :], batch["masks"][i], batch["ignore_mask"]
            )

            if self.training:
                for aux_out in aux_outs:  # type: ignore
                    class_loss += self.hparams.aux_weight * self._get_class_loss(
                        aux_out[:, i, :, :], batch["masks"][i], batch["ignore_mask"]
                    )

            class_losses.append(class_loss)

        loss = mean(stack(class_losses))

        return out, loss

    def _get_class_loss(self, out, class_mask, ignore_mask):
        loss = binary_cross_entropy_with_logits(
            out, class_mask.float(), reduction="none"
        )

        non_ignore_mask = ignore_mask == 0
        loss = loss.where(non_ignore_mask, tensor(0.0, device=self.device))

        return loss.sum() / loss.count_nonzero()

    @property
    def decoder_norm_cfg(self):
        if self.hparams.sync_bn:
            return dict(type="SyncBN", requires_grad=True)
        else:
            return None
