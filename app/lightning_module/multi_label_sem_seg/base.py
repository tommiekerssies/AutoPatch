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
        parser.add_argument("--monitor", type=str, default="val_IoU_0")
        parser.add_argument("--monitor_mode", type=str, default="max")

    def __init__(self):
        super().__init__()

        self.train_IoUs = ModuleList()
        for _ in range(self.hparams.num_classes):
            self.train_IoUs.append(IoUMetric())

        self.train_mIoU = None
        if len(self.train_IoUs) > 1:
            self.train_mIoU = MeanMetric(self.train_IoUs)

        self.val_IoUs = ModuleList()
        for _ in range(self.hparams.num_classes):
            self.val_IoUs.append(IoUMetric())

        self.val_mIoU = None
        if len(self.val_IoUs) > 1:
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
        batch_size = batch["image"].shape[0]
        out, aux_outs = self(batch)

        for i in range(self.hparams.num_classes):
            metrics[i](out[:, i, :, :], batch["masks"][i], batch["ignore_mask"])
            self.log(
                f"{prefix}_IoU_{i}",
                metrics[i],
                batch_size=batch_size,
                **self.log_kwargs,
            )

        if mean_metric:
            self.log(
                f"{prefix}_mIoU", mean_metric, batch_size=batch_size, **self.log_kwargs
            )

        loss = self.loss(out, batch)
        self.log(f"{prefix}_loss", loss, batch_size=batch_size, **self.log_kwargs)

        for i, aux_out in enumerate(aux_outs):
            aux_loss = self.loss(aux_out, batch)
            self.log(
                f"{prefix}_aux_loss_{i}",
                aux_loss,
                batch_size=batch_size,
                **self.log_kwargs,
            )
            loss += self.hparams.aux_weight * self.loss(aux_out, batch)

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

        aux_outs = []
        if self.training and hasattr(self.model, "auxiliary_head"):
            aux_outs.extend(
                resize(
                    input=aux_head(x),
                    size=img_size,
                    mode=resize_mode,
                    align_corners=self.hparams.align_corners,
                )
                for aux_head in self.model.auxiliary_head
            )

        return out, aux_outs

    def loss(self, out, batch):
        class_losses = [
            self._get_class_loss(
                out[:, i, :, :], batch["masks"][i], batch["ignore_mask"]
            )
            for i in range(self.hparams.num_classes)
        ]

        return mean(stack(class_losses))

    def _get_class_loss(self, out, class_mask, ignore_mask):
        loss = binary_cross_entropy_with_logits(
            out, class_mask.float(), reduction="none"
        )

        non_ignore_mask = ignore_mask == 0
        loss_masked = loss.where(non_ignore_mask, tensor(0.0, device=self.device))

        return loss_masked.sum() / non_ignore_mask.sum()

    @property
    def head_cfg(self):
        return dict(
            type="FCNHead",
            num_classes=self.hparams.num_classes,
            norm_cfg=dict(type="SyncBN"),
            concat_input=False,
            threshold=0.0,  # not used, but here to prevent log warning
        )
