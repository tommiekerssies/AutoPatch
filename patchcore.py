import faiss.contrib.torch_utils
from faiss import IndexFlatL2, StandardGpuResources, index_cpu_to_gpu
from timeit import default_timer
from pytorch_lightning import LightningModule
from torch import (
    Tensor,
    no_grad,
    argmax,
    eq,
    flatten,
    mean,
    ones_like,
    cat,
    where,
    stack,
    max as torch_max,
)
from torch.nn.functional import adaptive_avg_pool1d, interpolate, avg_pool2d
from torchmetrics import MeanMetric
from typing import List, Tuple
from feature_extractor import FeatureExtractor
from torchmetrics import Metric
from metrics import F1Score, PrecisionRecallCurve


class PatchCore(LightningModule):
    def __init__(
        self,
        backbone: FeatureExtractor,
        k_nn: int,
        patch_stride: int,
        patch_kernel_size: int,
        patch_channels: int,
        img_size: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.k_nn = k_nn
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_channels = patch_channels
        self.img_size = img_size
        self.search_index = IndexFlatL2(patch_channels)
        self.latency = MeanMetric()
        self.seg_pr_curve = PrecisionRecallCurve()
        self.clf_pr_curve = PrecisionRecallCurve()
        self.seg_f1 = F1Score()
        self.clf_f1 = F1Score()
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        self.trainer.datamodule.train_dataset.to(self.device)
        self.trainer.datamodule.train_dataset.img_size = self.img_size

        if self.device.type == "cuda" and type(self.search_index) == IndexFlatL2:
            resource = StandardGpuResources()
            resource.noTempMemory()
            self.search_index = index_cpu_to_gpu(
                resource, self.device.index, self.search_index
            )

    def on_validation_start(self) -> None:
        self.trainer.datamodule.val_dataset.to(self.device)
        self.trainer.datamodule.val_dataset.img_size = self.img_size

    def on_test_start(self) -> None:
        self.trainer.datamodule.test_dataset.to(self.device)
        self.trainer.datamodule.test_dataset.img_size = self.img_size

    def training_step(self, batch, _) -> None:
        with no_grad():
            x, _ = batch
            patches = self.eval()(x)
            self.search_index.add(patches)

    def validation_step(self, batch, _) -> Tuple[Tensor, Tensor]:
        start_time = default_timer()
        x, y = batch
        patches = self(x)
        y_hat = self._predict(patches, batch_size=len(x))

        self.latency(default_timer() - start_time)
        self.seg_pr_curve(y_hat, y)
        self.clf_pr_curve(_clf(y_hat), _clf(y))

        return y_hat, y

    def validation_epoch_end(self, outputs: List[Tuple[Tensor]]) -> None:
        self.seg_f1.threshold = _f1_optimal_threshold(self.seg_pr_curve)
        self.clf_f1.threshold = _f1_optimal_threshold(self.clf_pr_curve)

        y_hat = cat([output[0] for output in outputs])
        y = cat([output[1] for output in outputs])

        self.seg_f1(y_hat, y)
        self.clf_f1(_clf(y_hat), _clf(y))

    def test_step(self, batch, _) -> None:
        start_time = default_timer()
        x, y = batch
        patches = self(x)
        y_hat = self._predict(patches, batch_size=len(x))

        self.latency(default_timer() - start_time)
        self.seg_f1(y_hat, y)
        self.clf_f1(_clf(y_hat), _clf(y))

    def forward(self, x: Tensor) -> Tensor:
        """Extracts patches from the backbone and returns them as a tensor."""
        layer_features: dict = self.backbone(x)

        # Extract patches from each layer
        layer_patches = [
            avg_pool2d(
                z,
                kernel_size=self.patch_kernel_size,
                stride=self.patch_stride,
                padding=int((self.patch_kernel_size - 1) / 2),
            )
            for z in layer_features.values()
        ]

        # Make sure all layers have the same resolution as the first layer
        self.patch_resolution = layer_patches[0].shape[-2:]
        for i in range(1, len(layer_patches)):
            layer_patches[i] = interpolate(
                layer_patches[i], size=self.patch_resolution, mode="bilinear"
            )

        # Flatten the patches
        for i in range(len(layer_patches)):
            layer_patches[i] = layer_patches[i].permute(0, 2, 3, 1)  # [N, H, W, C]
            layer_patches[i] = layer_patches[i].flatten(end_dim=-2)  # [N*H*W, C]

        # Make sure all layers have the same number of channels as the last layer
        for i in range(len(layer_patches) - 1):
            layer_patches[i] = adaptive_avg_pool1d(
                layer_patches[i], layer_patches[-1].shape[1]
            )

        # Combine the layers into the final patches
        patches = stack(layer_patches, dim=-1)  # [N*H*W, C, L]
        patches = patches.flatten(start_dim=-2)  # [N*H*W, C*L]
        return adaptive_avg_pool1d(patches, self.patch_channels)

    def _predict(
        self,
        patches: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Predict the anomaly scores for the patches using the memory bank."""
        scores, _ = self.search_index.search(patches, k=self.k_nn)  # [N*H*W, K=1]
        scores = mean(scores, dim=-1)  # [N*H*W]
        patch_scores = scores.reshape(
            batch_size, 1, *self.patch_resolution
        )  # [N, C=1, H, W]
        return interpolate(patch_scores, size=self.img_size, mode="bilinear")

    def configure_optimizers(self):
        pass


def _clf(y: Tensor) -> Tensor:  # [N, C=1, H, W]  # [N, C=1, H, W]
    """Clone y and convert it from pixel-wise to image-wise."""
    y_clf = y.clone()
    y_clf = flatten(y_clf, start_dim=-2)  # [N, C=1, H*W]
    return torch_max(y_clf, dim=-1).values  # [N, C=1]


def _f1_optimal_threshold(pr_curve: PrecisionRecallCurve) -> float:
    precision_curve, recall_curve, thresholds = pr_curve.compute()
    denominator = precision_curve + recall_curve
    denominator = where(eq(denominator, 0), ones_like(denominator), denominator)
    f1_scores = 2 * precision_curve * recall_curve / denominator
    return thresholds[argmax(f1_scores)]
