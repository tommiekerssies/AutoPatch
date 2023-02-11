import faiss.contrib.torch_utils
from faiss import IndexFlatL2, StandardGpuResources, index_cpu_to_gpu
from timeit import default_timer
from pytorch_lightning import LightningModule
from torch.nn.functional import adaptive_avg_pool1d, interpolate, avg_pool2d
from torchmetrics import MeanMetric
from torchmetrics_v1_9_3 import AveragePrecision
from typing import Tuple
from feature_extractor import FeatureExtractor
import torch


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
        self.automatic_optimization = False
        self.backbone = backbone
        self.k_nn = k_nn
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_channels = patch_channels
        self.img_size = img_size
        self.latency = MeanMetric()
        self.average_precision = AveragePrecision()
        self.search_index = IndexFlatL2(patch_channels)

    def on_fit_start(self) -> None:
        self.trainer.datamodule.to(self.device)
        self.trainer.datamodule.set_img_size(self.img_size)
        if self.device.type == "cuda" and type(self.search_index) == IndexFlatL2:
            resource = StandardGpuResources()
            resource.noTempMemory()
            self.search_index = index_cpu_to_gpu(
                resource, self.device.index, self.search_index
            )

    def training_step(self, batch, _) -> None:
        with torch.no_grad():
            x, _ = batch
            patches = self.eval()._patchify(x)
            self.search_index.add(patches)

    def validation_step(self, batch, _) -> Tuple[torch.Tensor, torch.Tensor]:
        start_time = default_timer()
        x, _ = batch
        y_hat = self(x)
        self.latency(default_timer() - start_time)
        return y_hat

    def validation_epoch_end(self, validation_step_outputs):
        y_hat = torch.cat(validation_step_outputs)
        self.nominal_score_mean = torch.mean(y_hat)
        self.nominal_score_std = torch.std(y_hat)

    def test_step(self, batch, _) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute average precision on a test batch (equivalent to area under precision-recall curve)."""
        # Get y_hat and y, both shape [N, C=1, H, W]
        x, y = batch
        y_hat = self(x)

        # Convert from pixel-wise to image-wise
        y_hat = torch.flatten(y_hat, start_dim=-2)  # [N, C=1, H*W]
        y = torch.flatten(y, start_dim=-2)  # [N, C=1, H*W]
        y_hat = torch.max(y_hat, dim=-1).values  # [N, C=1]
        y = torch.max(y, dim=-1).values  # [N, C=1]

        # Apply z-score normalization based on the mean and std of the validation set
        y_hat -= self.nominal_score_mean
        y_hat /= self.nominal_score_std

        # Finally, compute the metric
        self.average_precision(y_hat, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the anomaly scores for the patches using the memory bank.
        Args:
            patches: [N*H*W, C=patch_channels]
            batch_size: N
        """
        # Extract patches from the backbone
        patches = self._patchify(x)  # [N*H*W, C=patch_channels]

        # Compute average distance to the k nearest neighbors
        scores, _ = self.search_index.search(patches, k=self.k_nn)  # [N*H*W, K=1]
        scores = torch.mean(scores, dim=-1)  # [N*H*W]

        # Reshape the flattened scores to the shape of the original input (x)
        patch_scores = scores.reshape(
            len(x), 1, *self.patch_resolution
        )  # [N, C=1, H=self.patch_resolution[0], W=self.patch_resolution[1]]

        # Finally, upsample the patch-level predictions to the original image size
        return interpolate(
            patch_scores, size=self.img_size, mode="bilinear"
        )  # [N, C=1, H=self.img_size, W=self.img_size]

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts patches from the backbone.
        Args:
            x: [N, C, H, W]
        """
        # Extract patches from each layer
        layer_patches = [
            avg_pool2d(
                z,
                kernel_size=self.patch_kernel_size,
                stride=self.patch_stride,
                padding=int((self.patch_kernel_size - 1) / 2),
            )
            for z in self.backbone(x).values()
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
        patches = torch.stack(layer_patches, dim=-1)  # [N*H*W, C, L]
        patches = patches.flatten(start_dim=-2)  # [N*H*W, C*L]
        return adaptive_avg_pool1d(
            patches, self.patch_channels
        )  # [N*H*W, C=patch_channels]

    def configure_optimizers(self):
        pass
