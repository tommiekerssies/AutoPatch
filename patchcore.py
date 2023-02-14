import faiss.contrib.torch_utils
from faiss import IndexFlatL2, StandardGpuResources, index_cpu_to_gpu
from timeit import default_timer
from pytorch_lightning import LightningModule
from torch.nn.functional import adaptive_avg_pool1d, interpolate, avg_pool2d
from torchmetrics import MeanMetric
from sampler import ApproximateGreedyCoresetSampler
from torchmetrics_v1_9_3 import AveragePrecision
from typing import Tuple
from feature_extractor import FeatureExtractor
import torch
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3


class PatchCore(LightningModule):
    def __init__(
        self,
        supernet: OFAMobileNetV3,
        stage_depths: list,
        block_kernel_sizes: list,
        block_expand_ratios: list,
        extraction_blocks: list,
        img_size: int,
        k_nn: int,
        patch_kernel_size: int,
        patch_stride: int,
        final_patch_channels: int,
        layer_patch_channels: int = None,
        coreset_ratio: float = None,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.img_size = img_size
        self.k_nn = k_nn
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.layer_patch_channels = layer_patch_channels
        self.final_patch_channels = final_patch_channels

        supernet.set_active_subnet(
            d=stage_depths, ks=block_kernel_sizes, e=block_expand_ratios
        )
        self.backbone = FeatureExtractor(
            supernet, [f"blocks.{i}" for i in extraction_blocks]
        )

        self.latency = MeanMetric()
        self.average_precision = AveragePrecision()

        self.sampler = ApproximateGreedyCoresetSampler(ratio=coreset_ratio or 1)
        self.memory_bank = IndexFlatL2(self.final_patch_channels)

    def on_fit_start(self) -> None:
        self.trainer.datamodule.to(self.device)
        self.trainer.datamodule.set_img_size(self.img_size)
        if self.device.type == "cuda" and type(self.memory_bank) == IndexFlatL2:
            resource = StandardGpuResources()
            resource.noTempMemory()
            self.memory_bank = index_cpu_to_gpu(
                resource, self.device.index, self.memory_bank
            )

    def training_step(self, batch, _) -> None:
        with torch.no_grad():
            x, _ = batch
            patches = self.eval()._patchify(x)
            patches = self.sampler.run(patches)
            self.memory_bank.add(patches)

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
            patches: [N*H*W, C]
            batch_size: N
        """
        # Extract patches from the backbone
        patches = self._patchify(x)  # [N*H*W, C]

        # Compute average distance to the k nearest neighbors
        scores, _ = self.memory_bank.search(patches, k=self.k_nn)  # [N*H*W, K=1]
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

        # Make sure all layers have the same number of channels if layer pooling is enabled
        if self.layer_patch_channels is not None:
            layer_patches = [
                adaptive_avg_pool1d(patches, self.layer_patch_channels)
                for patches in layer_patches
            ]

        # Combine the layers into the final patches and optionally apply final pooling
        patches = torch.cat(layer_patches, dim=-1)
        if patches.shape[-1] != self.final_patch_channels:
            patches = adaptive_avg_pool1d(patches, self.final_patch_channels)

        return patches

    def configure_optimizers(self):
        pass
