from math import ceil
import faiss.contrib.torch_utils
from faiss import IndexFlatL2, StandardGpuResources, index_cpu_to_gpu
from timeit import default_timer
from pytorch_lightning import LightningModule
from torch.nn.functional import adaptive_avg_pool1d, interpolate, avg_pool2d
from torchmetrics import MeanMetric
from sampler import ApproximateGreedyCoresetSampler
from torchmetrics_v1_9_3 import PrecisionRecallCurve
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
        self.pr_curve = PrecisionRecallCurve()

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
        patches = self._patchify(x)  # [N*H*W, C]
        scores = self._score(patches)  # [N*H*W]
        self.latency(default_timer() - start_time)
        return scores

    def validation_epoch_end(self, validation_step_outputs):
        self.val_scores = torch.cat(validation_step_outputs).sort().values  # [N*H*W]

    def test_step(self, batch, _) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        self.pr_curve(y_hat, y)

    def test_epoch_end(self, _):
        (
            self.precision_curve,
            self.recall_curve,
            self.thresholds,
        ) = self.pr_curve.compute()

    def forward(self, x: torch.Tensor):
        # Get the scores for each patch
        patches = self._patchify(x)  # [N*H*W, C]
        scores = self._score(patches)  # [N*H*W, C]

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

    def _score(self, patches: torch.Tensor) -> torch.Tensor:
        """Predict the anomaly scores for the patches using the memory bank.
        Args:
            patches: [N*H*W, C]
        """
        # Find the k nearest neighbor distances for each patch
        scores, _ = self.memory_bank.search(patches, k=self.k_nn)  # [N*H*W, K=1]

        # Compute mean distance to the k nearest neighbors
        return torch.mean(scores, dim=-1)  # [N*H*W]

    @staticmethod
    def _pixel_wise_to_img_wise(y):  # [N, C, H, W]
        y = torch.flatten(y, start_dim=-2)  # [N, C, H*W]
        return torch.max(y, dim=-1).values  # [N, C]

    def optimal_percentile(self):
        denominator = self.precision_curve + self.recall_curve
        denominator = torch.where(
            torch.eq(denominator, 0), torch.ones_like(denominator), denominator
        )
        f1_scores = 2 * self.precision_curve * self.recall_curve / denominator
        _, max_f1_idx = torch.max(f1_scores, dim=0)
        return (
            self._get_nearest_idx(self.val_scores, self.thresholds[max_f1_idx].item())
            + 1
        ) / len(self.val_scores)

    def f1_at_percentile(self, percentile: float):
        threshold = self.val_scores[ceil(percentile * len(self.val_scores)) - 1]
        idx = self._get_nearest_idx(self.thresholds, threshold)
        precision, recall = (
            self.precision_curve[idx].item(),
            self.recall_curve[idx].item(),
        )
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _get_nearest_idx(one_d_tensor: torch.Tensor, value: float) -> int:
        idx = (
            len(one_d_tensor) - 1
            if one_d_tensor[-1] < value
            else torch.searchsorted(one_d_tensor, value, right=True)
        )
        if one_d_tensor[idx - 1] == value:
            idx -= 1
        return idx

    def configure_optimizers(self):
        pass
