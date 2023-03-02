import faiss.contrib.torch_utils
from statistics import mean
from faiss import IndexFlatL2, StandardGpuResources, index_cpu_to_gpu
from timeit import default_timer
from pytorch_lightning import LightningModule
from torch.nn.functional import adaptive_avg_pool1d, interpolate, avg_pool2d
from torchmetrics import MeanMetric
from sampler import ApproximateGreedyCoresetSampler
from typing import Tuple
import torch
from torch.nn import Linear, Module
from torchmetrics_v1_9_3 import precision_recall_curve
from skimage.measure import label, regionprops


class PatchCore(LightningModule):
    def __init__(
        self,
        backbone: Module,
        img_size: int,
        patch_kernel_sizes: list,
        patch_channels: int,
        max_sampling_time: int,
        coreset_ratio=1.0,
        num_starting_points: int = 10,
        projection_channels=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.img_size = img_size
        self.patch_kernel_sizes = patch_kernel_sizes
        self.patch_channels = patch_channels
        self.automatic_optimization = False
        self.memory_bank = IndexFlatL2(self.patch_channels)
        self.latency = MeanMetric()

        if projection_channels is not None:
            self.mapper = Linear(patch_channels, projection_channels, bias=False)
        else:
            self.mapper = lambda x: x

        self.sampler = ApproximateGreedyCoresetSampler(
            ratio=coreset_ratio,
            num_starting_points=num_starting_points,
            mapper=self.mapper,
            max_sampling_time=max_sampling_time,
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # Get feature map from each extraction block
        layer_features = list(self.backbone(x).values())

        # Extract patches from each feature map
        layer_patches = [
            avg_pool2d(
                z,
                kernel_size=self.patch_kernel_sizes[i],
                padding=int((self.patch_kernel_sizes[i] - 1) / 2),
                stride=1,
            )
            for i, z in enumerate(layer_features)
        ]

        # Make sure all feature maps have the same resolution (the maximum)
        self.patch_resolution = max(patches.shape[-2:] for patches in layer_patches)
        for i in range(len(layer_patches)):
            layer_patches[i] = interpolate(
                layer_patches[i], size=self.patch_resolution, mode="bilinear"
            )

        # Flatten the patches
        for i in range(len(layer_patches)):
            layer_patches[i] = layer_patches[i].permute(0, 2, 3, 1)  # [N, H, W, C]
            layer_patches[i] = layer_patches[i].flatten(end_dim=-2)  # [N*H*W, C]

        # Make sure all layers have the same number of channels
        layer_patches = [
            adaptive_avg_pool1d(patches, self.patch_channels)
            for patches in layer_patches
        ]

        # Combine the layers into the final patches and apply final pooling
        patches = torch.cat(layer_patches, dim=-1)
        patches = adaptive_avg_pool1d(patches, self.patch_channels)

        return patches

    def forward(self, x: torch.Tensor):
        # Extract patches from the backbone
        patches = self._patchify(x)  # [N*H*W, C]

        # Predict the anomaly scores for the patches using the memory bank
        scores, _ = self.memory_bank.search(patches, k=1)  # [N*H*W, K=1]

        # Compute mean distance to the k nearest neighbors
        scores = torch.mean(scores, dim=-1)  # [N*H*W]

        # Reshape the flattened scores to the shape of the original input (x)
        scores = scores.reshape(
            len(x), 1, *self.patch_resolution
        )  # [N, C=1, H=self.patch_resolution[0], W=self.patch_resolution[1]]

        # Finally, upsample the patch-level predictions to the original image size
        return interpolate(
            scores, size=self.img_size, mode="bilinear"
        )  # [N, C=1, H=self.img_size, W=self.img_size]

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

    def test_step(self, batch, _) -> Tuple[torch.Tensor, torch.Tensor]:
        start_time = default_timer()
        x, y = batch
        y_hat = self(x)
        self.latency(default_timer() - start_time)
        return y_hat, y

    def test_epoch_end(self, outputs) -> None:
        y_hat = torch.cat([y_hat for y_hat, _ in outputs], dim=0).squeeze().cpu()
        y = torch.cat([y for _, y in outputs], dim=0).squeeze().cpu()
        y = (y > 0).int()  # As image was resized there may be values between 0 and 1

        regions_per_image = [regionprops(label(y[i])) for i in range(len(y))]
        mean_region_area = mean(
            [region.area for regions in regions_per_image for region in regions]
        )

        sample_weights = torch.ones_like(y).float()
        for i in range(len(regions_per_image)):
            for region in regions_per_image[i]:
                sample_weights[i, region.coords[:, 0], region.coords[:, 1]] = (
                    mean_region_area / region.area
                )

        precision, recall, _ = precision_recall_curve(
            y_hat.flatten(), y.flatten(), sample_weights=sample_weights.flatten()
        )

        self.region_weighted_avg_precision = -torch.sum(
            (recall[1:] - recall[:-1]) * precision[:-1]
        )

    def configure_optimizers(self):
        pass
