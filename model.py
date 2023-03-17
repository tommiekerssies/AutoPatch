import faiss.contrib.torch_utils
from statistics import mean
from faiss import IndexFlatL2, StandardGpuResources, index_cpu_to_gpu
from timeit import default_timer
from pytorch_lightning import LightningModule
from torch.nn.functional import interpolate, avg_pool2d
from torchmetrics import MeanMetric
from sampler import ApproximateGreedyCoresetSampler
from typing import Tuple
import torch
from torch.nn import Linear, Module
from torchmetrics_v1_9_3 import _auroc_compute, precision_recall_curve
from skimage.measure import label, regionprops


class Model(LightningModule):
    def __init__(
        self,
        backbone: Module,
        img_size: int,
        patch_sizes: list,
        patch_channels: int,
        coreset_ratio=1.0,
        num_starting_points: int = 10,
        projection_channels=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.patch_channels = patch_channels
        self.automatic_optimization = False
        self.memory_bank = IndexFlatL2(patch_channels)
        self.latency = MeanMetric()

        if projection_channels is not None:
            self.mapper = Linear(patch_channels, projection_channels, bias=False)
        else:
            self.mapper = lambda x: x

        self.sampler = ApproximateGreedyCoresetSampler(
            ratio=coreset_ratio,
            num_starting_points=num_starting_points,
            mapper=self.mapper,
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # Get output from each extraction block
        layer_outputs = list(self.backbone(x).values())

        if len(layer_outputs) != len(self.patch_sizes):
            raise RuntimeError(
                "The returned number of layers from the backbone and the number of patch sizes is not the same."
            )

        self.patch_resolution = max(output.shape[-2:] for output in layer_outputs)
        layer_patches = []
        for i in range(len(layer_outputs)):
            # Perform local neighbourhood aggregation
            patches = avg_pool2d(
                layer_outputs[i],
                kernel_size=self.patch_sizes[i],
                padding=int((self.patch_sizes[i] - 1) / 2),
                stride=1,
            )

            # Make sure the patches have the right resolution
            patches = interpolate(patches, size=self.patch_resolution, mode="bilinear")

            # Flatten the patches
            patches = patches.permute(0, 2, 3, 1)  # [N, H, W, C]
            patches = patches.flatten(end_dim=-2)  # [N*H*W, C]

            layer_patches.append(patches)

        # Combine the layers into the final patches
        return torch.cat(layer_patches, dim=-1)

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
        scores = interpolate(
            scores, size=self.img_size, mode="bilinear"
        )  # [N, C=1, H=self.img_size, W=self.img_size]

        return scores.squeeze()  # [N, H=self.img_size, W=self.img_size]

    def on_fit_start(self) -> None:
        self.trainer.datamodule.to(self.device)
        if self.device.type == "cuda" and type(self.memory_bank) == IndexFlatL2:
            resource = StandardGpuResources()
            resource.noTempMemory()
            self.memory_bank = index_cpu_to_gpu(
                resource, self.device.index, self.memory_bank
            )

    def training_step(self, batch, _) -> None:
        with torch.no_grad():
            x, _, _ = batch
            patches = self.eval()._patchify(x)
            patches = self.sampler.run(patches)
            self.memory_bank.add(patches)

    def _inference_step(self, batch):
        start_time = default_timer()
        x, y, x_type = batch
        y_hat = self(x)
        self.latency(default_timer() - start_time)
        return y_hat, y, x_type

    def validation_step(self, batch, _) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._inference_step(batch)

    def test_step(self, batch, _) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._inference_step(batch)

    def _compute_metrics(
        self, outputs: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_hat = torch.cat([y_hat for y_hat, _, _ in outputs], dim=0).cpu()
        y = torch.cat([y for _, y, _ in outputs], dim=0).cpu()

        self.AUROC = _auroc_compute(y_hat.flatten(), y.flatten(), "binary").item()
        self.partial_AUROC = _auroc_compute(
            y_hat.flatten(), y.flatten(), "binary", max_fpr=0.3
        ).item()

        precision, recall, _ = precision_recall_curve(y_hat.flatten(), y.flatten())
        self.AP = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1]).item()

        x_type = []
        for _, _, x_type_i in outputs:
            x_type.extend(x_type_i)

        region_count_per_type = {}
        regions_per_image = []
        for i in range(len(y)):
            regions = regionprops(label(y[i]))
            if x_type[i] not in region_count_per_type:
                region_count_per_type[x_type[i]] = 0
            region_count_per_type[x_type[i]] += len(regions)
            regions_per_image.append(regions)
        region_count_per_type.pop("good")

        mean_region_count = mean(region_count_per_type.values())
        mean_region_area = mean(
            [region.area for regions in regions_per_image for region in regions]
        )

        sample_weights = torch.ones_like(y, dtype=torch.float32)
        for i in range(len(regions_per_image)):
            for region in regions_per_image[i]:
                sample_weights[i, region.coords[:, 0], region.coords[:, 1]] = (
                    mean_region_area / region.area
                ) * (mean_region_count / region_count_per_type[x_type[i]])

        weighted_precision, weighted_recall, _ = precision_recall_curve(
            y_hat.flatten(), y.flatten(), sample_weights=sample_weights.flatten()
        )
        self.wAP = -torch.sum(
            (weighted_recall[1:] - weighted_recall[:-1]) * weighted_precision[:-1]
        ).item()

    def validation_epoch_end(self, outputs) -> None:
        self._compute_metrics(outputs)

    def test_epoch_end(self, outputs) -> None:
        self._compute_metrics(outputs)

    def configure_optimizers(self):
        pass
