import contextlib
from timeit import default_timer
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch import (
    argmax,
    eq,
    flatten,
    mean,
    no_grad,
    ones_like,
    cat,
    tensor,
    where,
    any,
    max as torch_max,
)
from torchmetrics import MeanMetric
from torchmetrics.classification import PrecisionRecallCurve, F1
import copy
from typing import List, Tuple
import torch
import torch.nn.functional as F
from faiss import IndexFlatL2
import faiss.contrib.torch_utils
from ofa.model_zoo import ofa_net


class LastLayerToExtractReachedException(Exception):
    pass


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            with contextlib.suppress(LastLayerToExtractReachedException):
                _ = self.backbone(images)
        return self.outputs


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class PatchCore(LightningModule):
    def __init__(
        self,
        img_size,
        k_nn,
        patch_channels,
        patch_kernel_size,
        patch_stride,
        extraction_layers,
        supernet_name,
        subnet_kernel_size,
        subnet_expansion_ratio,
        subnet_depth,
    ):
        super().__init__()
        self.img_size = img_size
        self.k_nn = k_nn
        self.patch_channels = patch_channels
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.automatic_optimization = False
        self.latency = MeanMetric()
        self.pr_curve = PrecisionRecallCurve()
        self.memory = IndexFlatL2(patch_channels)

        supernet = ofa_net(supernet_name, pretrained=True)
        supernet.set_active_subnet(
            ks=subnet_kernel_size, e=subnet_expansion_ratio, d=subnet_depth
        )
        self.backbone = NetworkFeatureAggregator(supernet, extraction_layers)

    def on_fit_start(self):
        self.memory.reset()

    def training_step(self, batch, _):
        self.eval()
        with no_grad():
            x, _ = batch
            patches = self(x)
            self.memory.add(patches)

    def validation_step(self, batch, _) -> Tuple[tensor, tensor]:
        x, y = batch
        if len(x) != 1:
            raise ValueError("Only batch size of 1 is supported for validation.")
        start_time = default_timer()
        patches = self(x)
        y_hat = self._predict(patches, batch_size=len(x))
        self.latency(default_timer() - start_time)
        self.pr_curve(flatten(y_hat), flatten(y))
        return y_hat, y

    def validation_epoch_end(self, outputs: List[Tuple[tensor]]) -> None:
        precision_curve, recall_curve, thresholds = self.pr_curve.compute()
        denominator = precision_curve + recall_curve
        denominator = where(eq(denominator, 0), ones_like(denominator), denominator)
        f1_scores = 2 * precision_curve * recall_curve / denominator

        self.threshold = thresholds[argmax(f1_scores)]
        self.segmentation_f1 = F1(threshold=self.threshold)
        self.classification_f1 = F1(threshold=self.threshold)

        y_hat = cat([output[0] for output in outputs])
        y = cat([output[1] for output in outputs])
        self._f1_scores(y_hat, y)

    def test_step(self, batch, _):
        x, y = batch
        patches = self(x)
        y_hat = self._predict(patches, batch_size=len(x))
        self._f1_scores(y_hat, y)

    def forward(self, x: tensor) -> tensor:
        """Extracts patches from the backbone and returns them as a tensor.
        Args:
            x: [bs, c, h, w].
        Returns:
            patches: [num_patches, patch_channels].
        """
        layer_features: dict = self.backbone(x)

        # Patchify each layer
        layer_patches = []
        layer_resolutions = []
        for z in layer_features.values():
            patches, resolution = self._patchify(z)
            layer_patches.append(patches)
            layer_resolutions.append(resolution)

        # Make sure all layers have the same resolution
        self.patch_resolution = layer_resolutions[0]
        for i in range(1, len(layer_patches)):
            layer_patches[i] = PatchCore._upscale_patches(
                layer_patches[i], layer_resolutions[i], self.patch_resolution
            )
        layer_patches = [
            patches.reshape(-1, *patches.shape[-3:]) for patches in layer_patches
        ]

        # Make sure all layers have the same number of channels
        highest_channels = max(z.shape[1] for z in layer_features.values())
        for i in range(len(layer_patches)):
            layer_patches[i] = layer_patches[i].reshape(len(layer_patches[i]), 1, -1)
            layer_patches[i] = F.adaptive_avg_pool1d(
                layer_patches[i], highest_channels
            ).squeeze(1)

        # Combine the layers into the final patches
        patches = torch.stack(layer_patches, dim=1)
        patches = patches.reshape(len(patches), 1, -1)
        patches = F.adaptive_avg_pool1d(patches, self.patch_channels)

        return patches.reshape(len(patches), -1)

    def _patchify(self, z: tensor) -> Tuple[tensor, list]:
        """Convert a tensor z into a tensor of respective patches.
        Args:
            z: [bs, c, h, w]
        Returns:
            x: [bs * w//stride * h//stride, c, patchsize, patchsize]
            number_of_total_patches
        """
        padding = int((self.patch_kernel_size - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patch_kernel_size,
            stride=self.patch_stride,
            padding=padding,
            dilation=1,
        )
        unfolded_features = unfolder(z)
        number_of_total_patches = []
        for s in z.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patch_kernel_size - 1) - 1
            ) / self.patch_stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *z.shape[:2], self.patch_kernel_size, self.patch_kernel_size, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        return unfolded_features, number_of_total_patches

    @staticmethod
    def _upscale_patches(patches, resolution, highest_resolution):
        """Upscale patches to the highest resolution.
        Args:
            patches: [bs * w//stride * h//stride, c, patchsize, patchsize]
            resolution: list
            highest_resolution: list
        Returns:
            patches: [bs * w//stride * h//stride, c, patchsize, patchsize]
        """
        patches = patches.reshape(
            patches.shape[0], resolution[0], resolution[1], *patches.shape[2:]
        )
        patches = patches.permute(0, -3, -2, -1, 1, 2)
        perm_base_shape = patches.shape
        patches = patches.reshape(-1, *patches.shape[-2:])

        patches = F.interpolate(
            patches.unsqueeze(1),
            size=(highest_resolution[0], highest_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        patches = patches.reshape(
            *perm_base_shape[:-2], highest_resolution[0], highest_resolution[1]
        )
        patches = patches.permute(0, -2, -1, 1, 2, 3)
        return patches.reshape(len(patches), -1, *patches.shape[-3:])

    def _predict(
        self,
        patches: tensor,
        batch_size: int,
    ):
        """Predict the anomaly scores for the patches using the memory bank.
        Args:
            patches: [num_patches, patch_channels]
            batch_size: int
        Returns:
            scores: [bs, w, h]
        """
        scores, _ = self.memory.search(patches, k=self.k_nn)  # [n*w*h, k=1]
        scores = mean(scores, dim=-1)  # [n*w*h]
        patch_scores = scores.reshape(batch_size, *self.patch_resolution)
        return F.interpolate(patch_scores.unsqueeze(1), size=self.img_size).squeeze(1)

    def _f1_scores(self, y_hat, y):
        """Calculates the F1 scores for the segmentation and classification tasks.
        Args:
            y_hat: [bs, c, w, h]
            y: [bs, c, w, h]
        """
        self.segmentation_f1(flatten(y_hat), flatten(y))
        self.classification_f1(torch_max(y_hat, dim=-1).values, any(y, dim=-1))

    def configure_optimizers(self):
        pass
