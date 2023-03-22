"""
Some code in this file is borrowed from PatchCore (https://github.com/amazon-science/patchcore-inspection) (Apache-2.0 License)
"""


import contextlib
from copy import deepcopy
from torch.nn import Module, Sequential
from torch import no_grad


class LastLayerToExtractReachedException(Exception):
    pass


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = deepcopy(layer_name == last_layer_to_extract)

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class FeatureExtractor(Module):
    def __init__(self, backbone, extraction_layers):
        super().__init__()
        """Efficient extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            extraction_layers: [list of str]
        """
        self.extraction_layers = extraction_layers
        self.backbone = backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for layer in extraction_layers:
            forward_hook = ForwardHook(self.outputs, layer, extraction_layers[-1])
            if "." in layer:
                extract_block, extract_idx = layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][layer]

            if isinstance(network_layer, Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )

    def forward(self, images):
        self.outputs.clear()
        with no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            with contextlib.suppress(LastLayerToExtractReachedException):
                _ = self.backbone(images)
        return self.outputs
