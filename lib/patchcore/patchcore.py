"""PatchCore and PatchCore detection methods."""
import logging

from numpy import concatenate, ndarray
import torch
import torch.nn.functional as F

from lib.patchcore.common import (
    Preprocessing,
    Aggregator,
    FaissNN,
    NearestNeighbourScorer,
    RescaleSegmentor,
)


LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(
        self,
        device,
        n_layers,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        **kwargs,
    ):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()

        self.device = device
        self.input_shape = input_shape

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        preprocessing = Preprocessing(n_layers, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = Aggregator(target_dim=target_embed_dimension)

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        nn_method = FaissNN(torch.cuda.is_available())
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

    def _embed(self, out, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            return features.detach().cpu().numpy() if detach else features

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in out]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, outs):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(outs)

    def _fill_memory_bank(self, outs):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        features = [self._embed(out) for out in outs]
        features = concatenate(features, axis=0)
        # features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, outs):
        _ = self.forward_modules.eval()

        with torch.no_grad():
            features = []
            patch_shapes = []
            outs_size = 0

            for out in outs:
                outs_size += out[0].shape[0]
                features_, patch_shapes_ = self._embed(out, provide_patch_shapes=True)
                features.append(features_)
                patch_shapes.append(patch_shapes_)

            features = concatenate(features, axis=0)
            patch_shapes = concatenate(patch_shapes, axis=0)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            # self.anomaly_scorer.nn_method.reset_index()
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batch_size=outs_size
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batch_size=outs_size
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(outs_size, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return list(image_scores), list(masks)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batch_size):
        return x.reshape(batch_size, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        return x.numpy() if was_numpy else x
