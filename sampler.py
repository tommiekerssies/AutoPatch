"""
Some code in this file is borrowed from PatchCore (https://github.com/amazon-science/patchcore-inspection) (Apache-2.0 License)
"""

import torch
from tqdm import tqdm
import numpy as np
from torch.nn import Linear


class ApproximateGreedyCoresetSampler:
    def __init__(
        self,
        ratio: float,
        num_starting_points: int,
        mapper: Linear,
        max_sampling_time: int = None,
    ):
        if not 0 < ratio <= 1:
            raise ValueError("ratio value not in (0,1].")
        self.ratio = ratio
        self.num_starting_points = num_starting_points
        self.mapper = mapper
        self.max_sampling_time = max_sampling_time

    def run(self, patches: torch.Tensor) -> torch.Tensor:
        """Subsamples patches using Greedy Coreset.

        Args:
            patches: [N x D]
        """
        if self.ratio == 1:
            return patches
        sample_indices = self._compute_greedy_coreset_indices(self.mapper(patches))
        return patches[sample_indices]

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, patches: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            patches: [NxD] input feature bank to sample.
        """
        start_points = np.random.choice(
            len(patches), self.num_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            patches, patches[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(patches) * self.ratio)

        with tqdm(range(num_coreset_samples), desc="Subsampling...") as t:
            for _ in t:
                if (
                    self.max_sampling_time is not None
                    and t.format_dict["elapsed"] > self.max_sampling_time
                ):
                    raise RuntimeError("Max sampling time reached")
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    patches, patches[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)
