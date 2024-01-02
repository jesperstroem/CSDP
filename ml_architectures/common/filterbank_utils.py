"""
This class contains utility methods for filterbanks.
"""
import numpy as np


class FilterbankUtilities(object):
    @staticmethod
    def linear_tri_filterbank(num_filters: int, frequency_dim: int):
        """
        Compute a triangular linear filterbank shape.

        Args:
            num_filters (int): The number of filters in the bank.
            frequency_dim (int): The number of frequency bins in the spectrograms.

        Returns:
            ndarray: Filterbank shape
        """

        bins = np.floor(np.linspace(0, frequency_dim, num_filters + 2)).astype(int)
        filterbank = np.zeros([num_filters, frequency_dim])
        for r_idx in range(0, num_filters):
            # triangle up ramp
            for c_idx in range(bins[r_idx], bins[r_idx + 1]):
                filterbank[r_idx, c_idx] = (c_idx - bins[r_idx]) / (
                    bins[r_idx + 1] - bins[r_idx]
                )
            # triangle down ramp
            for c_idx in range(bins[r_idx + 1], bins[r_idx + 2]):
                filterbank[r_idx, c_idx] = (bins[r_idx + 2] - c_idx) / (
                    bins[r_idx + 2] - bins[r_idx + 1]
                )
        filterbank = np.transpose(filterbank)
        filterbank.astype(np.float32)
        return filterbank
