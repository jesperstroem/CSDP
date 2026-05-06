"""Unit tests for pure signal-processing functions and data models — no data files required."""

import numpy as np
import pytest

from csdp_datastore.models import FilterSettings
from csdp_pipeline.preprocessing.spectrogram import create_spectrogram_images

# ── FilterSettings ────────────────────────────────────────────────────────


class TestFilterSettings:
    def test_defaults_are_highpass(self):
        fs = FilterSettings()
        assert fs.type == "highpass"
        assert fs.order == 2

    def test_bandpass_when_both_cutoffs_given(self):
        fs = FilterSettings(lcut=0.1, hcut=40.0)
        assert fs.type == "bandpass"
        assert fs.cutoffs == [0.1, 40.0]
        assert fs.order == 2

    def test_lowpass_when_only_hcut(self):
        fs = FilterSettings(lcut=None, hcut=40.0)
        assert fs.type == "lowpass"
        assert fs.cutoffs == 40.0

    def test_highpass_when_only_lcut(self):
        fs = FilterSettings(lcut=0.5)
        assert fs.type == "highpass"
        assert fs.cutoffs == 0.5

    def test_custom_order(self):
        fs = FilterSettings(lcut=0.1, hcut=40.0, order=5)
        assert fs.order == 5


# ── create_spectrogram_images ──────────────────────────────────────────────


class TestCreateSpectrogramImages:
    """
    Default Spectrogram params: win_size=2, fs_fourier=100, overlap=1, sample_rate=100.
    epoch_length = 100 * 30 = 3000 samples.
    Assertion: (3000 - 200) % (200 - 100) = 0 ✓
    """

    def test_returns_one_spectrogram_per_epoch(self):
        fs = 100
        n_epochs = 3
        x = np.random.randn(n_epochs * fs * 30)
        _, _, specs = create_spectrogram_images(x, sample_rate=fs)
        assert len(specs) == n_epochs

    def test_single_epoch(self):
        fs = 100
        x = np.random.randn(fs * 30)
        _, _, specs = create_spectrogram_images(x, sample_rate=fs)
        assert len(specs) == 1

    def test_each_spectrogram_is_2d(self):
        fs = 100
        x = np.random.randn(2 * fs * 30)
        _, _, specs = create_spectrogram_images(x, sample_rate=fs)
        for s in specs:
            assert s.ndim == 2, "each spectrogram should be (freq_bins, time_bins)"

    def test_all_epochs_same_shape(self):
        fs = 100
        x = np.random.randn(4 * fs * 30)
        _, _, specs = create_spectrogram_images(x, sample_rate=fs)
        shapes = [s.shape for s in specs]
        assert len(set(shapes)) == 1, "all epoch spectrograms must have identical shape"

    def test_truncates_to_full_epochs(self):
        """Extra samples beyond a complete epoch are silently dropped."""
        fs = 100
        extra = 50
        x = np.random.randn(2 * fs * 30 + extra)
        _, _, specs = create_spectrogram_images(x, sample_rate=fs)
        assert len(specs) == 2
