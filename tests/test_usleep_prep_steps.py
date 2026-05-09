"""Unit tests for usleep_prep_steps — pure numpy/scipy functions, no data files needed."""

import numpy as np
import pytest

from csdp_datastore.models import FilterSettings
from csdp_pipeline.preprocessing.usleep_prep_steps import (
    clip_channel,
    clip_channels,
    filter_channel,
    remove_dc,
    resample_channel,
    scale_channel,
    scale_channel_manual,
)

RNG = np.random.default_rng(42)


# ── remove_dc ─────────────────────────────────────────────────────────────


class TestRemoveDC:
    def test_output_mean_is_zero(self):
        x = RNG.standard_normal(1000) + 5.0  # large DC offset
        out = remove_dc(x)
        assert abs(np.mean(out)) < 1e-10

    def test_zero_mean_input_unchanged(self):
        x = RNG.standard_normal(500)
        x -= x.mean()
        out = remove_dc(x)
        np.testing.assert_allclose(out, x)

    def test_shape_preserved(self):
        x = RNG.standard_normal(300)
        assert remove_dc(x).shape == x.shape


# ── clip_channel ───────────────────────────────────────────────────────────


class TestClipChannel:
    def test_extreme_values_clipped(self):
        x = np.zeros(200)
        x[0] = 1e6  # massive outlier
        out = clip_channel(x)
        assert np.max(np.abs(out)) < 1e6

    def test_normal_signal_largely_unchanged(self):
        x = RNG.standard_normal(1000)
        out = clip_channel(x, min_max_times_global_iqr=20)
        # A normal distribution is very unlikely to exceed 20 × IQR
        np.testing.assert_array_equal(out, x)

    def test_shape_preserved(self):
        x = RNG.standard_normal(500)
        assert clip_channel(x).shape == x.shape

    def test_custom_threshold(self):
        x = np.array([-100.0, 0.0, 1.0, 2.0, 100.0])
        out = clip_channel(x, min_max_times_global_iqr=1)
        # With a very tight threshold, outliers must be clipped
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        assert np.max(np.abs(out)) <= iqr * 1 + 1e-9


# ── clip_channels ──────────────────────────────────────────────────────────


class TestClipChannels:
    def test_shape_preserved(self):
        x = RNG.standard_normal((3, 500))
        out = clip_channels(x)
        assert out.shape == x.shape

    def test_does_not_modify_input(self):
        x = RNG.standard_normal((2, 300))
        x_copy = x.copy()
        clip_channels(x)
        np.testing.assert_array_equal(x, x_copy)

    def test_outlier_clipped_in_correct_channel(self):
        x = RNG.standard_normal((2, 500))
        x[1, 0] = 1e9  # outlier only in channel 1
        out = clip_channels(x)
        assert out[1, 0] < 1e9
        # Channel 0 should be identical (no outlier)
        np.testing.assert_array_equal(out[0], x[0])


# ── scale_channel ──────────────────────────────────────────────────────────


class TestScaleChannel:
    def test_output_is_1d(self):
        x = RNG.standard_normal(500)
        out = scale_channel(x)
        assert out.ndim == 1

    def test_output_length_matches_input(self):
        x = RNG.standard_normal(400)
        assert len(scale_channel(x)) == len(x)

    def test_output_median_near_zero(self):
        x = RNG.standard_normal(1000) + 10.0  # large offset
        out = scale_channel(x)
        assert abs(np.median(out)) < 0.05

    def test_output_iqr_near_one(self):
        x = RNG.standard_normal(1000)
        out = scale_channel(x)
        iqr = np.subtract(*np.percentile(out, [75, 25]))
        assert abs(iqr - 1.0) < 0.1


# ── scale_channel_manual ───────────────────────────────────────────────────


class TestScaleChannelManual:
    def test_shape_preserved(self):
        x = RNG.standard_normal((2, 500))
        out = scale_channel_manual(x)
        assert out.shape == x.shape

    def test_per_channel_median_near_zero(self):
        x = RNG.standard_normal((3, 1000))
        x[0] += 5.0
        x[1] -= 3.0
        out = scale_channel_manual(x)
        for row in out:
            assert abs(np.median(row)) < 0.1


# ── resample_channel ───────────────────────────────────────────────────────


class TestResampleChannel:
    def test_downsample_halves_length(self):
        x = RNG.standard_normal(2000)
        out = resample_channel(x, output_rate=100, source_sample_rate=200)
        assert len(out) == 1000

    def test_upsample_doubles_length(self):
        x = RNG.standard_normal(500)
        out = resample_channel(x, output_rate=200, source_sample_rate=100)
        assert len(out) == 1000

    def test_identity_resample_preserves_length(self):
        x = RNG.standard_normal(800)
        out = resample_channel(x, output_rate=128, source_sample_rate=128)
        assert len(out) == len(x)

    def test_output_is_1d(self):
        x = RNG.standard_normal(300)
        out = resample_channel(x, output_rate=100, source_sample_rate=150)
        assert out.ndim == 1


# ── filter_channel ─────────────────────────────────────────────────────────


class TestFilterChannel:
    """
    Strategy: construct signals with known spectral content, apply a filter,
    and verify that the intended frequency band is attenuated or preserved.
    """

    def test_lowpass_attenuates_high_frequency(self):
        fs = 200
        t = np.linspace(0, 5, fs * 5, endpoint=False)
        # 1 Hz low-freq component + 50 Hz high-freq component
        x = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 50 * t)
        fs_obj = FilterSettings(lcut=None, hcut=10.0)
        out = filter_channel(x, fs, fs_obj)
        # After filtering, amplitude should be close to 1 (only 1 Hz remains)
        assert np.max(np.abs(out)) < 1.5

    def test_highpass_attenuates_dc_offset(self):
        fs = 128
        t = np.linspace(0, 10, fs * 10, endpoint=False)
        # DC offset + 20 Hz signal
        x = np.full_like(t, 5.0) + np.sin(2 * np.pi * 20 * t)
        fs_obj = FilterSettings(lcut=1.0)
        out = filter_channel(x, fs, fs_obj)
        # DC should be removed; mean should be near zero
        assert abs(np.mean(out)) < 0.5

    def test_output_shape_preserved(self):
        fs = 128
        x = RNG.standard_normal(fs * 30)
        fs_obj = FilterSettings(lcut=0.3)
        out = filter_channel(x, fs, fs_obj)
        assert out.shape == x.shape

    def test_bandpass_preserves_in_band_signal(self):
        fs = 256
        t = np.linspace(0, 10, fs * 10, endpoint=False)
        # 10 Hz signal well within 1–40 Hz band
        x = np.sin(2 * np.pi * 10 * t)
        fs_obj = FilterSettings(lcut=1.0, hcut=40.0)
        out = filter_channel(x, fs, fs_obj)
        # In-band signal should survive with most of its amplitude
        assert np.max(np.abs(out)) > 0.8
