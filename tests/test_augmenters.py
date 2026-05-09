"""Tests for augmentation pipeline elements — pure tensor operations, no data files."""

import torch
import pytest

from csdp_pipeline.pipeline_elements.augmenters import (
    Augmenter,
    GlobalGaussianNoise,
    RegionalGaussianNoise,
)


def _make_channels(n_channels: int, length: int = 3840):
    """1D tensor list matching the format augmenters expect: list of (length,) tensors."""
    return [torch.zeros(length, dtype=torch.float32) for _ in range(n_channels)]


def _make_batch(n_eeg: int = 1, n_eog: int = 1, n_epochs: int = 4, fs: int = 128):
    samples = n_epochs * fs * 30
    eegs = torch.zeros(n_eeg, samples, dtype=torch.float32)
    eogs = torch.zeros(n_eog, samples, dtype=torch.float32)
    labels = torch.zeros(n_epochs, dtype=torch.long)
    tags = [{"dataset": "mock"}]
    return eegs, eogs, labels, tags


# ── GlobalGaussianNoise ────────────────────────────────────────────────────


class TestGlobalGaussianNoise:
    def test_output_shape_unchanged(self):
        aug = GlobalGaussianNoise(apply_prob=1.0, sigma=0.1, mean=0.0)
        x = _make_channels(2, length=3840)
        out = aug.augment(x)
        assert len(out) == 2
        assert out[0].shape == (3840,)

    def test_noise_applied_to_one_channel(self):
        # apply_prob=1.0 → augmentation always fires on exactly one random channel
        aug = GlobalGaussianNoise(apply_prob=1.0, sigma=1.0, mean=0.0)
        x = _make_channels(2, length=1000)
        out = aug.augment(x)
        # At least one channel should have changed from the zero baseline
        changed = sum(1 for ch in out if not torch.all(ch == 0))
        assert changed >= 1

    def test_zero_sigma_leaves_values_as_mean(self):
        aug = GlobalGaussianNoise(apply_prob=1.0, sigma=0.0, mean=3.0)
        x = _make_channels(1, length=500)
        out = aug.augment(x)
        # noise = mean + 0 * N(0,1) = 3.0 everywhere
        assert torch.allclose(out[0], torch.full((500,), 3.0))


# ── RegionalGaussianNoise ──────────────────────────────────────────────────


class TestRegionalGaussianNoise:
    def test_output_shape_unchanged(self):
        aug = RegionalGaussianNoise(min_frac=0.1, max_frac=0.3, apply_prob=1.0, sigma=0.1, mean=0.0)
        x = _make_channels(2, length=4000)
        out = aug.augment(x)
        assert len(out) == 2
        assert out[0].shape == (4000,)

    def test_only_region_is_modified(self):
        # With very tight region (10%–10% of length) on a zeros signal,
        # the rest should remain zero while the region gets noise.
        aug = RegionalGaussianNoise(min_frac=0.1, max_frac=0.2, apply_prob=1.0, sigma=10.0, mean=0.0)
        x = _make_channels(1, length=1000)
        out = aug.augment(x)
        # Not all values should be zero after augmentation
        assert not torch.all(out[0] == 0)

    def test_invalid_fractions_raise(self):
        with pytest.raises(AssertionError):
            RegionalGaussianNoise(min_frac=0.0, max_frac=0.5, apply_prob=1.0, sigma=1.0, mean=0.0)
        with pytest.raises(AssertionError):
            RegionalGaussianNoise(min_frac=0.1, max_frac=1.1, apply_prob=1.0, sigma=1.0, mean=0.0)


# ── Augmenter (IPipe) ──────────────────────────────────────────────────────


class TestAugmenter:
    def test_output_shape_preserved(self):
        aug = Augmenter(min_frac=0.1, max_frac=0.5, apply_prob=1.0, sigma=0.1, mean=0.0)
        batch = _make_batch()
        eegs_out, eogs_out, _, _ = aug.process(batch)
        assert eegs_out.shape == (1, 4 * 128 * 30)
        assert eogs_out.shape == (1, 4 * 128 * 30)

    def test_labels_pass_through_unchanged(self):
        aug = Augmenter(min_frac=0.1, max_frac=0.5, apply_prob=1.0, sigma=0.5, mean=0.0)
        batch = _make_batch()
        _, _, labels_out, _ = aug.process(batch)
        assert torch.equal(labels_out, batch[2])

    def test_tags_pass_through_unchanged(self):
        aug = Augmenter(min_frac=0.1, max_frac=0.5, apply_prob=1.0, sigma=0.5, mean=0.0)
        batch = _make_batch()
        _, _, _, tags_out = aug.process(batch)
        assert tags_out == batch[3]

    def test_zero_prob_leaves_signal_unchanged(self):
        # apply_prob=0.0 → augmentation never fires
        aug = Augmenter(min_frac=0.1, max_frac=0.5, apply_prob=0.0, sigma=100.0, mean=50.0)
        batch = _make_batch()
        eegs_out, eogs_out, _, _ = aug.process(batch)
        # Batch was all zeros; output should still be all zeros
        assert torch.all(eegs_out == 0)
        assert torch.all(eogs_out == 0)
