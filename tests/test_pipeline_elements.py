"""Tests for pipeline elements using synthetic tensors — no HDF5 data needed."""

import torch

from csdp_pipeline.pipeline_elements.resampler import Resampler
from csdp_pipeline.pipeline_elements.spectrogram import Spectrogram


def _make_batch(n_epochs: int, fs: int, n_eeg: int = 2, n_eog: int = 1):
    """Build a synthetic (eegs, eogs, labels, tags) tuple matching the IPipe contract."""
    samples = n_epochs * fs * 30
    eegs = torch.randn(n_eeg, samples, dtype=torch.float64)
    eogs = torch.randn(n_eog, samples, dtype=torch.float64)
    labels = torch.zeros(n_epochs, dtype=torch.long)
    tags = [{"dataset": "mock", "subject": "s0", "record": "r0"}]
    return eegs, eogs, labels, tags


# ── Resampler ─────────────────────────────────────────────────────────────


class TestResampler:
    def test_no_resampling_preserves_shape(self):
        r = Resampler(source_sample=128, target_sample=128)
        batch = _make_batch(n_epochs=4, fs=128)
        eegs, eogs, _, _ = r.process(batch)
        assert eegs.shape == (2, 4 * 128 * 30)
        assert eogs.shape == (1, 4 * 128 * 30)

    def test_downsample_halves_length(self):
        r = Resampler(source_sample=256, target_sample=128)
        batch = _make_batch(n_epochs=2, fs=256)
        eegs, eogs, _, _ = r.process(batch)
        assert eegs.shape == (2, 2 * 128 * 30)
        assert eogs.shape == (1, 2 * 128 * 30)

    def test_upsample_doubles_length(self):
        r = Resampler(source_sample=64, target_sample=128)
        batch = _make_batch(n_epochs=2, fs=64)
        eegs, eogs, _, _ = r.process(batch)
        assert eegs.shape == (2, 2 * 128 * 30)
        assert eogs.shape == (1, 2 * 128 * 30)

    def test_labels_and_tags_pass_through_unchanged(self):
        r = Resampler(source_sample=128, target_sample=128)
        batch = _make_batch(n_epochs=3, fs=128)
        _, _, labels_out, tags_out = r.process(batch)
        assert torch.equal(labels_out, batch[2])
        assert tags_out == batch[3]

    def test_channel_count_preserved_after_resampling(self):
        r = Resampler(source_sample=200, target_sample=128)
        batch = _make_batch(n_epochs=2, fs=200, n_eeg=4, n_eog=2)
        eegs, eogs, _, _ = r.process(batch)
        assert eegs.shape[0] == 4
        assert eogs.shape[0] == 2


# ── Spectrogram ───────────────────────────────────────────────────────────


class TestSpectrogram:
    """
    Default params: win_size=2, fs_fourier=100, overlap=1, sample_rate=100.
    Input must be multiples of sample_rate*30 = 3000 samples per channel.
    Output shape: (n_channels, n_epochs, freq_bins, time_bins).
    """

    def test_output_channel_and_epoch_dims(self):
        spec = Spectrogram(win_size=2, fs_fourier=100, overlap=1, sample_rate=100)
        batch = _make_batch(n_epochs=3, fs=100, n_eeg=2, n_eog=1)
        eegs_out, eogs_out, _, _ = spec.process(batch)
        assert eegs_out.shape[0] == 2, "EEG channel count"
        assert eegs_out.shape[1] == 3, "epoch count"
        assert eogs_out.shape[0] == 1, "EOG channel count"
        assert eogs_out.shape[1] == 3, "epoch count"

    def test_spectrogram_bins_are_2d(self):
        spec = Spectrogram(win_size=2, fs_fourier=100, overlap=1, sample_rate=100)
        batch = _make_batch(n_epochs=2, fs=100)
        eegs_out, _, _, _ = spec.process(batch)
        # eegs_out: (channels, epochs, freq, time)
        assert eegs_out.ndim == 4

    def test_labels_and_tags_pass_through(self):
        spec = Spectrogram(win_size=2, fs_fourier=100, overlap=1, sample_rate=100)
        batch = _make_batch(n_epochs=2, fs=100)
        _, _, labels_out, tags_out = spec.process(batch)
        assert torch.equal(labels_out, batch[2])
        assert tags_out == batch[3]
