import numpy as np
import pytest
import mne
import torch

from csdp_pipeline.pipeline_elements.sleep_dataset_class import sleep_dataset_from_paths

SFREQ = 128
CH_NAMES = ["F4:M1", "C4:M1", "E1:M2", "E2:M1"]
N_EPOCHS = 5  # 5 × 30 s = 150 s of data


def _make_raw(sfreq=SFREQ, ch_names=CH_NAMES, n_epochs=N_EPOCHS):
    n_samples = int(sfreq * 30 * n_epochs)
    rng = np.random.default_rng(42)
    data = rng.normal(0, 50e-6, (len(ch_names), n_samples))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture
def patch_open(monkeypatch):
    raw = _make_raw()
    monkeypatch.setattr(sleep_dataset_from_paths, "open_eeg_file", lambda f, preload=True: raw)
    return raw


# ── get_available_channels ────────────────────────────────────────────────────

class TestGetAvailableChannels:
    def test_returns_channel_names(self, monkeypatch):
        raw = _make_raw()
        monkeypatch.setattr(sleep_dataset_from_paths, "open_eeg_file", lambda f, preload=True: raw)
        result = sleep_dataset_from_paths.get_available_channels(["fake.edf"])
        assert result == [CH_NAMES]

    def test_multiple_files(self, monkeypatch):
        raw = _make_raw()
        monkeypatch.setattr(sleep_dataset_from_paths, "open_eeg_file", lambda f, preload=True: raw)
        result = sleep_dataset_from_paths.get_available_channels(["a.edf", "b.edf"])
        assert len(result) == 2
        assert result[0] == result[1] == CH_NAMES


# ── construction ──────────────────────────────────────────────────────────────

class TestConstructFromPaths:
    def test_ch_names_channel_count(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        x, _ = ds.get_full_record(0)
        assert x.shape[0] == len(CH_NAMES)

    def test_ch_names_epoch_samples(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        x, _ = ds.get_full_record(0)
        assert x.shape[1] % (SFREQ * 30) == 0

    def test_derivations_channel_count(self, patch_open):
        derivations = [(["F4:M1"], ["C4:M1"]), (["E1:M2"], ["E2:M1"])]
        ds = sleep_dataset_from_paths(["fake.edf"], derivations=derivations, L=1)
        x, _ = ds.get_full_record(0)
        assert x.shape[0] == 2

    def test_no_args_loads_all_channels(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], L=1)
        x, _ = ds.get_full_record(0)
        assert x.shape[0] == len(CH_NAMES)

    def test_output_is_float32_tensor(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        x, _ = ds.get_full_record(0)
        assert isinstance(x, torch.Tensor)
        assert x.dtype == torch.float32

    def test_nan_epochs_shape(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        assert len(ds.nanEpochs) == 1
        assert ds.nanEpochs[0].shape == (len(CH_NAMES), N_EPOCHS)

    def test_nan_epochs_are_bool(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        assert ds.nanEpochs[0].dtype == bool


# ── __getitem__ modes ─────────────────────────────────────────────────────────

class TestFullRecordsMode:
    def test_len_equals_num_files(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1, fullRecords=True)
        assert len(ds) == 1

    def test_getitem_returns_data_and_index(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1, fullRecords=True)
        item = ds[0]
        assert len(item) == 2
        x, file_idx = item
        assert isinstance(x, torch.Tensor)
        assert file_idx == 0


class TestMinibatchMode:
    def test_len_positive(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        assert len(ds) > 0

    def test_getitem_data_shape(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        x, _ = ds[0]
        assert x.shape == (len(CH_NAMES), SFREQ * 30)

    def test_getitem_L2_shape(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=2)
        x, _ = ds[0]
        assert x.shape == (len(CH_NAMES), SFREQ * 30 * 2)


# ── checkDerivations ──────────────────────────────────────────────────────────

class TestCheckDerivations:
    def test_valid_derivation_passes(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        raw = _make_raw()
        result = ds.checkDerivations(raw.info, [(["F4:M1"], ["C4:M1"])])
        assert len(result) == 1

    def test_invalid_channel_returns_none_descriptor(self, patch_open):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        raw = _make_raw()
        result = ds.checkDerivations(raw.info, [(["INVALID"], ["F4:M1"])])
        # the bad side becomes empty → propagates as None descriptor
        assert result[0][0] == [None] or result[0][0] == []


# ── SDC round-trip ────────────────────────────────────────────────────────────

class TestSdcRoundTrip:
    def test_data_preserved(self, patch_open, tmp_path):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        sdc_path = str(tmp_path / "test")
        ds.save_to_sdc(sdc_path)

        ds2 = sleep_dataset_from_paths([], sdcFile=sdc_path)
        x1, _ = ds.get_full_record(0)
        x2, _ = ds2.get_full_record(0)
        assert torch.allclose(x1, x2)

    def test_shape_preserved(self, patch_open, tmp_path):
        ds = sleep_dataset_from_paths(["fake.edf"], ch_names=CH_NAMES, L=1)
        sdc_path = str(tmp_path / "test")
        ds.save_to_sdc(sdc_path)

        ds2 = sleep_dataset_from_paths([], sdcFile=sdc_path)
        assert len(ds2) == len(ds)
