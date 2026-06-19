"""Tests for pipeline data models — no data files required."""

import os
import tempfile

import torch

from csdp_pipeline.pipeline_elements.models import Dataset_Split, ISample, ITag, Split

# ── Dataset_Split ─────────────────────────────────────────────────────────


class TestDatasetSplit:
    def test_get_subjects_train(self):
        ds = Dataset_Split("mock.hdf5", train=["s1", "s2"], val=["s3"], test=["s4"])
        assert ds.get_subjects_from_string("train") == ["s1", "s2"]

    def test_get_subjects_val(self):
        ds = Dataset_Split("mock.hdf5", train=[], val=["s3"], test=[])
        assert ds.get_subjects_from_string("val") == ["s3"]

    def test_get_subjects_test(self):
        ds = Dataset_Split("mock.hdf5", train=[], val=[], test=["s5", "s6"])
        assert ds.get_subjects_from_string("test") == ["s5", "s6"]


# ── Split ─────────────────────────────────────────────────────────────────


class TestSplit:
    def _make_split(self):
        ds = Dataset_Split("mock.hdf5", train=["s1"], val=["s2"], test=["s3"])
        return Split(id="fold_0", dataset_splits=[ds], base_data_path="/data")

    def test_get_dict_has_datasets_key(self):
        split = self._make_split()
        d = split.get_dict()
        assert "datasets" in d
        assert "base_data_path" in d

    def test_get_dict_dataset_has_all_splits(self):
        split = self._make_split()
        d = split.get_dict()
        ds_entry = d["datasets"]["mock.hdf5"]
        assert "train" in ds_entry
        assert "val" in ds_entry
        assert "test" in ds_entry

    def test_dump_and_reload_via_file(self):
        split = self._make_split()
        with tempfile.TemporaryDirectory() as tmpdir:
            split.dump_file(tmpdir)
            json_path = os.path.join(tmpdir, "fold_0.json")
            assert os.path.exists(json_path)
            reloaded = Split.file(json_path)
        assert reloaded.id == "fold_0"
        assert reloaded.base_data_path == "/data"
        assert len(reloaded.dataset_splits) == 1

    def test_repr_contains_id(self):
        split = self._make_split()
        assert "fold_0" in repr(split)


# ── ISample / ITag ────────────────────────────────────────────────────────


class TestISample:
    def test_index_stored(self):
        s = ISample(42)
        assert s.index == 42

    def test_fields_assignable(self):
        s = ISample(0)
        s.eeg = torch.zeros(1, 10)
        s.eog = torch.zeros(1, 10)
        s.labels = torch.zeros(3, dtype=torch.long)
        s.tag = ITag("ds", "sub1", "rec1", ["EEG_C3-M2"], ["EOG_L-M2"])
        assert s.eeg.shape == (1, 10)
        assert s.tag.dataset == "ds"


class TestITag:
    def test_default_construction(self):
        tag = ITag()
        assert tag.dataset == ""
        assert tag.subject == ""
        assert tag.record == ""

    def test_full_construction(self):
        tag = ITag("ds", "sub1", "rec1", ["EEG_C3-M2"], ["EOG_L-M2"], 0, 3840)
        assert tag.eeg == ["EEG_C3-M2"]
        assert tag.start_idx == 0
        assert tag.end_idx == 3840
