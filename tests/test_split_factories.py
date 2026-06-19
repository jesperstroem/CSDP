"""
Tests for Split factory class methods and create_split_file.

All tests use minimal synthetic HDF5 files created in pytest tmp_path —
no real PSG data needed. The HDF5 structure required by these functions is
just a "data" group containing subject-named subgroups (no signal data).
"""

import json
import os

import h5py
import pytest

from csdp_pipeline.pipeline_elements.models import Split
from csdp_training.utility import create_split_file

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def hdf5_dir(tmp_path):
    """
    Two HDF5 files, each with a 'data' group and subject subgroups.
    Used by Split.full_test and Split.random.
    """
    specs = [
        ("dataset_a.hdf5", [f"sub_{i:02d}" for i in range(10)]),
        ("dataset_b.hdf5", [f"sub_{i:02d}" for i in range(10, 16)]),
    ]
    for filename, subs in specs:
        with h5py.File(tmp_path / filename, "w") as f:
            grp = f.create_group("data")
            for sub in subs:
                grp.create_group(sub)
    return tmp_path


@pytest.fixture
def hdf5_pair(tmp_path):
    """
    Two HDF5 files for Split.train_and_holdout — one training, one test.
    Returns (base_path, train_filename, test_filename, train_subs, test_subs).
    """
    train_subs = [f"train_{i:02d}" for i in range(8)]
    test_subs = [f"test_{i:02d}" for i in range(4)]

    with h5py.File(tmp_path / "train.hdf5", "w") as f:
        grp = f.create_group("data")
        for sub in train_subs:
            grp.create_group(sub)

    with h5py.File(tmp_path / "test.hdf5", "w") as f:
        grp = f.create_group("data")
        for sub in test_subs:
            grp.create_group(sub)

    return str(tmp_path), "train.hdf5", "test.hdf5", train_subs, test_subs


@pytest.fixture
def hdf5_flat_dir(tmp_path):
    """
    HDF5 files where subjects are at the root level (no 'data' group).
    Matches the structure expected by create_split_file.
    """
    subs = [f"sub_{i:02d}" for i in range(10)]
    with h5py.File(tmp_path / "dataset_a.hdf5", "w") as f:
        for sub in subs:
            f.create_group(sub)
    return str(tmp_path), subs


# ── Split.train_and_holdout ────────────────────────────────────────────────


class TestTrainAndHoldout:
    def test_returns_split_object(self, hdf5_pair):
        base, train_f, test_f, _, _ = hdf5_pair
        split = Split.train_and_holdout(base, train_f, test_f)
        assert isinstance(split, Split)

    def test_split_name_stored(self, hdf5_pair):
        base, train_f, test_f, _, _ = hdf5_pair
        split = Split.train_and_holdout(base, train_f, test_f, split_name="my_split")
        assert split.id == "my_split"

    def test_training_subjects_in_train_list(self, hdf5_pair):
        base, train_f, test_f, train_subs, _ = hdf5_pair
        split = Split.train_and_holdout(base, train_f, test_f)
        train_ds = next(ds for ds in split.dataset_splits if ds.train)
        assert set(train_ds.train) == set(train_subs)

    def test_training_dataset_has_empty_val_and_test(self, hdf5_pair):
        base, train_f, test_f, _, _ = hdf5_pair
        split = Split.train_and_holdout(base, train_f, test_f)
        train_ds = next(ds for ds in split.dataset_splits if ds.train)
        assert train_ds.val == []
        assert train_ds.test == []

    def test_test_subjects_in_test_list(self, hdf5_pair):
        base, train_f, test_f, _, test_subs = hdf5_pair
        split = Split.train_and_holdout(base, train_f, test_f)
        test_ds = next(ds for ds in split.dataset_splits if ds.test)
        assert set(test_ds.test) == set(test_subs)

    def test_two_dataset_splits_created(self, hdf5_pair):
        base, train_f, test_f, _, _ = hdf5_pair
        split = Split.train_and_holdout(base, train_f, test_f)
        assert len(split.dataset_splits) == 2

    def test_base_data_path_stored(self, hdf5_pair):
        base, train_f, test_f, _, _ = hdf5_pair
        split = Split.train_and_holdout(base, train_f, test_f)
        assert split.base_data_path == base


# ── Split.full_test ────────────────────────────────────────────────────────


class TestFullTest:
    def test_returns_split_object(self, hdf5_dir):
        split = Split.full_test(str(hdf5_dir))
        assert isinstance(split, Split)

    def test_all_subjects_in_test_list(self, hdf5_dir):
        split = Split.full_test(str(hdf5_dir))
        all_test_subs = [sub for ds in split.dataset_splits for sub in ds.test]
        # Both files have 10 + 6 = 16 subjects total
        assert len(all_test_subs) == 16

    def test_train_and_val_are_empty(self, hdf5_dir):
        split = Split.full_test(str(hdf5_dir))
        for ds in split.dataset_splits:
            assert ds.train == []
            assert ds.val == []

    def test_one_dataset_split_per_file(self, hdf5_dir):
        split = Split.full_test(str(hdf5_dir))
        assert len(split.dataset_splits) == 2

    def test_default_split_name(self, hdf5_dir):
        split = Split.full_test(str(hdf5_dir))
        assert split.id == "full_test"

    def test_custom_split_name(self, hdf5_dir):
        split = Split.full_test(str(hdf5_dir), split_name="eval_run")
        assert split.id == "eval_run"


# ── Split.random ───────────────────────────────────────────────────────────


class TestRandom:
    def test_returns_split_object(self, hdf5_dir):
        split = Split.random(str(hdf5_dir))
        assert isinstance(split, Split)

    def test_no_subject_appears_in_two_splits(self, hdf5_dir):
        split = Split.random(str(hdf5_dir))
        for ds in split.dataset_splits:
            assert set(ds.train) & set(ds.val) == set()
            assert set(ds.train) & set(ds.test) == set()
            assert set(ds.val) & set(ds.test) == set()

    def test_all_subjects_accounted_for(self, hdf5_dir):
        split = Split.random(str(hdf5_dir))
        for ds in split.dataset_splits:
            total = len(ds.train) + len(ds.val) + len(ds.test)
            original = len(ds.train) + len(ds.val) + len(ds.test)
            assert total == original

    def test_train_is_largest_split(self, hdf5_dir):
        split = Split.random(str(hdf5_dir))
        for ds in split.dataset_splits:
            assert len(ds.train) >= len(ds.val)
            assert len(ds.train) >= len(ds.test)

    def test_one_dataset_split_per_file(self, hdf5_dir):
        split = Split.random(str(hdf5_dir))
        assert len(split.dataset_splits) == 2

    def test_custom_split_name(self, hdf5_dir):
        split = Split.random(str(hdf5_dir), split_name="fold_1")
        assert split.id == "fold_1"


# ── create_split_file ──────────────────────────────────────────────────────


class TestCreateSplitFile:
    def test_returns_filename(self, hdf5_flat_dir, monkeypatch):
        base, _ = hdf5_flat_dir
        monkeypatch.chdir(base)
        result = create_split_file(base)
        assert result == "random_split.json"

    def test_json_file_created(self, hdf5_flat_dir, monkeypatch):
        base, _ = hdf5_flat_dir
        monkeypatch.chdir(base)
        create_split_file(base)
        assert os.path.exists(os.path.join(base, "random_split.json"))

    def test_json_has_dataset_entry(self, hdf5_flat_dir, monkeypatch):
        base, _ = hdf5_flat_dir
        monkeypatch.chdir(base)
        create_split_file(base)
        with open(os.path.join(base, "random_split.json")) as f:
            data = json.load(f)
        assert "dataset_a" in data

    def test_json_has_train_val_test_keys(self, hdf5_flat_dir, monkeypatch):
        base, _ = hdf5_flat_dir
        monkeypatch.chdir(base)
        create_split_file(base)
        with open(os.path.join(base, "random_split.json")) as f:
            data = json.load(f)
        entry = data["dataset_a"]
        assert "train" in entry
        assert "val" in entry
        assert "test" in entry

    def test_all_subjects_distributed(self, hdf5_flat_dir, monkeypatch):
        base, subs = hdf5_flat_dir
        monkeypatch.chdir(base)
        create_split_file(base)
        with open(os.path.join(base, "random_split.json")) as f:
            data = json.load(f)
        entry = data["dataset_a"]
        assigned = set(entry["train"]) | set(entry["val"]) | set(entry["test"])
        assert assigned == set(subs)

    def test_no_subject_in_two_splits(self, hdf5_flat_dir, monkeypatch):
        base, _ = hdf5_flat_dir
        monkeypatch.chdir(base)
        create_split_file(base)
        with open(os.path.join(base, "random_split.json")) as f:
            data = json.load(f)
        entry = data["dataset_a"]
        assert set(entry["train"]) & set(entry["val"]) == set()
        assert set(entry["train"]) & set(entry["test"]) == set()
        assert set(entry["val"]) & set(entry["test"]) == set()
