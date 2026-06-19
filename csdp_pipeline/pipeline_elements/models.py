import json
import os

import h5py
import torch
from sklearn.model_selection import train_test_split


class Dataset_Split:
    dataset_filepath: str
    train: list[str]
    val: list[str]
    test: list[str]

    def __init__(self, dataset_filepath: str, train: list[str] = [], val: list[str] = [], test: list[str] = []):
        self.dataset_filepath = dataset_filepath
        self.train = train
        self.val = val
        self.test = test

    def get_subjects_from_string(self, str):
        if str == "train":
            return self.train
        elif str == "val":
            return self.val
        else:
            return self.test


class Split:
    id: str
    dataset_splits: list[Dataset_Split]
    base_data_path: str

    @classmethod
    def file(cls, split_file_path):
        """Function to create a split object from an existing split .json file.

        Args:
            split_file_path (string): Exact filepath to an existing split file.

        Returns:
            Split: A split object defined by the given split file
        """

        dataset_splits = []

        with open(split_file_path) as f:
            data = json.load(f)

        base_hdf5_path = data["base_data_path"]
        id = os.path.basename(split_file_path.rstrip(".json"))
        inner_data = data["datasets"]

        datasets = list(map(lambda x: x[0], inner_data.items()))

        for dset in datasets:
            s = Dataset_Split(
                f"{base_hdf5_path}/{dset}",
                train=inner_data[dset]["train"],
                val=inner_data[dset]["val"],
                test=inner_data[dset]["test"],
            )

            dataset_splits.append(s)

        return cls(id=id, dataset_splits=dataset_splits, base_data_path=base_hdf5_path)

    @classmethod
    def train_and_holdout(cls, base_data_path, training_hdf5_path, test_hdf5_path, split_name="train_and_holdout"):
        """Function to create a split object based on two HDF5 datasets, where one is the training set, and the other is the hold-out testset

        Args:
            base_data_path (string): Base-path to the HDF5 data files
            training_hdf5_path (string): Exact filepath to the training HDF5 file
            test_hdf5_path (string): Exact filepath to the hold-out test HDF5 file
            split_name (str, optional): Name of the split. Defaults to "train_and_holdout".

        Returns:
            Split: A split object
        """
        dataset_splits: list[Dataset_Split] = []

        train_path = f"{base_data_path}/{training_hdf5_path}"
        with h5py.File(train_path, "r") as hdf5:
            subs = list(hdf5["data"].keys())

            training_split = Dataset_Split(train_path, subs, [], [])

            dataset_splits.append(training_split)

        test_path = f"{base_data_path}/{test_hdf5_path}"
        with h5py.File(test_path, "r") as hdf5:
            subs = list(hdf5["data"].keys())

            test = Dataset_Split(test_path, [], [], subs)

            dataset_splits.append(test)

        return cls(id=split_name, dataset_splits=dataset_splits, base_data_path=base_data_path)

    @classmethod
    def full_test(cls, base_hdf5_path, split_name="full_test"):
        """Function to create a split, where all subjects are used for testing

        Args:
            base_hdf5_path (string): Base-path to the HDF5 files
            split_name (str, optional): Name of the split. Defaults to "full_test".

        Returns:
            Split: A split object
        """
        hdf5_paths = os.listdir(base_hdf5_path)
        hdf5_paths = [f"{base_hdf5_path}/{path}" for path in hdf5_paths]

        dataset_splits: list[Dataset_Split] = []

        for path in hdf5_paths:
            with h5py.File(path, "r") as hdf5:
                subs = list(hdf5["data"].keys())

                split = Dataset_Split(path, [], [], subs)

                dataset_splits.append(split)

        return cls(id=split_name, dataset_splits=dataset_splits, base_data_path=base_hdf5_path)

    @classmethod
    def random(cls, base_hdf5_path, split_name="random", split_percentages=(0.8, 0.1, 0.1)):
        """Function to create a random split from HDF5 data files

        Args:
            base_hdf5_path (string): Base-path to the HDF5 files
            split_name (str, optional): Name of the split. Defaults to "random".
            split_percentages (tuple, optional): Subject-based percentages for the split. Defaults to (0.8, 0.1, 0.1).

        Returns:
            Split: A split object
        """
        hdf5_paths = os.listdir(base_hdf5_path)
        hdf5_paths = [f"{base_hdf5_path}/{path}" for path in hdf5_paths]

        dataset_splits: list[Dataset_Split] = []

        for path in hdf5_paths:
            with h5py.File(path, "r") as hdf5:
                subs = list(hdf5["data"].keys())

                train, test = train_test_split(subs, test_size=1 - split_percentages[0])
                val, test = train_test_split(
                    test, test_size=split_percentages[2] / (split_percentages[2] + split_percentages[1])
                )

                split = Dataset_Split(path, train, val, test)

                dataset_splits.append(split)

        return cls(id=split_name, dataset_splits=dataset_splits, base_data_path=base_hdf5_path)

    def dump_file(self, path):
        """Function to save a .json file of this split object. Can be used with Split.file(...) to reload the split object.

        Args:
            path (string): Directory to save the .json file
        """
        dic = self.get_dict()

        with open(f"{path}/{self.id}.json", "w") as outfile:
            json.dump(dic, outfile)

    def __init__(self, id="", dataset_splits=[], base_data_path=""):
        self.id = id
        self.dataset_splits = dataset_splits
        self.base_data_path = base_data_path

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, dataset_splits={self.dataset_splits}, base_data_path={self.base_data_path})"

    def get_dict(self) -> dict:
        dic = dict()
        dic["datasets"] = {}

        for split in self.dataset_splits:
            dic["datasets"][os.path.basename(split.dataset_filepath)] = {
                "train": split.train,
                "val": split.val,
                "test": split.test,
            }

        dic["base_data_path"] = self.base_data_path

        return dic


class ITag:
    dataset: str
    subject: str
    record: str
    eeg: [str]
    eog: [str]
    start_idx: str
    end_idx: str

    def __init__(
        self, dataset: str = "", subject: str = "", record: str = "", eeg=[], eog=[], start_idx=-1, end_idx=-1
    ):
        self.dataset = dataset
        self.subject = subject
        self.record = record
        self.eeg = eeg
        self.eog = eog
        self.start_idx = start_idx
        self.end_idx = end_idx


class ISample:
    def __init__(self, index: int):
        self.index = index

    index: int
    eeg: torch.Tensor | None
    eog: torch.Tensor | None
    labels: torch.Tensor | None
    tag: ITag | None
