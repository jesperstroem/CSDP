import math
import os
import random

import h5py
import numpy as np
import torch

from csdp_pipeline.pipeline_elements.models import Dataset_Split, ISample, ITag, Split
from csdp_pipeline.pipeline_elements.samplers import ISampler


class Full_Eval_Dataset_Sampler(ISampler):
    def __init__(self, split_data: Split, split_type: str = "val"):

        assert split_type == "val" or split_type == "test"

        self.split_type = split_type
        self.split_data = split_data
        self.samples = self.__get_data()
        self.num_samples = len(self.samples)

    def get_sample(self, index) -> ISample:
        return self.samples[index]

    def __read_dataset(self, dataset_split: Dataset_Split):
        dataset_path = dataset_split.dataset_filepath
        subs = dataset_split.val if self.split_type == "val" else dataset_split.test

        samples: list[ISample] = []

        with h5py.File(dataset_path, "r") as hdf5:
            for subj_key in subs:
                try:
                    subj = hdf5[subj_key]

                    rec_keys = subj.keys()

                    for rec_key in rec_keys:
                        rec = subj[rec_key]

                        hyp = rec["hypnogram"][()]
                        psg = rec["psg"]

                        psg_keys = psg.keys()
                        eeg_keys = list(filter(lambda x: "EEG" in x, psg_keys))
                        eog_keys = list(filter(lambda x: "EOG" in x, psg_keys))

                        eeg_data = []
                        eog_data = []

                        for c in eeg_keys:
                            channel_data = psg[c][()]

                            channel_data = torch.tensor(channel_data)

                            eeg_data.append(channel_data)

                        for c in eog_keys:
                            channel_data = psg[c][()]

                            channel_data = torch.tensor(channel_data)

                            eog_data.append(channel_data)

                        eeg_data = torch.stack(eeg_data, dim=0)
                        eog_data = torch.stack(eog_data, dim=0)

                        hyp = torch.tensor(hyp, dtype=torch.int64)

                        sample = ISample(index=0)
                        sample.eeg = eeg_data
                        sample.eog = eog_data
                        sample.labels = hyp
                        sample.tag = ITag(dataset=os.path.basename(dataset_path), subject=subj_key, record=rec_key)

                        samples.append(sample)

                except Exception:
                    print(
                        f"Did not find subject {subj_key} in dataset {dataset_path} with split type {self.split_type}"
                    )
                    continue

        return samples

    def __get_data(self):
        all_samples: list[ISample] = []

        for dataset_split in self.split_data.dataset_splits:
            all_samples.extend(self.__read_dataset(dataset_split))

        return all_samples


class Full_Train_Dataset_Sampler(ISampler):
    def __init__(self, window_size: int, splitdata: Split):

        self.window_length = window_size
        self.splitdata = splitdata
        self.eegs, self.eogs, self.hyp, self.window_counts, self.data_indexes = self.__get_data()

        self.num_samples = self.window_counts[-1]

    def get_sample(self, index) -> ISample:
        return self.__get_sample(index)

    def __get_sample(self, index):
        record_index = self.data_indexes[index]

        hyp = self.hyp[record_index]
        eeg_data = self.eegs[record_index]
        eog_data = self.eogs[record_index]

        # print(f"Third time: {time.time() - start}")
        num_epochs = hyp.shape[0]

        # Get how many windows we are offset for the given record
        window_offset = index - self.window_counts[record_index]

        # Calculate how many "extra" epochs there are for the given record
        rest_epochs = num_epochs - (math.floor(num_epochs / self.window_length) * self.window_length)

        # Calculate a random offset in epochs
        random_epoch_offset = random.randint(0, rest_epochs)

        # Calculate the first and last epoch to pick out for this sample
        y_start_idx = (window_offset * self.window_length) + random_epoch_offset
        y_end_idx = y_start_idx + self.window_length

        # Do the same, but for the data
        x_start_idx = y_start_idx * 30 * 128
        x_end_idx = x_start_idx + (self.window_length * 128 * 30)

        # Pick the data and return it.
        x_eeg = eeg_data[:, x_start_idx:x_end_idx]
        x_eog = eog_data[:, x_start_idx:x_end_idx]

        y_sample = hyp[y_start_idx:y_end_idx]

        sample = ISample(0)
        sample.eeg = x_eeg
        sample.eog = x_eog

        sample.tag = ITag(dataset="", subject="", record="", eeg="", eog="", start_idx=x_start_idx, end_idx=x_end_idx)

        sample.labels = y_sample

        return sample

    def __get_data(self):
        record_eegs = []
        record_eogs = []
        record_hyps = []
        window_count = [0]
        data_indexes = []

        record_counter = 0

        for split in self.splitdata.dataset_splits:
            file_path = split.dataset_filepath
            subs = split.train

            with h5py.File(file_path, "r") as hdf5:
                for subj_key in subs:
                    subj = hdf5[subj_key]

                    rec_keys = subj.keys()

                    for rec_key in rec_keys:
                        rec = subj[rec_key]

                        hyp = rec["hypnogram"][()]
                        psg = rec["psg"]

                        psg_keys = psg.keys()

                        eeg_keys = list(filter(lambda x: "EEG" in x, psg_keys))
                        eog_keys = list(filter(lambda x: "EOG" in x, psg_keys))

                        eeg_data = []
                        eog_data = []

                        for c in eeg_keys:
                            channel_data = psg[c][()]

                            whole_windows = math.floor(len(channel_data) / 128 / 30 / self.window_length)

                            eeg_data.append(channel_data)

                        for c in eog_keys:
                            channel_data = psg[c][()]

                            whole_windows = math.floor(len(channel_data) / 128 / 30 / self.window_length)

                            eog_data.append(channel_data)

                        record_eegs.append(torch.tensor(np.array(eeg_data)))
                        record_eogs.append(torch.tensor(np.array(eog_data)))

                        record_hyps.append(torch.tensor(hyp, dtype=torch.int64))

                        window_count.append(whole_windows + window_count[-1])

                        data_indexes = data_indexes + ([record_counter] * whole_windows)

                        record_counter += 1

        return record_eegs, record_eegs, record_hyps, window_count, data_indexes
