# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import torch

import h5py
import math
import json

from csdp_pipeline.pipeline_elements.pipe import IPipe

class Determ_sampler(IPipe):
    def __init__(self, base_file_path, datasets, split_file, split_type, num_epochs, subject_percentage = 1, sample_rate = 128, channel_picker_func = None):
        self.base_file_path = base_file_path
        self.datasets = datasets
        self.split_type = split_type
        self.split_file = split_file
        self.subject_percentage = subject_percentage
        self.records = self.list_records()
        self.epoch_length = num_epochs
        self.sample_rate = sample_rate

        self.channel_picker_func = channel_picker_func           

    def process(self, index):
        sample = self.__get_sample(index)

        return sample

    def list_records(self):
        list_of_records = []

        for f in self.datasets:
            with h5py.File(f"{self.base_file_path}/{f}.hdf5", "r") as hdf5:
                with open(self.split_file, "r") as splitfile:
                    splitdata = json.load(splitfile)

                    try:
                        sets = splitdata[f]
                        subjects = sets[self.split_type]
                    except:
                        subjects = list(hdf5.keys())

                num_subjects = len(subjects)
                num_subjects_to_use = math.ceil(num_subjects*self.subject_percentage)
                subjects = subjects[0:num_subjects_to_use]

                if len(subjects) == 0:
                    raise ValueError(f"No subjects in split type: {self.split_type}")

                for s in subjects:
                    try:
                        records = list(hdf5[s])
                    except:
                        print(f"Did not find subject {s} in dataset {f} for splittype {self.split_type}")
                        continue

                    for r in records:
                        list_of_records.append((f,s,r))

        return list_of_records

    def __pick_first_available_channel_pair(self, channels):
        eegs = [x for x in channels if x.startswith("EEG")]
        eogs = [x for x in channels if x.startswith("EOG")]

        return eegs[0], eogs[0]

    def __get_sample(self, index):
        r = self.records[index]

        dataset = r[0]
        subject = r[1]
        rec = r[2]

        with h5py.File(f"{self.base_file_path}/{dataset}.hdf5", "r") as hdf5:
            y = hdf5[subject][rec]["hypnogram"][()]

            psg_channels = list(hdf5[subject][rec]["psg"].keys())

            eeg, eog = self.channel_picker_func(psg_channels) if self.channel_picker_func != None else self.__pick_first_available_channel_pair(psg_channels)

            if eeg != None: 
                eeg_data = hdf5[subject][rec]["psg"][eeg][:]
            else:
                eeg_data = []

            if eog != None:
                eog_data = hdf5[subject][rec]["psg"][eog][:]
            else:
                eog_data = []

        eeg_data = torch.Tensor(eeg_data)
        eog_data = torch.Tensor(eog_data)

        eeg_data = eeg_data.unsqueeze(0)
        eog_data = eog_data.unsqueeze(0)

        tag = f"{dataset}/{subject}/{rec}/{eeg}, {eog}"
        y = torch.tensor(y)

        return eeg_data, eog_data, y, tag