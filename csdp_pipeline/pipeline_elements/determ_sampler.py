# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import torch

import h5py
import math
import json
import numpy as np

from csdp_pipeline.pipeline_elements.pipe import IPipe
from csdp_pipeline.pipeline_elements.pipeline_dataset import PipelineDataset
from torch.utils.data import DataLoader

class Determ_sampler(IPipe):
    def __init__(self,
                 base_file_path, 
                 datasets,
                 split_type, 
                 num_epochs, 
                 split_file = None,
                 subject_percentage: float = 1.0,
                 get_all_channels = False):
        self.base_file_path = base_file_path
        self.datasets = datasets
        self.split_type = split_type
        self.split_file = split_file
        self.subject_percentage = subject_percentage
        self.records = self.list_records()
        print(f"Number of {split_type} records: {len(self.records)}")
        self.epoch_length = num_epochs
        self.get_all_channels = get_all_channels

    def process(self, index):
        sample = self.__get_sample(index)

        return sample

    def list_records(self):
        list_of_records = []

        for f in self.datasets:
            with h5py.File(f"{self.base_file_path}/{f}.hdf5", "r") as hdf5:
                
                if self.split_file != None:
                    with open(self.split_file, "r") as splitfile:
                        splitdata = json.load(splitfile)

                        try:
                            sets = splitdata[f]
                            subjects = sets[self.split_type]
                        except:
                            print(f"Could not find configured split for dataset {f} and splittype {self.split_type}. All subjects are sampled.")
                            subjects = list(hdf5.keys())
                else:
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

    def __get_sample(self, index):
        r = self.records[index]

        dataset = r[0]
        subject = r[1]
        rec = r[2]

        with h5py.File(f"{self.base_file_path}/{dataset}.hdf5", "r") as hdf5:
            y = hdf5[subject][rec]["hypnogram"][()]

            psg_channels = list(hdf5[subject][rec]["psg"].keys())

            eeg_data, eog_data, eeg_tag, eog_tag = self.__load_data(hdf5, subject, rec, psg_channels)
 
        tag = f"{dataset}/{subject}/{rec}/{eeg_tag}, {eog_tag}"
        y = torch.tensor(y)

        return eeg_data, eog_data, y, tag
    
    def determine_single_key(self, keys):
        if len(keys) > 0:
            key = keys[0]
            tag = key
            keys = [key]
        else:
            tag = "none"
        
        return keys, tag

    def __load_data(self, hdf5, subject, rec, psg_channels):
        eeg_data = []
        eog_data = []

        available_eeg_keys = [x for x in psg_channels if x.startswith("EEG")]
        available_eog_keys = [x for x in psg_channels if x.startswith("EOG")]

        if self.get_all_channels == False:
            eeg_keys, eeg_tag = self.determine_single_key(available_eeg_keys)
            eog_keys, eog_tag = self.determine_single_key(available_eog_keys)
        else:
            eeg_keys = available_eeg_keys
            eog_keys = available_eog_keys
            eeg_tag = "all"
            eog_tag = "all"

        for ch in eeg_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eeg_data.append(data)
            
        for ch in eog_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eog_data.append(data)

        eog_data = np.array(eog_data)
        eog_data = torch.Tensor(eog_data)

        eeg_data = np.array(eeg_data)
        eeg_data = torch.Tensor(eeg_data)
        
        return eeg_data, eog_data, eeg_tag, eog_tag
    
            
if __name__ == '__main__':
    s = Determ_sampler("C:/Users/au588953/hdf5",
                       ["abc"],
                       "val",
                       35,
                       None,
                       subject_percentage=1,
                       get_all_channels=False)
    
    dataset = PipelineDataset([s], 10)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    dataiter = iter(loader)
    
