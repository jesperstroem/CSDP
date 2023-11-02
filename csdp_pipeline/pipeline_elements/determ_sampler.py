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

class Determ_sampler(IPipe):
    def __init__(self,
                 base_file_path, 
                 datasets,
                 split_type, 
                 num_epochs, 
                 split_file = None,
                 subject_percentage = 1,
                 get_all_channels = False, 
                 eeg_picker_func = None,
                 eog_picker_func = None):
        self.base_file_path = base_file_path
        self.datasets = datasets
        self.split_type = split_type
        self.split_file = split_file
        self.subject_percentage = subject_percentage
        self.records = self.list_records()
        self.epoch_length = num_epochs
        self.get_all_channels = get_all_channels

        self.eeg_picker_func = eeg_picker_func        
        self.eog_picker_func = eog_picker_func

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
                            print("Could not find configured split")
                            exit()
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

    def __pick_first_available_channel(self, channels, type):
        channels = [x for x in channels if x.startswith(type)]

        return channels[0]

    def __get_sample(self, index):
        r = self.records[index]

        dataset = r[0]
        subject = r[1]
        rec = r[2]

        with h5py.File(f"{self.base_file_path}/{dataset}.hdf5", "r") as hdf5:
            y = hdf5[subject][rec]["hypnogram"][()]

            psg_channels = list(hdf5[subject][rec]["psg"].keys())

            if self.get_all_channels:
                eeg_data, eog_data, eeg_tag, eog_tag = self.__load_all_channels(hdf5, subject, rec, psg_channels)
            else:
                eeg_data, eog_data, eeg_tag, eog_tag = self.__load_single_channel(hdf5, subject, rec, psg_channels)

        tag = f"{dataset}/{subject}/{rec}/{eeg_tag}, {eog_tag}"
        y = torch.tensor(y)

        return eeg_data, eog_data, y, tag
    
    def __load_all_channels(self, hdf5, subject, rec, psg_channels):
        eeg_data = []
        eog_data = []

        eeg_keys = [x for x in psg_channels if x.startswith("EEG")]
        eog_keys = [x for x in psg_channels if x.startswith("EOG")]

        for ch in eeg_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eeg_data.append(data)
            eeg_tag = "all"

        for ch in eog_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eog_data.append(data)
            eog_tag = "all"
        
        eeg_data = np.array(eeg_data)
        eog_data = np.array(eog_data)
        eeg_data = torch.Tensor(eeg_data)
        eog_data = torch.Tensor(eog_data)

        return eeg_data, eog_data, eeg_tag, eog_tag

    def __load_single_channel(self, hdf5, subject, rec, psg_channels):
        try:
            eeg = self.eeg_picker_func(psg_channels) if self.eeg_picker_func != None else self.__pick_first_available_channel(psg_channels, "EEG")
            
            eeg_data = hdf5[subject][rec]["psg"][eeg][:]
            eeg_tag = eeg   
        except:
            eeg_data = []
            eeg_tag = "none"

        try:
            eog = self.eog_picker_func(psg_channels) if self.eog_picker_func != None else self.__pick_first_available_channel(psg_channels, "EOG")

            eog_data = hdf5[subject][rec]["psg"][eog][:]
            eog_tag = eog
        except:
            eog_data = []
            eog_tag = "none" 

        eeg_data = torch.Tensor(eeg_data)
        eog_data = torch.Tensor(eog_data)
        eeg_data = eeg_data.unsqueeze(0)
        eog_data = eog_data.unsqueeze(0)

        return eeg_data, eog_data, eeg_tag, eog_tag