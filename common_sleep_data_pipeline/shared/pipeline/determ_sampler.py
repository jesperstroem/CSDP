# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import torch
import os
import sys
import numpy as np
import h5py
import math
import json
import time

sys.path.append(os.path.abspath('../..'))
from shared.pipeline.pipe import IPipe

class Determ_sampler(IPipe):
    def __init__(self, base_file_path, datasets, split_file, split_type, num_epochs, single_channels = False, subject_percentage = 1, sample_rate = 128):
        self.base_file_path = base_file_path
        self.datasets = datasets
        self.split_type = split_type
        self.split_file = split_file
        self.subject_percentage = subject_percentage
        self.records = self.list_records()
        self.epoch_length = num_epochs
        self.sample_rate = sample_rate
        self.single_channels = single_channels
        
    def process(self, index):
        #if index >= len(self.records):
        #    return -2
        
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
    
    def __find_eog(self, phrase):
        if 'EOG' in phrase:
            return phrase
    
    def __find_eeg(self, phrase):
        if 'EEG' in phrase:
            return phrase

    def __get_sample(self, index):
        r = self.records[index]

        dataset = r[0]
        subject = r[1]
        rec = r[2]

        eogs = []
        eegs = []

        #start = time.time()

        with h5py.File(f"{self.base_file_path}/{dataset}.hdf5", "r") as hdf5:
            y = hdf5[subject][rec]["hypnogram"][()]

            psg = hdf5[subject][rec]["psg"]

            if self.single_channels == False:
                for k in psg.keys():
                    if "EEG" in k:
                        eegs.append(psg[k][()])
                    if "EOG" in k:
                        eogs.append(psg[k][()])
                
                eegs = torch.tensor(np.array(eegs))
                eogs = torch.tensor(np.array(eogs))

            else:
                eeg = psg.visit(self.__find_eeg)
                eog = psg.visit(self.__find_eog)
                
                #start = time.time()
                eeg1 = psg[eeg][:]
                eog1 = psg[eog][:]

                eeg3 = torch.Tensor(eeg1)
                eog3 = torch.Tensor(eog1)

                eegs = eeg3.unsqueeze(0)
                eogs = eog3.unsqueeze(0)

        tag = f"{dataset}/{subject}/{rec}"
        y = torch.tensor(y)
        
        return eegs, eogs, y, tag
                
if __name__ == '__main__':
    s = Determ_sampler("/home/alec/repos/data/hdf5",
                ["dod-h"],
                "/home/jose/repo/Speciale2023/shared/usleep_split.json",
                split_type="val",
                num_epochs=200,
                subject_percentage=1.0)
    i=0
    while True:
        x_eeg, x_eog ,y,tag = s.process(i)
        print(tag)
        i+=1
