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

from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../..'))
from shared.pipeline.pipe import IPipe
from shared.pipeline.pipeline_dataset import PipelineDataset


class Sampler(IPipe):
    def __init__(self, base_file_path, datasets, split_file_path, split_type, num_epochs, subject_percentage = 1):
        self.base_file_path = base_file_path
        self.datasets = datasets
        self.split_type = split_type
        self.split_file = split_file_path
        self.subject_percentage = subject_percentage
        self.subjects, self.num_records = self.__list_files()
        self.probs = self.calc_probs()
        self.epoch_length = num_epochs
        
    def process(self, index):
        success = False
            
        while not success:
            sample = self.__get_sample()
            
            if sample != None:
                success = True

        return sample
    
    def calc_probs(self):
        total_num_datasets = len(self.datasets)
        total_num_records = sum(self.num_records)
        
        probs = []
        s = []
        
        for i, f in enumerate(self.datasets):
            num_records = self.num_records[i]
            
            strat_prob = num_records/total_num_records
            dis_prob = 1 / total_num_datasets
            
            prob_d = 0.5 * strat_prob + 0.5*dis_prob
            probs.append(prob_d)

        return probs
    
    def __get_sample(self):
        
        possible_sets = self.datasets
        probs = self.probs
        
         # Choose random dataset
        r_dataset = np.random.choice(possible_sets, 1, p=probs)[0]
        index = possible_sets.index(r_dataset)
 
        with h5py.File(f"{self.base_file_path}/{r_dataset}.hdf5", "r") as hdf5:
            subjects = self.subjects[index]
 
            if len(subjects) == 0:
                    raise ValueError(f"No subjects in split type: {self.split_type}")
            
            # Choose random subject
            r_subject = np.random.choice(subjects, 1)[0]
            records = list(hdf5[r_subject])
            
            #choose Random record
            r_record = np.random.choice(records, 1)[0]
            hyp = hdf5[r_subject][r_record]["hypnogram"][()]
            psg = list(hdf5[r_subject][r_record]["psg"])
            
            #Choose random eeg and eog
            eegs = [x for x in psg if x.startswith("EEG")]
            eogs = [x for x in psg if x.startswith("EOG")]
            
            # Eog not always available, skip if so
            try:
                r_eog = np.random.choice(eogs, 1)[0]
                r_eeg = np.random.choice(eegs, 1)[0]
            except ValueError:
                return None
            
            # Choose random index of a random label
            label_set = np.unique(hyp)

            r_label = np.random.choice(label_set, 1)[0]
    
            indexes = [i for i in range(len(hyp)) if hyp[i] == r_label]
            r_index = np.random.choice(indexes, 1)[0]
            
            # Randomly shift the position of the random label index
            r_shift = np.random.choice(list(range(0,self.epoch_length)), 1)[0]
            
            assert r_shift <= 200
            
            start_index = r_index-r_shift
            
            if start_index < 0:
                start_index = 0
            elif (start_index + self.epoch_length) >= len(hyp):
                start_index = len(hyp) - self.epoch_length
            
            y = hyp[start_index:start_index+self.epoch_length]
            
            y = torch.tensor(y)
            
            x_start_index = start_index*128*30
            
            # Pick out the segment data from both channels
            eeg_chnl = hdf5[r_subject][r_record]["psg"][r_eeg]

            eog_chnl = hdf5[r_subject][r_record]["psg"][r_eog]

            r_eeg_segment = eeg_chnl[x_start_index:x_start_index+(self.epoch_length*30*128)]
            r_eog_segment = eog_chnl[x_start_index:x_start_index+(self.epoch_length*30*128)]
            
            x_eeg = torch.tensor(r_eeg_segment)
            x_eog = torch.tensor(r_eog_segment)
            
            x_eeg = x_eeg.unsqueeze(0)
            x_eog = x_eog.unsqueeze(0)
            
            # Create a tag for debugging purposes
            tag = f"{r_dataset}/{r_subject}/{r_record}/{r_eeg}, {r_eog}/{x_start_index}-{x_start_index+(self.epoch_length*30*128)}"

        return x_eeg, x_eog, y, tag
    
    def __list_files(self):
        subjects = []
        num_records = []
        base_path = self.base_file_path
        
        for f in self.datasets:
            with h5py.File(base_path+"/"+f"{f}.hdf5", "r") as hdf5:
                
                with open(self.split_file, "r") as splitfile:
                    splitdata = json.load(splitfile)
                    try:
                        # Trý finding the correct split
                        sets = splitdata[f]
                        subs = sets[self.split_type]
                    except:
                        # If none is configured, take all subjects
                        subs = list(hdf5.keys())
                    
                num_subjects = len(subs)
                num_subjects_to_use = math.ceil(num_subjects*self.subject_percentage)
                subs = subs[0:num_subjects_to_use]

                tot_records = 0
                subjects_to_add = []
                
                for subj_key in subs:
                    try:
                        subj = hdf5[subj_key]
                    except:
                        print(f"Did not find subject {subj_key} in dataset {f} for splittype {self.split_type}")
                        continue
                        
                    records = len(subj.keys())
                    tot_records += records
                    subjects_to_add.append(subj_key)
                
                num_records.append(tot_records)
                subjects.append(subjects_to_add)

        return subjects, num_records
                
if __name__ == '__main__':
    s = Sampler("/home/alec/repos/data/hdf5",
                ["dod-h"],
                "/home/jose/repo/Speciale2023/shared/usleep_split.json",
                split_type="test",
                num_epochs=200,
                subject_percentage=1.0)
    
    while True:
        x_eeg, x_eog ,y,tag = s.process(1)
        print(tag)
