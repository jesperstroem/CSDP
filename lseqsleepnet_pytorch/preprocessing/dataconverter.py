# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

from .utils import create_spectrogram_images
from .storage import StorageAPI
import h5py
import os
import torch
import numpy as np

def get_class_sequence_idx(label):
    d = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4
    }

    return d[label]

'''
Class to convert a collection of datasets to spectrograms.
Assumes folder structure of root->datasets->subjects->records->data/labels
'''
class SleepDataConverter:
    def __init__(self, 
                 source_dir, 
                 target_dir,
                 win_size = 2,
                 fs_fourier = 100,
                 overlap = 1,
                 source_sample = 128,
                 target_sample = 100):
        self.source_root = source_dir
        self.target_root = target_dir
        self.win_size = win_size
        self.fs_fourier = fs_fourier
        self.source_sample = source_sample
        self.target_sample = target_sample
        self.overlap = overlap
    
    def convert(self):
        for dataset_dir in os.listdir(self.source_root):
            if ".hdf5" in dataset_dir:
                self.__process_dataset(self.source_root+'/'+dataset_dir, dataset_dir)
    
    def record(self, dataset, subject_index, record_index):
        folder_dir = self.__get_folder_dir(dataset, subject_index, record_index)
        
        return StorageAPI.load_data(folder_dir, 'spec')

    def __process_dataset(self, dataset_dir, dataset):
        with h5py.File(dataset_dir, "r") as f:
            print("Subjects: %s" % f.keys())
            for k in f.keys():
                self.__process_subject(f[k], dataset, k)

    def __process_subject(self, subject, dataset, subject_index):
        for k in subject.keys():
            record = subject[k]
            self.__process_record(record, dataset, subject_index, k)
            
    def __process_record(self, record, dataset, subject_index, record_index):
        data = record['psg']
        data = data[list(data.keys())[0]][()]
        labels = record['hypnogram'][()]
  
        _,_,spectrograms = create_spectrogram_images(data,
                                                     self.source_sample,
                                                     self.target_sample, 
                                                     self.win_size, 
                                                     self.fs_fourier, 
                                                     self.overlap)
        
        assert(len(spectrograms) == len(labels))
        
        spectrograms = torch.tensor(np.array(spectrograms))
        
        nan = torch.isnan(spectrograms)
        
        indices = nan.nonzero()

        self.__save_record((spectrograms, labels), dataset, subject_index, record_index)
    
    def __save_record(self, data, dataset, subject_index, record_index):
        folder_dir = self.__get_folder_dir(dataset, subject_index, record_index)
        
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        
        StorageAPI.save_data(folder_dir, 'spec', data)
        
        print('Record with path {} saved'.format(folder_dir))
    
    def __get_folder_dir(self, dataset, subject_index, record_index):
        folder_dir = '{root}/{dataset}/s{subject}/r{record}'.format(root=self.target_root,
                                                                    dataset=dataset,
                                                                    subject=subject_index,
                                                                    record=record_index)
        return folder_dir