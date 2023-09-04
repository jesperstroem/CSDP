import os
import mne
import wfdb
import h5py

import scipy.io
import numpy as np
import torch

from .base import SleepdataPipeline


class PHYS(SleepdataPipeline):

    def label_mapping(self):
        return {
            0: self.Labels.N1,
            1: self.Labels.N2,
            2: self.Labels.N3,
            3: self.Labels.REM,
            4: self.Labels.UNKNOWN,
            5: self.Labels.Wake,
        }
        
    def dataset_name(self):
        return "phys"
    
    def channel_mapping(self):
        return {
            "F3-M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA), 
            "F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "C3-M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "O1-M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "E1-M2": self.Mapping(self.TTRef.EL, self.TTRef.RPA)
        }
    
    def list_records(self, basepath):
        paths_dict = {}
        
        for subject in os.listdir(basepath):
            filebase = basepath+subject+'/'+subject
            
            data_path = filebase+'.hea'
            label_path = filebase+'-arousal.mat'
            
            exists = os.path.exists(label_path) and os.path.exists(data_path)
            
            if not exists:
                self.log_warning('The record did not exist', subject)
                continue
            
            paths_dict[subject] = [(data_path, label_path)]

        return paths_dict
    
    
    def read_psg(self, record):
        sample_rate = 200
        data_path, label_path = record
        
        try:
            data_path = data_path.rstrip('.hea')
            r = wfdb.rdrecord(data_path)
        except ValueError:
            self.log_error("Could not read data file", subject=None, record=data_path)
            return None

        with h5py.File(label_path, 'r') as f:
            # Labels
            s1 = f['data']['sleep_stages']['nonrem1'][()].flatten()
            s2 = f['data']['sleep_stages']['nonrem2'][()].flatten()
            s3 = f['data']['sleep_stages']['nonrem3'][()].flatten()
            rem = f['data']['sleep_stages']['rem'][()].flatten()
            udf = f['data']['sleep_stages']['undefined'][()].flatten()
            w = f['data']['sleep_stages']['wake'][()].flatten()
            
            #The labels are boolean masks for every sample in the signal
            #We stack these together
            stacked = np.stack([s1,s2,s3,rem,udf,w])
            
            #Since only one of the masks is true for a given point in the signal, we can argmax it
            y = np.argmax(stacked, axis=0)
            
            #The masks are always the same value for 6000 samples in a row, because samplerate=200 --> 200*30=6000
            #So we take every 6000th value in the original array and get the final labels. XD OMEGALUL
            y = y[0::5999]
            
            # Theres always a label for an incomplete epoch of data, so we remove it
            y = y[:-1]
            
        # Data    
        dic = dict()
        dataframe = r.to_dataframe()

        for ch in self.channel_mapping().keys():
            data = dataframe[ch].to_numpy()
            label_len = len(y)*sample_rate*30
            
            if len(data) < label_len:
                self.log_error("Not enough data for the amount of labels", subject=None, record=data_path)
                return None
            
            data = data[:label_len]
            
            assert len(data) == label_len, f"Datalength: {len(data)}, label length: {label_len}"
            
            dic[ch] = (data, sample_rate)
            
        return dic, y