import os
from h5py import File
import scipy.io
import numpy as np
import pandas as pd
import mne

from .base import SleepdataPipeline

class SVUH(SleepdataPipeline):
    """
    ABOUT THIS DATASET 
    
    """
  
    def sample_rate(self):
        return 128
        
        
    def dataset_name(self):
        return "svuh"
    
    
    def label_mapping(self):
        return {
            "0": self.Labels.Wake,
            "1": self.Labels.REM,
            "2": self.Labels.N1,
            "3": self.Labels.N2,
            "4": self.Labels.N3,
            "5": self.Labels.N3, # Stage 4 in SVUH is same as N3
            "6": self.Labels.UNKNOWN,
            "7": self.Labels.UNKNOWN,
            "8": self.Labels.UNKNOWN
        }
    
    
    def channel_mapping(self):
        return {
            "Lefteye": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "RightEye": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "C3A2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4A1": self.Mapping(self.TTRef.C4, self.TTRef.LPA)
        }
    
    
    def list_records(self, basepath):
        basepath = basepath + 'files/'
        file_base = "ucddb"
        file_path = basepath+'/'+file_base
        subject_ids = ["002","003","005","006","007","008","009","010",
                       "011","012","013","014","015","017","018","019",
                      "020","021","022","023","024","025","026","027","028"]
        
        dic = dict()
        
        for id in subject_ids:
            prepend = file_path+id
            
            if os.path.isfile(prepend+".rec"):
                self.log_info('Renamed file {} to .edf'.format(prepend+".rec"))
                os.rename(prepend+".rec", prepend+".edf")
                
            dic[id] = [(prepend+".edf", prepend+"_stage.txt")]
            
        return dic
    
    def read_psg(self, record):
        (datapath, labelpath) = record
        
        data = mne.io.read_raw_edf(datapath, verbose=False)

        dic = dict()
        
        with open(labelpath, 'rb') as f:
            y = list(map(lambda x: chr(x[0]), f.readlines()))
        
        x_len = len(y)*self.sample_rate()*30
        
        for channel in self.channel_mapping().keys():
            channel_data = data[channel]
            relative_channel_data = channel_data[0][0]
            dic[channel] = (relative_channel_data[:x_len], self.sample_rate())
        
        return dic, y