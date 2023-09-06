import os
from h5py import File
import scipy.io
import numpy as np
from abc import abstractmethod

import sys
sys.path.append('../DataClasses')
from DataClasses.base import SleepdataPipeline

class Isruc_base(SleepdataPipeline):
    """
    ABOUT THIS DATASET 
    
    """
  
    def sample_rate(self):
        return 200
        
    @property
    @abstractmethod    
    def dataset_name(self):
        pass
    
    
    def label_mapping(self):
        return {
            "0": self.Labels.Wake,
            "1": self.Labels.N1,
            "2": self.Labels.N2,
            "3": self.Labels.N3,
            "5": self.Labels.REM,
        }
    
    
    def channel_mapping(self):
        return {
            "F3_A2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "C3_A2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "F4_A1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "C4_A1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "O1_A2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2_A1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "ROC_A1": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "LOC_A2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
        }
    
    def list_records(self, basepath):
        paths_dict = {}
        
        record_paths = os.listdir(basepath)
        
        for path in record_paths:
            # Fucking ugly and hacky, delete ASAP
            if "ipynb_checkpoints" in path:
                continue
            
            recordpath = basepath+path+'/'
            datapath = recordpath+"subject"+path+".mat"
            labelpath = recordpath+path+'_'+"1.txt"
            
            paths_dict[path] = [(datapath, labelpath)]
        
        return paths_dict
    
    
    def read_psg(self, record):
        datapath, labelpath = record
        
        x = dict()
        
        mat = scipy.io.loadmat(datapath)
        for key in self.channel_mapping().keys():
            # 30 epochs of data was removed due to noise resulting in more labels
            chnl = np.array(mat[key]).flatten()
            x_len = len(chnl)
            x[key] = (chnl, self.sample_rate())
            
        with open(labelpath, "r") as f:
            y = list(map(lambda x: x[0], f.readlines()))
            y_trunc = y[:int(x_len/self.sample_rate()/30)]
            trunc_len = len(y)-len(y_trunc)
            if trunc_len > 31:
                self.log_warning(f"Length of truncated y was: {trunc_len}.", subject=None, record=labelpath)
                return None
            
        return x, y_trunc
        