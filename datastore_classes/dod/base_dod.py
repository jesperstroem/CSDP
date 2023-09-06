import os
from h5py import File
from abc import abstractmethod

import sys
sys.path.append('../DataClasses')
from DataClasses.base import SleepdataPipeline


class Base_DOD(SleepdataPipeline):
    """
    ABOUT THIS DATASET 
    
    """
    def label_mapping(self):
        return {
            -1: self.Labels.UNKNOWN,
            0: self.Labels.Wake,
            1: self.Labels.N1,
            2: self.Labels.N2,
            3: self.Labels.N3,
            4: self.Labels.REM
        }
  

    def sample_rate(self):
        return 250 # We assume this due to what can be seen in the DOD code. USleep says 256 though..
    
    
    @property
    @abstractmethod
    def dataset_name(self):
        pass
    
    
    def channel_mapping(self):
        return {
            "C3_M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4_M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "F4_F4": self.Mapping(self.TTRef.F3, self.TTRef.F4),
            "F3_M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F3_O1": self.Mapping(self.TTRef.F3, self.TTRef.O1),
            "F4_O2": self.Mapping(self.TTRef.F4, self.TTRef.O2),
            "O1_M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2_M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "EOG1": self.Mapping(self.TTRef.EL, self.TTRef.RPA), # TODO: Find out refs
            "EOG2": self.Mapping(self.TTRef.ER, self.TTRef.RPA), # TODO: Find out refs
        }
    
    
    def list_records(self, basepath):
        paths_dict = dict()
        
        for dir, subdir, filenames in os.walk(basepath):
            for file in filenames:
                record_no = file.split(".")[0]
                record_path = f"{dir}/{file}"
                
                paths_dict[record_no] = [(record_path, )]
                
        return paths_dict
    
    
    def read_psg(self, record):
        x = dict()

        record = record[0]
        
        try:        
            with File(record, "r") as h5:
                signals = h5.get("signals")
                eeg_channels = signals.get("eeg")
                eog_channels = signals.get("eog")
            
                channel_len = len(eeg_channels.get(list(eeg_channels.keys())[0]))
                x_num_epochs = int(channel_len/self.sample_rate()/30)
                        
                for channel in eeg_channels:
                    x[channel] = (eeg_channels.get(channel)[()], self.sample_rate())
                for channel in eog_channels:
                    x[channel] = (eog_channels.get(channel)[()], self.sample_rate())
            
                y = list(h5.get("hypnogram")[()])
            
                assert(len(y) == x_num_epochs), "Length of signal does not match the number of labels."
        except:
            self.log_info("Could not read record", record=record)
            return None

        return x, y
