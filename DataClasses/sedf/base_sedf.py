import mne
from abc import abstractmethod

import sys
sys.path.append('../DataClasses')
from DataClasses.base import SleepdataPipeline


class Base_Sedf(SleepdataPipeline):
    """
    ABOUT THIS DATASET
    
    The naming convention of records is as follows: SC4ssNE0 where:
    SC: Sleep Cassette study.
    ss: Subject number (notice that most subjects has 2 records), e.g. 00.
    N: Night (1 or 2).
    
    Channels included in dataset: ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker'].

    
    EEG and EOG signals were each sampled at 100Hz.
    """        
    def label_mapping(self):
        return {
            "Sleep stage W": self.Labels.Wake,
            "Sleep stage 1": self.Labels.N1,
            "Sleep stage 2": self.Labels.N2,
            "Sleep stage 3": self.Labels.N3,
            "Sleep stage 4": self.Labels.N3,
            "Sleep stage R": self.Labels.REM,
            "Sleep stage ?": self.Labels.UNKNOWN,
            "Movement time": self.Labels.UNKNOWN
        }
    
  
    def sample_rate(self):
        return 100
        
        
    @property
    @abstractmethod    
    def dataset_name(self):
        pass
    
    
    def channel_mapping(self):
        return {
            "EOG horizontal": self.Mapping(self.TTRef.EL, self.TTRef.ER), 
            "EEG Fpz-Cz": self.Mapping(self.TTRef.Fpz, self.TTRef.Cz),
            "EEG Pz-Oz": self.Mapping(self.TTRef.Pz, self.TTRef.Oz)
        }
    
    
    @abstractmethod    
    def list_records(self):
        pass
    
    @abstractmethod
    def read_psg(self, record):
        pass