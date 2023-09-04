import os
from h5py import File

from .base import SleepdataPipeline
import pandas as pd
import mne
import numpy as np

class EESM(SleepdataPipeline):
    """
    ABOUT THIS DATASET 
    """
    
    def label_mapping(self):
        return {
            1: self.Labels.Wake,
            2: self.Labels.REM,
            3: self.Labels.N1,
            4: self.Labels.N2,
            5: self.Labels.N3,
            6: self.Labels.UNKNOWN,
            7: self.Labels.UNKNOWN,
            8: self.Labels.UNKNOWN
        }
        
    def dataset_name(self):
        return "eesm"
    
    def left(self):
        return ["ELA","ELB", "ELC", "ELT", "ELE", "ELI"]
    
    def right(self):
        return ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI" ]
    
    def channel_mapping(self):
        return {
            "EEG": self.Mapping(self.TTRef.CLE, self.TTRef.LPA),
            "EOG": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
        }
    
    def list_records(self, basepath):
        paths_dict = {}
        
        subject_paths = [x for x in os.listdir(basepath) if x.startswith("sub")]

        for s_path in subject_paths:
            subject_id = s_path
            record_paths = [x for x in os.listdir(basepath+s_path) if x.startswith("ses")]
            
            records = []
            
            for r_path in record_paths:
                base_data_path = f"{basepath}{s_path}/{r_path}/eeg"
                
                data_path = f"{base_data_path}/{s_path}_{r_path}_task-sleep_eeg.set"
                label_path = f"{base_data_path}/{s_path}_{r_path}_task-sleep_acq-scoring_events.tsv"
                
                if os.path.exists(data_path) and os.path.exists(label_path):
                    records.append((data_path, label_path))
                
            paths_dict[subject_id] = records
            
        return paths_dict
    
    
    def read_psg(self, record):
        psg_path, hyp_path = record
        
        try:
            label_pd = pd.read_csv(hyp_path, sep = '\t')
        except:
            self.log_warning("Could not read CSV file", subject="", record=psg_path)
            return None
                
        y = label_pd["Staging1"].values.tolist()
        x = dict()
        
        data = mne.io.read_raw_eeglab(psg_path, verbose=False)
        sample_rate = int(data.info['sfreq'])
        
        df = data.to_data_frame()
        
        # Get first 6 and next 6 channels
        l = df[self.left()]
        r = df[self.right()]
        
        # Mean the values for each side
        l = l.mean(axis=1)
        r = r.mean(axis=1)
             
        # Remove NANS
        l = l.fillna(0, inplace=False)
        r = r.fillna(0, inplace=False)
        
        # Subtract and create a "fake" eog channel
        eeg = l-r
        eog = eeg.copy(deep=True)
        
        eeg = eeg.values.tolist()
        eog = eog.values.tolist()
        
        x["EEG"] = (eeg, sample_rate)
        x["EOG"] = (eog, sample_rate)
        
        return x,y