import os
import mne
import sys

from .sdo_base import SleepdataOrg


class CCSHS(SleepdataOrg):
    """
    ABOUT THIS DATASET 
    
    Channels included in dataset: ['Airflow', 'CannulaFlow', 'SUM', 'Chest', 'ABD', 'Snore', 'M1', 'M2', 'C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'T3', 'T4', 'E1', 'E2', 'ECG1', 'ECG2', 'Lchin', 'Rchin', 'Cchin', 'ECG3', 'Lleg1', 'Lleg2', 'Rleg1', 'Rleg2', 'SAO2Nellcor', 'PulseNellcor', 'PlethNellcor', 'EtCO2', 'Cap', 'RR', 'SaO2', 'Pulse', 'Position', 'DHR']

    
    EEG and EOG signals were each sampled at 200Hz.
    """  
    
    def channel_mapping(self):
        r2 = self.TTRef.Fpz
        
        return {
            "C3": self.Mapping(self.TTRef.C3, r2),
            "C4": self.Mapping(self.TTRef.C4, r2),
            "A1": self.Mapping(self.TTRef.LPA, r2),
            "A2": self.Mapping(self.TTRef.RPA, r2),
            "LOC": self.Mapping(self.TTRef.EL, r2),
            "ROC": self.Mapping(self.TTRef.ER, r2)
        }
        