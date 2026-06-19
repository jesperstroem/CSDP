from ..models import Mapping, TTRef
from .sdo_base import SleepdataOrg


class CHAT(SleepdataOrg):
    """
    ABOUT THIS DATASET

    Channels included in dataset: ['Airflow', 'CannulaFlow', 'SUM', 'Chest', 'ABD', 'Snore', 'M1', 'M2', 'C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'T3', 'T4', 'E1', 'E2', 'ECG1', 'ECG2', 'Lchin', 'Rchin', 'Cchin', 'ECG3', 'Lleg1', 'Lleg2', 'Rleg1', 'Rleg2', 'SAO2Nellcor', 'PulseNellcor', 'PlethNellcor', 'EtCO2', 'Cap', 'RR', 'SaO2', 'Pulse', 'Position', 'DHR']


    EEG and EOG signals were each sampled at 200Hz.
    """

    def channel_mapping(self):
        r2 = TTRef.Fpz

        return {
            "M1": Mapping(TTRef.LPA, r2),
            "M2": Mapping(TTRef.RPA, r2),
            "C3": Mapping(TTRef.C3, r2),
            "C4": Mapping(TTRef.C4, r2),
            "O1": Mapping(TTRef.O1, r2),
            "O2": Mapping(TTRef.O2, r2),
            "F3": Mapping(TTRef.F3, r2),
            "F4": Mapping(TTRef.F4, r2),
            "T3": Mapping(TTRef.T7, r2),
            "T4": Mapping(TTRef.T8, r2),
            "E1": Mapping(TTRef.EL, r2),
            "E2": Mapping(TTRef.ER, r2),
        }
