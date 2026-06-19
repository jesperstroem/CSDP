from ..models import Mapping, TTRef
from .sdo_base import SleepdataOrg


class ABC(SleepdataOrg):
    """
    ABOUT THIS DATASET

    Channels included in dataset: ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'M1', 'M2', 'E1', 'E2', 'ECG1', 'ECG2', 'LLeg1', 'LLeg2', 'RLeg1', 'RLeg2', 'Chin1', 'Chin2', 'Chin3', 'Airflow', 'Abdo', 'Thor', 'Snore', 'Sum', 'PosSensor', 'Ox Status', 'Pulse', 'SpO2', 'Nasal Pressure', 'CPAP Flow', 'CPAP Press', 'Pleth', 'Derived HR', 'Light', 'Manual Pos'].

    All channels are measured against Fpz according to https://sleepdata.org/datasets/abc/pages/montage-and-sampling-rate-information.md
    {EDF label}-Fpz (e.g. F3-Fpz)

    EEG and EOG signals were each sampled at 256Hz.
    """

    def channel_mapping(self):
        r2 = TTRef.Fpz

        return {
            "F3": Mapping(TTRef.F3, r2),
            "F4": Mapping(TTRef.F4, r2),
            "C3": Mapping(TTRef.C3, r2),
            "C4": Mapping(TTRef.C4, r2),
            "O1": Mapping(TTRef.O1, r2),
            "O2": Mapping(TTRef.O2, r2),
            "M1": Mapping(TTRef.LPA, r2),
            "M2": Mapping(TTRef.RPA, r2),
            "E1": Mapping(TTRef.EL, r2),
            "E2": Mapping(TTRef.ER, r2),
        }
