from ..models import Mapping, TTRef
from .sdo_base import SleepdataOrg


class SHHS(SleepdataOrg):
    def channel_mapping(self):
        return {
            "EEG(sec)": Mapping(TTRef.C3, TTRef.RPA),
            "EEG 2": Mapping(TTRef.C3, TTRef.RPA),
            "EEG sec": Mapping(TTRef.C3, TTRef.RPA),
            "EEG(SEC)": Mapping(TTRef.C3, TTRef.RPA),
            "EEG2": Mapping(TTRef.C3, TTRef.RPA),
            "EEG": Mapping(TTRef.C4, TTRef.LPA),
            "EOG(L)": Mapping(TTRef.EL, TTRef.Nz),
            "EOG(R)": Mapping(TTRef.ER, TTRef.Nz),
        }
