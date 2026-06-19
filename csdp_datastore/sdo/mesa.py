from ..models import Mapping, TTRef
from .sdo_base import SleepdataOrg


class MESA(SleepdataOrg):
    def channel_mapping(self):
        return {
            "EOG-L": Mapping(TTRef.EL, TTRef.Fpz),
            "EOG-R": Mapping(TTRef.ER, TTRef.Fpz),
            "EEG1": Mapping(TTRef.Fz, TTRef.Cz),
            "EEG2": Mapping(TTRef.Cz, TTRef.Oz),
            "EEG3": Mapping(TTRef.C4, TTRef.LPA),
        }
