from ..models import Mapping, TTRef
from .sdo_base import SleepdataOrg

# Does not work at the moment because of different samplerates across records.


class CFS(SleepdataOrg):
    def channel_mapping(self):
        return {
            "C3": Mapping(TTRef.C3, TTRef.Fpz),
            "C4": Mapping(TTRef.C4, TTRef.Fpz),
            "M1": Mapping(TTRef.LPA, TTRef.Fpz),
            "M2": Mapping(TTRef.RPA, TTRef.Fpz),
            "LOC": Mapping(TTRef.EL, TTRef.Fpz),
            "ROC": Mapping(TTRef.ER, TTRef.Fpz),
        }
