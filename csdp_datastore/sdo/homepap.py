from ..models import Mapping, TTRef
from .sdo_base import SleepdataOrg


class HOMEPAP(SleepdataOrg):
    def channel_mapping(self):
        return {
            "E1": Mapping(TTRef.EL, TTRef.Fpz),
            "E2": Mapping(TTRef.ER, TTRef.Fpz),
            "F3": Mapping(TTRef.F3, TTRef.Fpz),
            "F4": Mapping(TTRef.F4, TTRef.Fpz),
            "C3": Mapping(TTRef.C3, TTRef.Fpz),
            "C4": Mapping(TTRef.C4, TTRef.Fpz),
            "O1": Mapping(TTRef.O1, TTRef.Fpz),
            "O2": Mapping(TTRef.O2, TTRef.Fpz),
            "M1": Mapping(TTRef.LPA, TTRef.Fpz),
            "M2": Mapping(TTRef.RPA, TTRef.Fpz),
            "E1-M2": Mapping(TTRef.EL, TTRef.RPA),
            "E2-M1": Mapping(TTRef.ER, TTRef.LPA),
            "F3-M2": Mapping(TTRef.F3, TTRef.RPA),
            "F4-M1": Mapping(TTRef.F4, TTRef.LPA),
            "C3-M2": Mapping(TTRef.C3, TTRef.RPA),
            "C4-M1": Mapping(TTRef.C4, TTRef.LPA),
            "O1-M2": Mapping(TTRef.O1, TTRef.RPA),
            "O2-M1": Mapping(TTRef.O2, TTRef.LPA),
        }
