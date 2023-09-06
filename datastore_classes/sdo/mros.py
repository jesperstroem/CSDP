from .sdo_base import SleepdataOrg

class MROS(SleepdataOrg):    
    def channel_mapping(self):
        return {
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "A1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "A2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "ROC": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "LOC": self.Mapping(self.TTRef.EL, self.TTRef.Fpz)
        }
        