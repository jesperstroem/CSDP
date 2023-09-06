from .sdo_base import SleepdataOrg

class MESA(SleepdataOrg):    

    def channel_mapping(self):
        return {
            "EOG-L": self.Mapping(self.TTRef.EL, self.TTRef.Fpz),
            "EOG-R": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "EEG1": self.Mapping(self.TTRef.Fz, self.TTRef.Cz),
            "EEG2": self.Mapping(self.TTRef.Cz, self.TTRef.Oz),
            "EEG3": self.Mapping(self.TTRef.C4, self.TTRef.LPA)
        }
        