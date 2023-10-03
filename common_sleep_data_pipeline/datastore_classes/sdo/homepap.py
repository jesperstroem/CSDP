from .sdo_base import SleepdataOrg

class HOMEPAP(SleepdataOrg):

    def channel_mapping(self):
        return {
            "E1": self.Mapping(self.TTRef.EL, self.TTRef.Fpz),
            "E2": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "F3": self.Mapping(self.TTRef.F3, self.TTRef.Fpz),
            "F4": self.Mapping(self.TTRef.F4, self.TTRef.Fpz),
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "O1": self.Mapping(self.TTRef.O1, self.TTRef.Fpz),
            "O2": self.Mapping(self.TTRef.O2, self.TTRef.Fpz),
            "M1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "M2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "E1-M2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "E2-M1": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "F3-M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "C3-M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "O1-M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA)
        }
        