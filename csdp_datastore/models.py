from enum import Enum, IntEnum, auto


class EarEEGRef(Enum):
    # Ear-EEG ONLY

    ELA = auto()
    ELB = auto()
    ELC = auto()
    ELT = auto()
    ELE = auto()
    ELI = auto()
    ERA = auto()
    ERB = auto()
    ERC = auto()
    ERT = auto()
    ERE = auto()
    ERI = auto()

    EL_AVG = auto()
    ER_AVG = auto()

    # Common ref
    REF = auto()


class TTRef(Enum):
    # 10-10 EEG system for scalp PSG

    """
    "MCN system renames four electrodes of the 10–20 system:
    T3 is now T7
    T4 is now T8
    T5 is now P7
    T6 is now P8"

    Source: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
    """

    Nz = auto()
    Fpz = auto()
    Fp1 = auto()
    Fp2 = auto()
    AF7 = auto()
    AF3 = auto()
    AFz = auto()
    AF4 = auto()
    AF8 = auto()
    F9 = auto()
    F7 = auto()
    F5 = auto()
    F3 = auto()
    F1 = auto()
    Fz = auto()
    F2 = auto()
    F4 = auto()
    F6 = auto()
    F8 = auto()
    F10 = auto()
    FT9 = auto()
    FT7 = auto()
    FC5 = auto()
    FC3 = auto()
    FC1 = auto()
    FCz = auto()
    FC2 = auto()
    FC4 = auto()
    FC6 = auto()
    FT8 = auto()
    FT10 = auto()
    T7 = auto()  # Same as T3 in 10-20 system
    C5 = auto()
    C3 = auto()
    C1 = auto()
    Cz = auto()
    C2 = auto()
    C4 = auto()
    C6 = auto()
    T8 = auto()  # Same as T4 in 10-20 system
    TP9 = auto()
    TP7 = auto()
    CP5 = auto()
    CP3 = auto()
    CP1 = auto()
    CPz = auto()
    CP2 = auto()
    CP4 = auto()
    CP6 = auto()
    TP8 = auto()
    TP10 = auto()
    P9 = auto()
    P7 = auto()  # Same as T5 in 10-20 system
    P5 = auto()
    P3 = auto()
    P1 = auto()
    Pz = auto()
    P2 = auto()
    P4 = auto()
    P6 = auto()
    P8 = auto()  # Same as T6 in 10-20 system
    P10 = auto()
    PO7 = auto()
    PO3 = auto()
    POz = auto()
    PO4 = auto()
    PO8 = auto()
    O1 = auto()
    Oz = auto()
    O2 = auto()
    Iz = auto()
    LPA = auto()  # Same as A1 in 10-20 system
    RPA = auto()  # Same as A2 in 10-20 system

    EL = auto()
    ER = auto()

    # Computed linked Ear and Linked Ear Reference. May be rare, and so far is only in MASS. Can only find this article describing it: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5479869/
    CLE = auto()
    LER = auto()

    def __str__(self):
        return self.name


class Mapping:
    def __init__(self, ref1, ref2):
        self.ref1: Enum = ref1
        self.ref2: Enum = ref2

    def __eq__(self, other):
        return (self.ref1, self.ref2) == (other.ref1, other.ref2)

    def get_mapping(self):
        ctype = "EOG" if self.ref1 in [TTRef.EL, TTRef.ER] else "EEG"
        return f"{ctype}_{self.ref1.name}-{self.ref2.name}"


class Labels(IntEnum):
    Wake = 0
    N1 = 1
    N2 = 2
    N3 = 3
    REM = 4
    UNKNOWN = 5


class Rereference:
    first: TTRef
    second: TTRef
    result: TTRef

    def __init__(self, first: Mapping, second: Mapping, result: Mapping):
        self.first = first
        self.second = second
        self.result = result


class ChannelCalculations:
    drop_existing: bool
    rereferences: list[Rereference]

    def __init__(self, rereferences: list[Rereference], drop_existing=False):
        self.rereferences = rereferences
        self.drop_existing = drop_existing


class FilterSettings:
    def __init__(self, lcut=0.1, hcut=None, order=2):
        if lcut is not None and hcut is None:
            type = "highpass"
            self.cutoffs = lcut
        elif hcut is not None and lcut is None:
            type = "lowpass"
            self.cutoffs = hcut
        else:
            type = "bandpass"
            self.cutoffs = [lcut, hcut]

        self.order = order
        self.type = type

    cutoffs: list[float]
    order: int
    type: str
