import os

import mne

from .base import BaseDataset
from .models import Labels, Mapping, TTRef


class SVUH(BaseDataset):
    """
    ABOUT THIS DATASET

    """

    def sample_rate(self):
        return 128

    def dataset_name(self):
        return "svuh"

    def label_mapping(self):
        return {
            "0": Labels.Wake,
            "1": Labels.REM,
            "2": Labels.N1,
            "3": Labels.N2,
            "4": Labels.N3,
            "5": Labels.N3,  # Stage 4 in SVUH is same as N3
            "6": Labels.UNKNOWN,
            "7": Labels.UNKNOWN,
            "8": Labels.UNKNOWN,
        }

    def channel_mapping(self):
        return {
            "Lefteye": Mapping(TTRef.EL, TTRef.RPA),
            "RightEye": Mapping(TTRef.ER, TTRef.LPA),
            "C3A2": Mapping(TTRef.C3, TTRef.RPA),
            "C4A1": Mapping(TTRef.C4, TTRef.LPA),
        }

    def list_records(self, basepath):
        basepath = basepath + "files/"
        file_base = "ucddb"
        file_path = basepath + "/" + file_base
        subject_ids = [
            "002",
            "003",
            "005",
            "006",
            "007",
            "008",
            "009",
            "010",
            "011",
            "012",
            "013",
            "014",
            "015",
            "017",
            "018",
            "019",
            "020",
            "021",
            "022",
            "023",
            "024",
            "025",
            "026",
            "027",
            "028",
        ]

        dic = dict()

        for id in subject_ids:
            prepend = file_path + id

            if os.path.isfile(prepend + ".rec"):
                self.log_info("Renamed file {} to .edf".format(prepend + ".rec"))
                os.rename(prepend + ".rec", prepend + ".edf")

            dic[id] = [("1", prepend + ".edf", prepend + "_stage.txt")]

        return dic

    def read_psg(self, record):
        (datapath, labelpath) = record

        data = mne.io.read_raw_edf(datapath, verbose=False)

        dic = dict()

        with open(labelpath, "rb") as f:
            y = list(map(lambda x: chr(x[0]), f.readlines()))

        x_len = len(y) * self.sample_rate() * 30

        for channel in self.channel_mapping().keys():
            channel_data = data[channel]
            relative_channel_data = channel_data[0][0]
            dic[channel] = (relative_channel_data[:x_len], self.sample_rate())

        return dic, y
