import os
from abc import abstractmethod

from h5py import File

from csdp_datastore.base import BaseDataset

from ..models import Labels, Mapping, TTRef


class Base_DOD(BaseDataset):
    """
    ABOUT THIS DATASET

    """

    def label_mapping(self):
        return {-1: Labels.UNKNOWN, 0: Labels.Wake, 1: Labels.N1, 2: Labels.N2, 3: Labels.N3, 4: Labels.REM}

    def sample_rate(self):
        return 250  # We assume this due to what can be seen in the DOD code. USleep says 256 though..

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    def channel_mapping(self):
        return {
            "C3_M2": Mapping(TTRef.C3, TTRef.RPA),
            "C4_M1": Mapping(TTRef.C4, TTRef.LPA),
            "F4_F4": Mapping(TTRef.F3, TTRef.F4),
            "F3_M2": Mapping(TTRef.F3, TTRef.RPA),
            "F3_O1": Mapping(TTRef.F3, TTRef.O1),
            "F4_O2": Mapping(TTRef.F4, TTRef.O2),
            "O1_M2": Mapping(TTRef.O1, TTRef.RPA),
            "O2_M1": Mapping(TTRef.O2, TTRef.LPA),
            "EOG1": Mapping(TTRef.EL, TTRef.RPA),  # TODO: Find out refs
            "EOG2": Mapping(TTRef.ER, TTRef.RPA),  # TODO: Find out refs
        }

    def list_records(self, basepath):
        paths_dict = dict()

        for dir, subdir, filenames in os.walk(basepath):
            for file in filenames:
                record_no = file.split(".")[0]
                record_path = f"{dir}/{file}"

                paths_dict[record_no] = [
                    (
                        "1",
                        record_path,
                    )
                ]

        return paths_dict

    def read_psg(self, record):
        x = dict()

        record = record[0]

        try:
            with File(record, "r") as h5:
                signals = h5.get("signals")
                eeg_channels = signals.get("eeg")
                eog_channels = signals.get("eog")

                channel_len = len(eeg_channels.get(list(eeg_channels.keys())[0]))
                x_num_epochs = int(channel_len / self.sample_rate() / 30)

                for channel in eeg_channels:
                    x[channel] = (eeg_channels.get(channel)[()], self.sample_rate())
                for channel in eog_channels:
                    x[channel] = (eog_channels.get(channel)[()], self.sample_rate())

                y = list(h5.get("hypnogram")[()])

                assert len(y) == x_num_epochs, "Length of signal does not match the number of labels."
        except Exception:
            self.log_info("Could not read record")
            return None

        return x, y
