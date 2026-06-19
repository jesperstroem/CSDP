import os

from h5py import File

from .base import BaseDataset
from .models import Labels, Mapping, TTRef


class DCSM(BaseDataset):
    """
    ABOUT THIS DATASET

    Channels included in dataset: ['ABDOMEN', 'C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'E2-M2', 'ECG-II', 'F3-M2', 'F4-M1', 'LAT', 'NASAL', 'O1-M2', 'O2-M1', 'RAT', 'SNORE', 'SPO2', 'THORAX'].

    EEG and EOG signals were each sampled at 256Hz.
    """

    def label_mapping(self):
        return {"W": Labels.Wake, "N1": Labels.N1, "N2": Labels.N2, "N3": Labels.N3, "REM": Labels.REM}

    def sample_rate(self):
        return 256

    def dataset_name(self):
        return "dcsm"

    def channel_mapping(self):
        return {
            "E1-M2": Mapping(TTRef.EL, TTRef.RPA),
            "E2-M2": Mapping(TTRef.ER, TTRef.RPA),
            "C3-M2": Mapping(TTRef.C3, TTRef.RPA),
            "C4-M1": Mapping(TTRef.C4, TTRef.LPA),
            "F3-M2": Mapping(TTRef.F3, TTRef.RPA),
            "F4-M1": Mapping(TTRef.F4, TTRef.LPA),
            "O1-M2": Mapping(TTRef.O1, TTRef.RPA),
            "O2-M1": Mapping(TTRef.O2, TTRef.LPA),
        }

    def list_records(self, basepath) -> dict[str, list[tuple]]:
        paths_dict = {}

        record_paths = os.listdir(basepath)

        for path in record_paths:
            record_path = f"{basepath}{path}"
            psg_path = f"{record_path}/psg.h5"
            hyp_path = f"{record_path}/hypnogram.ids"
            record_name = "1"  # Only one record per subject
            paths_dict[path] = [(record_name, psg_path, hyp_path)]

        return paths_dict

    def read_psg(self, record):
        psg_path, hyp_path = record

        x = dict()
        y = []

        try:
            with File(psg_path, "r") as h5:
                h5channels = h5.get("channels")

                for channel in self.channel_mapping().keys():
                    channel_data = h5channels[channel][:]

                    x[channel] = (
                        channel_data,
                        self.sample_rate(),
                    )  # We are assuming sample rate is same across channels
        except Exception as msg:
            self.log_error(msg)
            return None

        with open(hyp_path) as f:
            hypnogram = f.readlines()

            for element in hypnogram:
                prev_stages_time, stage_time, label = element.rstrip().split(",")
                stage_time = int(stage_time)

                n_epochs_in_stage = int(stage_time / 30)

                for label_entry in range(n_epochs_in_stage):
                    stg = label
                    assert stg is not None

                    y.append(stg)

        return x, y
