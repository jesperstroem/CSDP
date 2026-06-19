import os
import re
from abc import abstractmethod

import mne

from csdp_datastore.base import BaseDataset

from ..models import Labels, Mapping, TTRef


class Mass_base(BaseDataset):
    """
    ABOUT THIS DATASET

    """

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    def label_mapping(self):
        return {
            "Sleep stage ?": Labels.UNKNOWN,
            "Sleep stage 1": Labels.N1,
            "Sleep stage 2": Labels.N2,
            "Sleep stage 3": Labels.N3,
            "Sleep stage W": Labels.Wake,
            "Sleep stage R": Labels.REM,
        }

    def channel_mapping(self):
        ref = TTRef.CLE

        dic = {
            "EEG T4-CLE": Mapping(TTRef.T8, ref),
            "EEG P3-CLE": Mapping(TTRef.P3, ref),
            "EEG F4-CLE": Mapping(TTRef.F4, ref),
            "EEG T6-CLE": Mapping(TTRef.P8, ref),
            "EEG F8-CLE": Mapping(TTRef.F8, ref),
            "EEG Cz-CLE": Mapping(TTRef.Cz, ref),
            "EEG T5-CLE": Mapping(TTRef.P7, ref),
            "EEG F7-CLE": Mapping(TTRef.F7, ref),
            "EEG T3-CLE": Mapping(TTRef.T7, ref),
            "EEG Fz-CLE": Mapping(TTRef.Fz, ref),
            "EEG C4-CLE": Mapping(TTRef.C4, ref),
            "EEG O1-CLE": Mapping(TTRef.O1, ref),
            "EEG O2-CLE": Mapping(TTRef.O2, ref),
            "EEG C3-CLE": Mapping(TTRef.C3, ref),
            "EEG F3-CLE": Mapping(TTRef.F3, ref),
            "EEG P4-CLE": Mapping(TTRef.P4, ref),
            "EEG A2-CLE": Mapping(TTRef.RPA, ref),
            "EEG Pz-CLE": Mapping(TTRef.Pz, ref),
            # EOGS assumed to be cross ear as in other datasets
            "EOG Right Horiz": Mapping(TTRef.ER, TTRef.LPA),
            "EOG Left Horiz": Mapping(TTRef.EL, TTRef.RPA),
        }

        # For some reason, some of the records use another ref - LER
        ref = TTRef.LER

        appends = {
            "EEG T4-LER": Mapping(TTRef.T8, ref),
            "EEG P3-LER": Mapping(TTRef.P3, ref),
            "EEG F4-LER": Mapping(TTRef.F4, ref),
            "EEG T6-LER": Mapping(TTRef.P8, ref),
            "EEG F8-LER": Mapping(TTRef.F8, ref),
            "EEG Cz-LER": Mapping(TTRef.Cz, ref),
            "EEG T5-LER": Mapping(TTRef.P7, ref),
            "EEG F7-LER": Mapping(TTRef.F7, ref),
            "EEG T3-LER": Mapping(TTRef.T7, ref),
            "EEG Fz-LER": Mapping(TTRef.Fz, ref),
            "EEG C4-LER": Mapping(TTRef.C4, ref),
            "EEG O1-LER": Mapping(TTRef.O1, ref),
            "EEG O2-LER": Mapping(TTRef.O2, ref),
            "EEG C3-LER": Mapping(TTRef.C3, ref),
            "EEG F3-LER": Mapping(TTRef.F3, ref),
            "EEG P4-LER": Mapping(TTRef.P4, ref),
            "EEG Pz-LER": Mapping(TTRef.Pz, ref),
        }

        dic.update(appends)

        return dic

    def list_records(self, basepath):
        dic = dict()

        record_paths = os.listdir(basepath)

        for record_name in record_paths:
            if record_name.endswith("PSG.edf"):
                subject_id = record_name.rstrip(" PSG.edf")

                label_path = basepath + "annotations/" + subject_id + "_saf.txt"
                data_path = basepath + record_name

                dic[subject_id] = [("1", data_path, label_path)]

        return dic

    def find_intervals(self, lines):
        first_line = lines[0]
        last_line = lines[-1]
        first_line = re.split("\x14|\x15|\x00|\n", first_line)
        last_line = re.split("\x14|\x15|\x00|\n", last_line)

        start_time = float(first_line[0])
        end_time = float(last_line[0])

        return start_time, end_time

    def handle_line(self, line):
        line = re.split("\x14|\x15|\x00|\n", line)

        label = line[2]

        assert label in self.label_mapping()

        return label

    def handle_labels(self, labelpath):
        with open(labelpath, "r") as f:
            lines = f.readlines()
            y = list(map(lambda x: self.handle_line(x), lines))

        start_time, end_time = self.find_intervals(lines)

        return y, start_time, end_time

    def read_psg(self, record):
        datapath, labelpath = record

        data = mne.io.read_raw_edf(datapath, verbose=False)
        sample_rate = data.info["sfreq"]

        y, start_time, end_time = self.handle_labels(labelpath)

        start_time = start_time - data.first_time
        end_time = end_time - data.first_time + 30

        # Code from MNE to avoid near-zero errors
        # https://github.com/mne-tools/mne-python/blob/maint/1.3/mne/io/base.py#L1311-L1340
        if -sample_rate / 2 < start_time < 0:
            start_time = 0

        try:
            data.crop(start_time, end_time, True)
        except ValueError:
            self.log_error("Could not crop data")
            return None

        # In theory, the labels should match the data after cropping it, but it doesnt. Seems like the epoch length is not 30 seconds but 30.002 seconds or something..
        # Solution for now is to just crop from beginning, as i dont see how we fix this without creating a lot of trouble. It means the annotations are a little bit off, but hopefully it doesnt matter too much

        dic = dict()

        label_len = int(len(y) * 30 * sample_rate)

        diff = data.n_times - label_len

        # To make sure the difference is not too big.
        if diff > sample_rate * 3:
            self.log_error(f"Diff between label and data was {diff}, skipping")
            return None

        not_found_chnls = []

        for channel in self.channel_mapping().keys():
            try:
                channel_data = data.get_data(channel)
            except ValueError:
                not_found_chnls.append(channel)
                continue

            channel_data = channel_data[0][0:label_len]

            dic[channel] = (channel_data, sample_rate)

        if len(not_found_chnls) > 0:
            self.log_warning(
                "Did not find channels: {channels} was not found in the record. Possibilities are {present}".format(
                    channels=not_found_chnls, present=data.ch_names
                )
            )

        return dic, y
