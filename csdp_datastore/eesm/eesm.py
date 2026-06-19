import os

import mne
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from csdp_datastore.base import BaseDataset

from ..models import EarEEGRef, Labels, Mapping


class EESM_Cleaned(BaseDataset):
    """
    ABOUT THIS DATASET
    """

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        do_nan_interpolation: bool = True,
    ):
        self.do_nan_interpolation = do_nan_interpolation
        super().__init__(dataset_path=dataset_path, output_path=output_path)

    def label_mapping(self):
        return {
            1: Labels.Wake,
            2: Labels.REM,
            3: Labels.N1,
            4: Labels.N2,
            5: Labels.N3,
            6: Labels.UNKNOWN,
            7: Labels.UNKNOWN,
            8: Labels.UNKNOWN,
        }

    def dataset_name(self):
        return "eesm"

    def channel_mapping(self):
        return {
            "Left-Right": Mapping(EarEEGRef.EL_AVG, EarEEGRef.ER_AVG),
        }

    def list_records(self, basepath):
        paths_dict = {}

        subject_paths = [x for x in os.listdir(basepath) if x.startswith("sub")]

        for s_path in subject_paths:
            subject_id = s_path
            record_paths = [x for x in os.listdir(f"{basepath}/{s_path}") if x.startswith("ses")]

            records = []

            for r_path in record_paths:
                base_label_path = f"{basepath}/{s_path}/{r_path}/eeg"
                base_data_path = f"{basepath}/derivatives/cleaned_1/{s_path}/{r_path}/eeg"

                data_path = f"{base_data_path}/{s_path}_{r_path}_task-sleep_acq-PSG_desc-cleaned1_eeg.set"
                label_path = f"{base_label_path}/{s_path}_{r_path}_task-sleep_acq-scoring1_events.tsv"

                if os.path.exists(data_path) and os.path.exists(label_path):
                    records.append((r_path, data_path, label_path))

            paths_dict[subject_id] = records

        return paths_dict

    def read_psg(self, record):
        psg_path, hyp_path = record

        x = dict()

        try:
            label_pd = pd.read_csv(hyp_path, sep="\t")
        except Exception:
            self.log_warning("Could not read CSV file")
            return None

        y = label_pd["Scoring1"].values.tolist()

        raw_data: mne.io.Raw = mne.io.read_raw_eeglab(psg_path, verbose=False)
        sample_rate = int(raw_data.info["sfreq"])

        y = np.array(y)
        left_keys = ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"]
        right_keys = ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]

        left_data: np.ndarray = raw_data.get_data(picks=left_keys)
        right_data: np.ndarray = raw_data.get_data(picks=right_keys)

        left_avg = np.nanmean(left_data, axis=0)
        right_avg = np.nanmean(right_data, axis=0)

        deriv = left_avg - right_avg

        data, nEpochs_min = self.slice_and_interpolate_channel(deriv, sample_rate, len(y))

        x["Left-Right"] = (data, sample_rate)

        y = y[0:nEpochs_min]

        return x, y

    def slice_and_interpolate_channel(self, data, sample_rate, y_len):
        epochLength_old = int(sample_rate * 30)
        nEpochs = int(np.floor(len(data) / epochLength_old))
        data = data[0 : nEpochs * epochLength_old]

        data = data.reshape(1, -1)

        inputNans = np.isnan(data)

        data[inputNans] = 0

        if self.do_nan_interpolation:
            data = self.interpolateOverNans(data, sample_rate)

        nEpochs_min = min(nEpochs, y_len)

        data = data.flatten()

        data = data[0 : nEpochs_min * 30 * sample_rate]

        return data, nEpochs_min

    # From Kaares repository
    def findRuns(self, input):

        runStarts = []
        runLengths = []

        sequence = np.asarray(input).reshape(-1)
        if ~(sequence.all() | ((1 - sequence).all())):
            sequence = sequence.astype(int)  # diff complains if it's boolean
            changes = np.diff([0, *sequence, 0])
            runStarts = (changes > 0).nonzero()[0]
            runEnds = (changes < 0).nonzero()[0]
            runLengths = runEnds - runStarts
            assert all(runLengths > 0)

        return runStarts, runLengths

    # From Kaares repository
    def interpolateOverNans(self, allDeriv, fs):
        allDeriv[np.isnan(allDeriv[:, 0]), 0] = 0
        allDeriv[np.isnan(allDeriv[:, -1]), -1] = 0

        for iDeriv in range(allDeriv.shape[0]):
            nanSamples = np.isnan(allDeriv[iDeriv, :]).nonzero()[0]

            if nanSamples.size > 0:
                [nanStart, nanDur] = self.findRuns(np.isnan(allDeriv[iDeriv, :]))
                nanDur = nanDur - 1
                realSamples = np.unique([nanStart - 1, (nanStart + nanDur) + 1])

                distanceToReal = nanSamples * 0
                counter = 0
                for iRun in range(len(nanDur)):
                    distanceToReal[range(counter, counter + nanDur[iRun])] = [
                        *range(int(np.floor(nanDur[iRun] / 2))),
                        *range(int(np.ceil(nanDur[iRun] / 2)), 0, -1),
                    ]
                    counter = counter + nanDur[iRun]

                interpValues = interp1d(realSamples, allDeriv[iDeriv, realSamples])(nanSamples)
                interpValues = interpValues * np.exp(-distanceToReal / (fs * 1))

                allDeriv[iDeriv, nanSamples] = interpValues

        return allDeriv


class EESM_Uncleaned(EESM_Cleaned):
    def dataset_name(self):
        return "eesm_uncleaned"

    def list_records(self, basepath):
        paths_dict = {}

        subject_paths = [x for x in os.listdir(basepath) if x.startswith("sub")]

        for s_path in subject_paths:
            subject_id = s_path
            record_paths = [x for x in os.listdir(f"{basepath}/{s_path}") if x.startswith("ses")]

            records = []

            for r_path in record_paths:
                base_label_path = f"{basepath}/{s_path}/{r_path}/eeg"

                data_path = f"{base_label_path}/{s_path}_{r_path}_task-sleep_acq-PSG_eeg.set"
                label_path = f"{base_label_path}/{s_path}_{r_path}_task-sleep_acq-scoring1_events.tsv"

                if os.path.exists(data_path) and os.path.exists(label_path):
                    records.append((data_path, label_path))

            paths_dict[subject_id] = records

        return paths_dict
