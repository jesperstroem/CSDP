import pandas as pd
import mne
import numpy as np
from csdp_datastore import EESM_Cleaned
import os
import h5py

#""

class EESM2(EESM_Cleaned):
    def label_mapping(self):
        return {
            "Wake": self.Labels.Wake,
            "REM": self.Labels.REM,
            "N1": self.Labels.N1,
            "N2": self.Labels.N2,
            "N3": self.Labels.N3,
            "Artefact": self.Labels.UNKNOWN,
        }
        
    def dataset_name(self):
        return "eesm2"

    def channel_mapping(self):
        return {
            "EL0": self.Mapping(self.EarEEGRef.ELA, self.EarEEGRef.REF),
            "EL1": self.Mapping(self.EarEEGRef.ELB, self.EarEEGRef.REF),
            "ER0": self.Mapping(self.EarEEGRef.ERA, self.EarEEGRef.REF),
            "ER1": self.Mapping(self.EarEEGRef.ERB, self.EarEEGRef.REF),
        }    

    def list_records(self, basepath):
        paths_dict = {}
        
        subject_paths = [x for x in os.listdir(basepath) if x.startswith("sub")]

        for s_path in subject_paths:
            subject_id = s_path

            record_paths = [x for x in os.listdir(f"{basepath}/{s_path}") if x.startswith("ses")]
            record_paths = [x for x in record_paths if ("01" in x) or ("02" in x)]

            records = []

            for r_path in record_paths:
                base_path = f"{basepath}/{s_path}/{r_path}/eeg"

                data_path = f"{base_path}/{s_path}_{r_path}_task-sleep_acq-Ear_eeg.set"
                label_path = f"{base_path}/{s_path}_{r_path}_task-sleep_acq-PSG_events.tsv"
                
                if os.path.exists(data_path) and os.path.exists(label_path):
                    records.append((data_path, label_path))
                
            paths_dict[subject_id] = records

        return paths_dict

    def read_psg(self, record):
        psg_path, hyp_path = record

        x = dict()

        try:
            label_pd = pd.read_csv(hyp_path, sep = '\t')
        except:
            self.log_warning("Could not read CSV file", subject="", record=psg_path)
            return None

        y = label_pd["stages"].values.tolist()

        raw_data: mne.io.Raw = mne.io.read_raw_eeglab(psg_path, verbose=False)
        sample_rate = int(raw_data.info['sfreq'])

        y = np.array(y)

        for c in self.channel_mapping().keys():
            data: np.ndarray = raw_data.get_data(picks=c)

            data = data.flatten()

            data, nEpochs_min = self.slice_and_interpolate_channel(data, sample_rate, len(y))

            x[c] = (data, sample_rate)
        
        y=y[0:nEpochs_min]
        
        return x, y