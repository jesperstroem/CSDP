import pandas as pd
import mne
import numpy as np
from csdp_datastore import EESM_Cleaned

class EESM_Raw(EESM_Cleaned):
    def dataset_name(self):
        return "eesm_raw"

    def read_psg(self, record):
        psg_path, hyp_path = record

        x = dict()

        try:
            label_pd = pd.read_csv(hyp_path, sep = '\t')
        except:
            self.log_warning("Could not read CSV file", subject="", record=psg_path)
            return None
                
        y = label_pd["Scoring1"].values.tolist()
        
        raw_data: mne.io.Raw = mne.io.read_raw_eeglab(psg_path, verbose=False)
        sample_rate = int(raw_data.info['sfreq'])

        y = np.array(y)

        for c in self.channel_mapping().keys():
            data: np.ndarray = raw_data.get_data(picks=c)

            data = data.flatten()

            epochLength=int(sample_rate*30)
            nEpochs=int(np.floor(len(data)/epochLength))

            nEpochs_min=min(nEpochs, len(y))
            
            data=data[0:nEpochs_min*30*sample_rate]

            x[c] = (data, sample_rate)
        
        y=y[0:nEpochs_min]

        return x, y