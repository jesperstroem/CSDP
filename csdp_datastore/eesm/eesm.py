import os

from common_sleep_data_store.datastore_classes.base import BaseDataset
import pandas as pd
import mne
from scipy.interpolate import interp1d
import numpy as np

class EESM_Cleaned(BaseDataset):
    """
    ABOUT THIS DATASET 
    """
    
    def label_mapping(self):
        return {
            1: self.Labels.Wake,
            2: self.Labels.REM,
            3: self.Labels.N1,
            4: self.Labels.N2,
            5: self.Labels.N3,
            6: self.Labels.UNKNOWN,
            7: self.Labels.UNKNOWN,
            8: self.Labels.UNKNOWN
        }
        
    def dataset_name(self):
        return "eesm"

    def channel_mapping(self):
        return {
            "ELA": self.Mapping(self.EarEEGRef.ELA, self.EarEEGRef.REF),
            "ELB": self.Mapping(self.EarEEGRef.ELB, self.EarEEGRef.REF),
            "ELC": self.Mapping(self.EarEEGRef.ELC, self.EarEEGRef.REF),
            "ELT": self.Mapping(self.EarEEGRef.ELT, self.EarEEGRef.REF),
            "ELE": self.Mapping(self.EarEEGRef.ELE, self.EarEEGRef.REF),
            "ELI": self.Mapping(self.EarEEGRef.ELI, self.EarEEGRef.REF),
            "ERA": self.Mapping(self.EarEEGRef.ERA, self.EarEEGRef.REF),
            "ERB": self.Mapping(self.EarEEGRef.ERB, self.EarEEGRef.REF),
            "ERC": self.Mapping(self.EarEEGRef.ERC, self.EarEEGRef.REF),
            "ERT": self.Mapping(self.EarEEGRef.ERT, self.EarEEGRef.REF),
            "ERE": self.Mapping(self.EarEEGRef.ERE, self.EarEEGRef.REF),
            "ERI": self.Mapping(self.EarEEGRef.ERI, self.EarEEGRef.REF),
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
                
        y = label_pd["Scoring1"].values.tolist()
        
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
    
    def slice_and_interpolate_channel(self, data, sample_rate, y_len):
        epochLength_old=int(sample_rate*30)
        nEpochs=int(np.floor(len(data)/epochLength_old))
        data=data[0:nEpochs*epochLength_old]

        data=data.reshape(1,-1)

        inputNans=np.isnan(data)

        data[inputNans]=0

        data=self.interpolateOverNans(data,sample_rate)

        nEpochs_min=min(nEpochs,y_len)

        data = data.flatten()
        
        data=data[0:nEpochs_min*30*sample_rate]

        return data, nEpochs_min
    
    # From Kaares repository
    def findRuns(self, input):

        runStarts=[]
        runLengths=[]

        sequence=np.asarray(input).reshape(-1)
        if ~(sequence.all() | ((1-sequence).all())):
            sequence=sequence.astype(int) #diff complains if it's boolean
            changes=np.diff([0, *sequence, 0])
            runStarts=(changes>0).nonzero()[0]
            runEnds=(changes<0).nonzero()[0]
            runLengths=runEnds-runStarts
            assert all(runLengths>0)

        return runStarts, runLengths
    
    # From Kaares repository
    def interpolateOverNans(self, allDeriv,fs):
        allDeriv[np.isnan(allDeriv[:,0]),0]=0
        allDeriv[np.isnan(allDeriv[:,-1]),-1]=0


        for iDeriv in range(allDeriv.shape[0]):
            
            nanSamples=np.isnan(allDeriv[iDeriv,:]).nonzero()[0]

            if nanSamples.size>0:
                [nanStart, nanDur]=self.findRuns(np.isnan(allDeriv[iDeriv,:]))
                nanDur=nanDur-1
                realSamples=np.unique([nanStart-1, (nanStart+nanDur)+1])
                
                distanceToReal=nanSamples*0
                counter=0
                for iRun in range(len(nanDur)):
                    distanceToReal[range(counter,counter+nanDur[iRun])]=[*range(int(np.floor(nanDur[iRun]/2))), *range(int(np.ceil(nanDur[iRun]/2)),0,-1) ]
                    counter=counter+nanDur[iRun]
            
                interpValues=interp1d(realSamples,allDeriv[iDeriv,realSamples])(nanSamples)
                interpValues=interpValues*np.exp(-distanceToReal/(fs*1))
                
                allDeriv[iDeriv,nanSamples]=interpValues

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