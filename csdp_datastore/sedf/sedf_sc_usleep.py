import os

from .base_sedf import Base_Sedf


class Sedf_SC_USleep(Base_Sedf):
    """
    ABOUT THIS DATASET
    
    The naming convention of records is as follows: SC4ssNE0 where:
    SC: Sleep Cassette study.
    ss: Subject number (notice that most subjects has 2 records), e.g. 00.
    N: Night (1 or 2).
    
    Channels included in dataset: ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker'].

    
    EEG and EOG signals were each sampled at 100Hz.
    """ 
    def dataset_name(self):
        return "sedf_sc"
    
    
    # This is probably outdated, since its on record and not subject basis.
    def list_records(self, basepath):
        paths_dict = {}
        
        record_paths = os.listdir(basepath)
        
        for path in record_paths:
            record_path = f"{basepath}{path}"
            
            for file in os.listdir(record_path):
                if "Hypnogram" in file:
                    hyp_path = f"{record_path}/{file}"
                elif "PSG" in file:
                    psg_path = f"{record_path}/{file}"
                else:
                    print("PSG or hypnogram file not found. Exiting..")
                    exit()
                    
            paths_dict[path] = [(psg_path, hyp_path)]
            
        return paths_dict
