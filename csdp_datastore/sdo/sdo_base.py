import os
import mne
import xml.etree.ElementTree as ET
from abc import abstractmethod
import subprocess

from csdp_datastore.base import BaseDataset

class SleepdataOrg(BaseDataset):

    def __init__(
        self, 
        dataset_path: str, 
        output_path: str,
        download_token: str = None,
        max_num_subjects: int = None, 
        scale_and_clip: bool = True,
        output_sample_rate: int = 128,
        data_format: str ="hdf5",
        logging_path: str = "./SleepDataPipeline/logs"
    ):
        
        super().__init__(dataset_path,
                         output_path,
                         max_num_subjects,
                         scale_and_clip,
                         output_sample_rate,
                         data_format,
                         logging_path)
        
        self.download_token = download_token

    """
    ABOUT THIS DATASET
    
    Baseclass for a collection of datasets downloaded from https://sleepdata.org. When downloading, you need to have the NSRR gem installed, and you need a user account and token from their website.
    
    """

    def label_mapping(self): 
        return {
            '0': self.Labels.Wake,
            '1': self.Labels.N1,
            '2': self.Labels.N2,
            '3': self.Labels.N3,
            '4': self.Labels.N3,
            '5': self.Labels.REM,
            '6': self.Labels.UNKNOWN,
            '9': self.Labels.UNKNOWN
        }
    
    @property
    @abstractmethod
    def channel_mapping(self):
        pass

    def download(self):
        args = ["nsrr", "download", f"{self.download_name()}/polysomnography/annotations-events-profusion", f"--token={self.download_token}"]
        
        p1 = subprocess.Popen(args,
                           stdout=subprocess.PIPE,
                           cwd=self.dataset_path)
        
        args = ["nsrr", "download", f"{self.download_name()}/polysomnography/edfs", f"--token={self.download_token}"]
        
        p2 = subprocess.Popen(args,
                           stdout=subprocess.PIPE,
                           cwd=self.dataset_path)
        
        codes = [p.wait() for p in [p1, p2]]
        
        self.dataset_path = f"{self.dataset_path}/{self.download_name()}"
        
    @abstractmethod
    def download_name(self):
        pass

    def dataset_name(self):
        return self.__class__.__name__.lower()
    
    def list_records(self, basepath):
        print(basepath)
        assert os.path.exists(basepath), "Path does not exist"

        paths_dict = {}
        
        poly_path = f"{basepath}/polysomnography/"
        hyp = "annotations-events-profusion"
        psg = "edfs"
        
        psg_path = f"{poly_path}{psg}"
        
        psg_files = []
        
        for dir, _, filenames in os.walk(psg_path):
            # On windows os.walk gives backslashes, so we need to replace them
            dir = dir.replace('\\','/')

            for file in filenames:
                psg_files.append(dir + "/" + file)
                
        psg_files = sorted(psg_files)

        for idx in range(len(psg_files)):
            psg_file_path = psg_files[idx]

            hyp_file_path = psg_file_path.replace('/'+psg+'/', '/'+hyp+'/', 1).replace('.edf', '-profusion.xml', 1)
            splits = hyp_file_path.split("-")
            subject_number = splits[-2]
            
            assert os.path.exists(psg_file_path), f"File {psg_file_path} does not exist"
            
            labels_exist = os.path.exists(hyp_file_path)
            
            if not labels_exist:
                self.log_warning(f"File does not exist, skipping this record", subject=None, record=hyp_file_path)
                continue
            
            paths_dict.setdefault(subject_number, []).append((psg_file_path, hyp_file_path))
        
        assert len(paths_dict) > 0, "No filepaths detected"

        return paths_dict
    
    # Override this if needed
    def slice_channel(self, x, y_len, sample_rate):
        epoch_diff = (len(x)/sample_rate/30) - y_len
        
        if epoch_diff < 0:
            msg = "Epoch diff can't be negative. You have more labels than you have data"
            raise Exception(msg)
        elif epoch_diff > 1.0:
            msg = "Epoch diff can't be bigger than 1. Sample_rate might be off"
            raise Exception(msg)
        
        x_len = y_len*sample_rate*30
        
        return x[:x_len]
    
    
    def read_psg(self, record):
        path_to_psg, path_to_hyp = record

        y = []
        tree = ET.parse(path_to_hyp)
        
        root = tree.getroot()
            
        sleep_stages = root.find("SleepStages")

        for stage in sleep_stages:
            y.append(stage.text)
            
        x = dict()
        
        data = mne.io.read_raw_edf(path_to_psg, verbose=False)

        sample_rate = int(data.info['sfreq'])

        not_found_chnls = []
                
        for channel in self.channel_mapping().keys():
            try:
                channel_data = data.get_data(channel)
            except ValueError:
                not_found_chnls.append(channel)                                                                         
                continue
            
            relative_channel_data = channel_data[0]

            try:
                final_channel_data = self.slice_channel(relative_channel_data, len(y), sample_rate)
            except Exception as msg:
                self.log_error(msg, subject=None, record=path_to_psg)
                return None

            assert len(final_channel_data) == len(y)*sample_rate*30, f"Channel length was {len(final_channel_data)}, but according to the number of labels it should be {len(y)*sample_rate*30}. Check the sample rate or override slice_channels if needed."
            
            x[channel] = (final_channel_data, sample_rate)
        
        assert len(x) > 0, "No data detected"
        
        # Checking if the "not found channels" already has been mapped under another name
        curr_mappings = [self.channel_mapping()[item] for item in x]
        not_found_chnls = [item for item in not_found_chnls if self.channel_mapping()[item] not in curr_mappings]

        if len(not_found_chnls) > 0:
            self.log_warning('Did not find channels: {channels} was not found in the record. Possibilities are {present}'.format(channels=not_found_chnls,
                                                                                                                                 present=data.ch_names),
                             record=path_to_psg)
            
        return x, y
