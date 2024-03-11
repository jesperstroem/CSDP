import os
from abc import ABC, abstractmethod
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from h5py import File
from enum import Enum, auto, IntEnum
from .logger import LoggingModule, EventSeverity
from scipy import signal

class FilterSettings():
    def __init__(self,
                 order = 5,
                 cutoffs: list[float] = [0.1, 40]):
        assert len(cutoffs) == 2

        self.cutoffs = cutoffs
        self.order = order

    cutoffs: list[float]
    order: int

class BaseDataset(ABC):
    def __init__(
        self, 
        dataset_path: str, 
        output_path: str,
        max_num_subjects: int = None, 
        filter: bool = True,
        filtersettings: FilterSettings = FilterSettings(),
        scale_and_clip: bool = True,
        output_sample_rate: int = 128,
        data_format: str ="hdf5",
        logging_path: str = "./SleepDataPipeline/logs"
    ):
        """_summary_

        Args:
            dataset_path (str): Absolute path to the root path of the dataset to be transformed
            output_path (str): Path where the preprocessed file will be saved
            max_num_subjects (int, optional): How many subjects to transform. Mostly used for testing new implementations. Defaults to None, which means all subjects will be transformed.
            scale_and_clip (bool, optional): If the data should be scaled and outlier clipped. Defaults to True.
            output_sample_rate (int, optional): The sample rate for the output data. Defaults to 128.
            data_format (str, optional): The dataformat of the preprocessed data. Defaults to "hdf5".
            logging_path (str, optional): Where to save the logs. Defaults to "./SleepDataPipeline/logs".
            port_on_init (bool, optional): If the data should be transformed as soon as you initialize the class. Defaults to True.
        """
        self.max_num_subjects = max_num_subjects
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.logger = LoggingModule(logging_path)
        self.scale_and_clip = scale_and_clip
        self.output_sample_rate = output_sample_rate
        self.filter = filter

        self.filtersettings = filtersettings
        
        if data_format == "hdf5":
            self.write_function = self.write_record_to_database_hdf5
        elif data_format == "parquet":
            self.write_function = self.write_record_to_database_parquet
        else:
            self.log_error("Invalid data format. Must be one of [hdf5, parquet].")
            exit(1)
        
        assert os.path.exists(self.dataset_path), f"Path {self.dataset_path} does not exist"
    
    class Mapping:
        def __init__(self, ref1, ref2):
            self.ref1 = ref1
            self.ref2 = ref2
        
        def __eq__(self, other):
            return (self.ref1, self.ref2) == (other.ref1, other.ref2)
        
        def get_mapping(self):
            ctype = 'EOG' if self.ref1 in [BaseDataset.TTRef.EL,
                                           BaseDataset.TTRef.ER] else 'EEG'
            return '{t}_{r1}-{r2}'.format(t=ctype,
                                          r1=self.ref1,
                                          r2=self.ref2)
    
    class Labels(IntEnum):
        Wake = 0
        N1 = 1
        N2 = 2
        N3 = 3
        REM = 4
        UNKNOWN = 5

    class EarEEGRef(Enum):
        # Ear-EEG ONLY

        ELA = auto()
        ELB = auto()
        ELC = auto()
        ELT = auto()
        ELE = auto()
        ELI = auto()
        ERA = auto()
        ERB = auto()
        ERC = auto()
        ERT = auto()
        ERE = auto()
        ERI = auto()
        
        #Common ref
        REF = auto()

    class TTRef(Enum):        
        # 10-10 EEG system for scalp PSG

        """
        "MCN system renames four electrodes of the 10–20 system:
        T3 is now T7
        T4 is now T8
        T5 is now P7
        T6 is now P8"
        
        Source: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
        """
        
        Nz = auto()
        Fpz = auto()
        Fp1 = auto()
        Fp2 = auto()
        AF7 = auto()
        AF3 = auto()
        AFz = auto()
        AF4 = auto()
        AF8 = auto()
        F9 = auto()
        F7 = auto()
        F5 = auto()
        F3 = auto()
        F1 = auto()
        Fz = auto()
        F2 = auto()
        F4 = auto()
        F6 = auto()
        F8 = auto()
        F10 = auto()
        FT9 = auto()
        FT7 = auto()
        FC5 = auto()
        FC3 = auto()
        FC1 = auto()
        FCz = auto()
        FC2 = auto()
        FC4 = auto()
        FC6 = auto()
        FT8 = auto()
        FT10 = auto()
        T7 = auto() # Same as T3 in 10-20 system
        C5 = auto()
        C3 = auto()
        C1 = auto()
        Cz = auto()
        C2 = auto()
        C4 = auto()
        C6 = auto()
        T8 = auto() # Same as T4 in 10-20 system
        TP9 = auto()
        TP7 = auto()
        CP5 = auto()
        CP3 = auto()
        CP1 = auto()
        CPz = auto()
        CP2 = auto()
        CP4 = auto()
        CP6 = auto()
        TP8 = auto()
        TP10 = auto()
        P9 = auto()
        P7 = auto() # Same as T5 in 10-20 system
        P5 = auto()
        P3 = auto()
        P1 = auto()
        Pz = auto()
        P2 = auto()
        P4 = auto()
        P6 = auto()
        P8 = auto() # Same as T6 in 10-20 system
        P10 = auto()
        PO7 = auto()
        PO3 = auto()
        POz = auto()
        PO4 = auto()
        PO8 = auto()
        O1 = auto()
        Oz = auto()
        O2 = auto()
        Iz = auto()
        LPA = auto() # Same as A1 in 10-20 system
        RPA = auto() # Same as A2 in 10-20 system
        
        EL = auto()
        ER = auto()
        
        # Computed linked Ear and Linked Ear Reference. May be rare, and so far is only in MASS. Can only find this article describing it: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5479869/
        CLE = auto()
        LER = auto()
        
        def __str__(self):
            return self.name
      
    @property
    @abstractmethod
    def label_mapping(self) -> dict[str, Labels]:
        """_summary_

        Returns:
            dict[str, Labels]: A dictionary where keys are the possible label values from the original data, and values are the corresponding AASM label.
        """
        pass
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """_summary_

        Returns:
            str: The name of the dataset. The preprocessed data will be saved as "<dataset_name>.<file_format>"
        """
        pass
    
    
    @abstractmethod
    def list_records(self) -> dict[str, list[tuple]]:
        """_summary_

        Function to list needed information about original dataset structure, in order for read_psg to have needed information about where to find PSGs and hypnograms.

        Returns:
            dict[str, list[tuple]]: A dictionary containing a key for each subject in the dataset. Each value should be a list of record paths in the form of a tuple: i.e a path for the PSG datafile and a path for the hypnogram file. If data and labels is saved in the same file, simply specify the same filepath for both.
        """
        pass
    
    
    @abstractmethod
    def read_psg(self, record: tuple[str, str]) -> tuple[dict, list]:
        """_summary_
        Function to read PSG data along with labels from a single record.
        
        Args:
            record (tuple[str, str]): A tuple x,y containing absolute file paths to the data file and label file for the given record. This is provided by the baseclass after defining the "list_records" function.

        Returns:
            tuple[dict, list]: Returns a tuple (x,y). 
            x: A dictionary of data from available PSG channels for a record in the dataset. The keys should be the original channel names. The value should be a tuple (data, sample_rate). 
            y: A list of labels for 30 second data chunks for all records in dataset.
        """
        pass
    
    @abstractmethod
    def channel_mapping(self) -> dict[str, Mapping]:
        """_summary_
        Function for mapping to new channel name in following format: 
        {channel type}_{electrode 1}-{electrode 2}
        Example: EEG_C3-M2
        
        The EEG placements follows the 10-20/10-10 EEG naming convention.
        https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
        
        Returns:
            dict[str, Mapping]: A dictionary where the keys are the channel names from the original data, and the values are the corresponding 10-20 electrode names defined from the "Mapping" class.
        """
        pass
    
    
    def log_info(self, msg, subject = None, record = None):
        self.logger.log(msg, self.dataset_name(), subject, record, EventSeverity.Info)
    
    
    def log_warning(self, msg, subject = None, record = None):
        self.logger.log(msg, self.dataset_name(), subject, record, EventSeverity.Warning)
        
        
    def log_error(self, msg, subject = None, record = None):
        self.logger.log(msg, self.dataset_name(), subject, record, EventSeverity.Error)
    
    def __check_paths(self, paths_dict):
        for k in paths_dict.keys():
            record_list = paths_dict[k]
            
            for r in record_list:
                for file_path in r:
                    assert os.path.exists(file_path), f"Datapath: {file_path}"
        
    def filter_channel(self, channel, fs):
        order = self.filtersettings.order
        l_cut = self.filtersettings.cutoffs[0]
        h_cut = self.filtersettings.cutoffs[1]

        sos = signal.butter(order, [l_cut, h_cut], btype='bandpass', fs=fs, output="sos")
        channel = signal.sosfiltfilt(sos, channel)
        return channel


    def __map_channels(self, dic, y_len):
        new_dict = dict()

        for key in dic.keys():
            mapping = self.channel_mapping()
            
            try:
                chnl = mapping[key]
            except KeyError:
                continue
            
            new_key = chnl.get_mapping()

            data, sample_rate = dic[key]
            
            assert len(data) == y_len*sample_rate*30, "Length of data does not match the length of labels"
            
            if self.filter:
                data = self.filter_channel(data, sample_rate)

            if self.scale_and_clip:
                data = self.scale_channel(data)
                data = self.clip_channel(data)

            new_dict[new_key] = self.resample_channel(data,
                                                      output_rate=self.output_sample_rate,
                                                      source_sample_rate=sample_rate) # TODO: Test that resampling works
            
        return new_dict
    
    
    def __map_labels(self, labels):
        return list(map(lambda x: self.label_mapping()[x], labels))
    
    def clip_channel(self, chnl, min_max_times_global_iqr = 20):
        #https://github.com/perslev/psg-utils/blob/main/psg_utils/preprocessing/quality_control_funcs.py
        iqr = np.subtract(*np.percentile(chnl, [75, 25]))
        
        threshold = iqr * min_max_times_global_iqr

        clipped = np.clip(chnl, -threshold, threshold)
        
        return clipped
    
    def scale_channel(self, chnl):
        #https://github.com/perslev/psg-utils/blob/main/psg_utils/preprocessing/scaling.py
        chnl = np.reshape(chnl, (-1,1))

        assert len(chnl.shape) == 2 and chnl.shape[1] == 1

        transformer = RobustScaler().fit(chnl)
        
        scaled = transformer.transform(chnl).flatten()
        
        assert len(scaled.shape) == 1
        
        return scaled
    
    def resample_channel(self, channel, output_rate, source_sample_rate):
        """
        Function to resample a single data channel to the desired sample rate.
        """

        channel_resampled = resample_poly(
            channel,
            output_rate,
            source_sample_rate,
            axis=0
        )

        return channel_resampled
    
    def save_dataset_metadata(self):
        filtering_used = self.filter
        filtersettings = self.filtersettings
        output_samplerate = self.output_sample_rate
        
        file_path = f"{self.output_path}{self.dataset_name()}.hdf5"

        try:
            with File(file_path, "a") as f:
                meta_grp = f.create_group("meta")

                filter_grp = meta_grp.create_group("filtersettings")
                filter_grp.create_dataset("filter_applied", data=filtering_used)
                filter_grp.create_dataset("win_len", data=filtersettings.win_len)
                filter_grp.create_dataset("cutoffs", data=filtersettings.cutoffs)

                meta_grp.create_dataset("output_samplerate", data=output_samplerate)

                self.log_info('Successfully saved metadata')
        except Exception as error:
            self.log_error(f"Could not save metadata due to error: {error}")

    def write_record_to_database_parquet(self, output_basepath, subject_number, record_number, x, y):
        """
        Function to write PSG data along with labels to the shared database containing all datasets in Parquet format.
        """
        
        psg_table = pa.table(x)
        hyp_table = pa.table({"labels": y})
        
        output_path = output_basepath + f"s_{subject_number}/r_{record_number}/"
        
        Path(output_path).mkdir(parents=True, exist_ok=True) # Because Parquet does not create directory
        pq.write_table(psg_table, output_path + "psg.parquet")
        pq.write_table(hyp_table, output_path + "hypnogram.parquet")
        
        
    def write_record_to_database_hdf5(self, output_basepath, subject_number, record_number, x, y): 
        """
        Function to write PSG data along with labels to the shared database containing all datasets in HDF5 format.
        """
        Path(output_basepath).mkdir(parents=True, exist_ok=True)
        
        file_path = f"{output_basepath}{self.dataset_name()}.hdf5"
        
        with File(file_path, "a") as f:
            #data_group = f.require_group("data")

            # Require subject group, since we want to use the existing one, if subject has more records
            grp_subject = f.require_group(f"{subject_number}")
            subgrp_record = grp_subject.create_group(f"{record_number}")
            
            subsubgrp_psg = subgrp_record.create_group("psg")
            
            for channel_name, channel_data in x.items():
                subsubgrp_psg.create_dataset(channel_name, data=channel_data)
            
            subgrp_record.create_dataset("hypnogram", data=y)
            self.log_info('Successfully wrote record to hdf5 file', subject_number, record_number)
    
    def download(self):
        self.log_warning('Download function was called, but no download functionality has been implemented')
        pass

    def port_data(self):
        paths_dict = self.list_records(basepath=self.dataset_path)

        self.__check_paths(paths_dict)

        subject_list = list(paths_dict.keys())[:self.max_num_subjects]

        if len(subject_list) == 0:
            self.log_error("No data found, could not port dataset")
            return

        file_path = f"{self.output_path}/{self.dataset_name()}.hdf5"
        exists = os.path.exists(file_path)

        if exists:
            self.log_warning("HDF5 file already exists. Removing it")
            os.remove(file_path)
        
        for subject_number in subject_list:
            record_number = 0
            
            for record in paths_dict[subject_number]:
                psg = self.read_psg(record)
                
                if psg == None:
                    self.log_error("PSG could not be read, skipping it", subject_number, record)
                    continue
                
                x, y = psg
                
                x = self.__map_channels(x, len(y))
                y = self.__map_labels(y)
                
                self.write_function(
                    f"{self.output_path}/",
                    subject_number,
                    record_number,
                    x, 
                    y
                )
                
                record_number = record_number + 1
        
        #self.save_dataset_metadata()
        self.log_info('Successfully ported dataset')
  
