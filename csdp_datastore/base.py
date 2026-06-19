import os
from abc import ABC, abstractmethod
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from h5py import File

from csdp_pipeline.preprocessing.usleep_prep_steps import (
    clip_channel,
    filter_channel,
    remove_dc,
    resample_channel,
    scale_channel,
)

from .logger import EventSeverity, LoggingModule
from .models import ChannelCalculations, FilterSettings, Labels, Mapping


class BaseDataset(ABC):
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        overwrite_existing: bool = True,
        max_num_subjects: int = None,
        filter: bool = True,
        filtersettings: FilterSettings = FilterSettings(),
        scale_and_clip: bool = True,
        output_sample_rate: int = 128,
        data_format: str = "hdf5",
        logging_path: str = "./SleepDataPipeline/logs",
        calculated_channel_config: ChannelCalculations = None,
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
        self.subject_context = None
        self.record_context = None
        self.max_num_subjects = max_num_subjects
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.logger = LoggingModule()
        self.scale_and_clip = scale_and_clip
        self.output_sample_rate = output_sample_rate
        self.overwrite_existing = overwrite_existing
        self.filter = filter
        self.calculated_channel_config = calculated_channel_config

        self.filtersettings = filtersettings

        if data_format == "hdf5":
            self.write_function = self.write_record_to_database_hdf5
        elif data_format == "parquet":
            self.write_function = self.write_record_to_database_parquet
        else:
            self.log_error("Invalid data format. Must be one of [hdf5, parquet].")
            exit(1)

        assert os.path.exists(self.dataset_path), f"Path {self.dataset_path} does not exist"

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

    def log_info(self, msg):
        self.logger.log(msg, self.dataset_name(), self.subject_context, self.record_context, EventSeverity.Info)

    def log_warning(self, msg):
        self.logger.log(msg, self.dataset_name(), self.subject_context, self.record_context, EventSeverity.Warning)

    def log_error(self, msg):
        self.logger.log(msg, self.dataset_name(), self.subject_context, self.record_context, EventSeverity.Error)

    def __check_paths(self, paths_dict):
        for k in paths_dict.keys():
            record_list = paths_dict[k]

            for r in record_list:
                name, psg, hyp = r

                for file_path in [psg, hyp]:
                    assert os.path.exists(file_path), f"Datapath: {file_path} was not found"

    def add_calculated_channels(self, data):
        config = self.calculated_channel_config

        if config is None:
            return data

        if config.drop_existing:
            new_data = dict()
        else:
            new_data = data

        for calculation in config.rereferences:
            first = calculation.first.get_mapping()
            second = calculation.second.get_mapping()
            calculated_key = calculation.result.get_mapping()

            try:
                new_data[calculated_key] = data[first] - data[second]
            except KeyError as e:
                self.log_error(f"Skipping calculation of {calculated_key} due to error: {e}")
                continue

        return new_data

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

            assert len(data) == y_len * sample_rate * 30, "Length of data does not match the length of labels"

            data = remove_dc(data)

            data = resample_channel(data, output_rate=self.output_sample_rate, source_sample_rate=sample_rate)

            new_dict[new_key] = data

        try:
            new_dict = self.add_calculated_channels(new_dict)
        except Exception as e:
            raise e

        if len(new_dict.keys()) == 0:
            raise Exception("No available data channels after channel calculation")

        for key in new_dict.keys():
            try:
                data = new_dict[key]
            except KeyError:
                continue

            if self.filter:
                data = filter_channel(data, self.output_sample_rate, self.filtersettings)

            if self.scale_and_clip:
                data = scale_channel(data)
                data = clip_channel(data)

            new_dict[key] = data

        return new_dict

    def __map_labels(self, labels):
        return list(map(lambda x: self.label_mapping()[x], labels))

    def save_dataset_metadata(self):
        filtering_used = self.filter
        filtersettings = self.filtersettings
        scaled_and_clipped = self.scale_and_clip
        output_samplerate = self.output_sample_rate

        file_path = f"{self.output_path}/{self.dataset_name()}.hdf5"

        try:
            with File(file_path, "a") as f:
                meta_grp = f.create_group("meta")

                filter_grp = meta_grp.create_group("filtersettings")
                filter_grp.create_dataset("filter_applied", data=filtering_used)
                filter_grp.create_dataset("order", data=filtersettings.order)
                filter_grp.create_dataset("cutoffs", data=filtersettings.cutoffs)
                filter_grp.create_dataset("type", data=filtersettings.type)

                meta_grp.create_dataset("output_samplerate", data=output_samplerate)
                meta_grp.create_dataset("scaled_and_clipped", data=scaled_and_clipped)

                self.log_info("Successfully saved metadata")
        except Exception as error:
            self.log_error(f"Could not save metadata due to error: {error}")

    def write_record_to_database_parquet(self, output_basepath, subject_number, record_number, x, y):
        """
        Function to write PSG data along with labels to the shared database containing all datasets in Parquet format.
        """

        psg_table = pa.table(x)
        hyp_table = pa.table({"labels": y})

        output_path = output_basepath + f"s_{subject_number}/r_{record_number}/"

        Path(output_path).mkdir(parents=True, exist_ok=True)  # Because Parquet does not create directory
        pq.write_table(psg_table, output_path + "psg.parquet")
        pq.write_table(hyp_table, output_path + "hypnogram.parquet")

    def write_record_to_database_hdf5(self, output_basepath, subject_id, record_id, x, y, meta):
        """
        Function to write PSG data along with labels to the shared database containing all datasets in HDF5 format.
        """
        Path(output_basepath).mkdir(parents=True, exist_ok=True)

        file_path = f"{output_basepath}{self.dataset_name()}.hdf5"

        with File(file_path, "a") as f:
            data_group = f.require_group("data")

            # Require subject group, since we want to use the existing one, if subject has more records
            grp_subject = data_group.require_group(f"{subject_id}")
            subgrp_record = grp_subject.create_group(f"{record_id}")

            subsubgrp_psg = subgrp_record.create_group("psg")

            for channel_name, channel_data in x.items():
                subsubgrp_psg.create_dataset(channel_name, data=channel_data)

            subgrp_record.create_dataset("hypnogram", data=y)

            metagroup = subgrp_record.create_group("meta")

            for k in self.meta.keys():
                metagroup.create_dataset(k, data=self.meta[k])

            self.log_info("Successfully wrote record to hdf5 file")

    def does_exist(self, file_path, subject_number, record_number) -> bool:
        file_exists = os.path.exists(file_path)

        if file_exists:
            with File(file_path, "r") as f:
                if subject_number not in f.keys():
                    return False

                subject_group = f[subject_number]

                if str(record_number) not in subject_group.keys():
                    return False

                return True
        else:
            return False

    def port_data(self):
        paths_dict = self.list_records(basepath=self.dataset_path)

        self.__check_paths(paths_dict)

        subject_list = list(paths_dict.keys())[: self.max_num_subjects]

        if len(subject_list) == 0:
            self.log_error("No data found, could not port dataset")
            return

        file_path = f"{self.output_path}/{self.dataset_name()}.hdf5"
        exists = os.path.exists(file_path)

        if exists:
            self.log_warning("HDF5 file already exists. Removing it")
            os.remove(file_path)

        subject_list = list(paths_dict.keys())[: self.max_num_subjects]

        if len(subject_list) == 0:
            self.log_error("No records found in the record list. No dataset created")
            return

        for subject_number in subject_list:
            for record in paths_dict[subject_number]:
                print("\n")

                self.meta = {}

                record_name, psg_path, hyp_path = record

                if (not self.overwrite_existing) and (self.does_exist(file_path, subject_number, record_name)):
                    self.log_info("Skipping record, since it already exists")
                    continue

                self.subject_context = subject_number
                self.record_context = record_name
                psg = self.read_psg((psg_path, hyp_path))

                if psg is None:
                    self.log_error("PSG could not be read, skipping it")
                    continue

                x, y = psg

                try:
                    x = self.__map_channels(x, len(y))
                except Exception as e:
                    self.log_error(f"Could not map data due to error: {e}")
                    continue

                y = self.__map_labels(y)

                self.write_function(f"{self.output_path}/", subject_number, record_name, x, y, self.meta)

                self.subject_context = None
                self.record_context = None

        # self.save_dataset_metadata()
        self.log_info("Successfully ported dataset")
        self.logger.final(self.dataset_name())
