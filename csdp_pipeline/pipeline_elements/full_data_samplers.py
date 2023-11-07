from csdp_pipeline.pipeline_elements.pipe import IPipe
import h5py
import json
import torch
import math
import random
import numpy as np

class Full_Eval_Dataset_Sampler(IPipe):
    def __init__(self,
                file_path: str,
                records_to_pick: list[str] = None,
                channels_to_pick: list[str] = None,
                split_file_path: str = None,
                split_type: str = "val",
                dataset_name: str = None):
        """_summary_

        Args:
            file_path (str): Path to the hdf5 file to sample data from
            channels_to_pick (list[str], optional): A list of channels to pick from the data. Defaults to None which means all channels are sampled.
            split_file_path (str, optional): A filepath to a json split file. Defaults to None which means all subjects are sampled from the data.
            split_type (str, optional): Which split type the dataloader should be. Determines which subjects are sampled from the split_file_path. Defaults to "val".
            dataset_name (str, optional): The name of the dataset specified in the json split file. Only needed if a split file is specified. Defaults to None
        """
        assert split_type == "val" or split_type == "test"
        assert type(file_path) == str

        self.records_to_pick = records_to_pick
        self.channels_to_pick = channels_to_pick
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.file_path = file_path
        self.split_file = split_file_path
        self.record_data, self.record_hyps, self.record_meta = self.__get_data()

    def process(self, index):
        return self.record_data[index], self.record_hyps[index], self.record_meta[index]

    def __get_data(self):
        with h5py.File(self.file_path, "r") as hdf5:

            if self.split_file != None:
                with open(self.split_file, "r") as splitfile:
                    splitdata = json.load(splitfile)
                    try:
                        # Try finding the correct split
                        sets = splitdata[self.dataset_name]
                        subs = sets[self.split_type]
                    except:
                        exit()
            else:
                subs = list(hdf5.keys())

            record_data = []
            record_hyps = []
            record_meta = []
            
            for subj_key in subs:
                try:
                    subj = hdf5[subj_key]

                    rec_keys = subj.keys()

                    if self.records_to_pick != None:
                        rec_keys = [k for k in rec_keys if k in self.records_to_pick]
                        print(f"Sampling from records {rec_keys}")

                    for rec_key in rec_keys:
                        rec = subj[rec_key]

                        hyp = rec["hypnogram"][()]
                        psg = rec["psg"]

                        data = []

                        psg_keys = psg.keys()

                        if self.channels_to_pick != None:
                            psg_keys = [k for k in psg_keys if k in self.channels_to_pick]

                        for c in psg_keys:
                            channel_data = psg[c][()]
                            
                            channel_data = torch.tensor(channel_data)

                            data.append(channel_data)
                        
                        data = torch.stack(data, dim=0)
                        hyp = torch.tensor(hyp, dtype=torch.int64)
                        
                        record_data.append(data)
                        record_hyps.append(hyp)
                        record_meta.append(f"{subj_key}-{rec_key}")

                except:
                    print(f"Did not find subject {subj_key} in dataset EESM with split type {self.split_type}")
                    continue
        
        return record_data, record_hyps, record_meta

class Full_Train_Dataset_Sampler(IPipe):
    def __init__(self,
                 file_path: str,
                 window_size: int,
                 records_to_pick: list[str] = None,
                 channels_to_pick: list[str] = None,
                 split_file_path: str = None,
                 dataset_name: str = None):
        """_summary_

        Args:
            file_path (str): Path to the hdf5 file to sample data from
            window_size (int): Size of the window to sample from the data.
            channels_to_pick (list[str]): A list of channels to pick from the data. Defaults to None which means all channels are sampled.
            split_file_path (str, optional): A filepath to a json split file. If specified, all the subjects under "train" will be sampled. Defaults to None which means all subjects are sampled from the data.
            dataset_name (str, optional): The name of the dataset specified in the json split file. Only needed if a split file is specified. Defaults to None
        """

        self.records_to_pick = records_to_pick
        self.window_length = window_size
        self.file_path = file_path
        self.channels_to_pick = channels_to_pick
        self.split_file = split_file_path
        self.dataset_name = dataset_name
        self.data, self.hyp, self.window_counts, self.data_indexes = self.__get_data()

        self.num_windows = self.window_counts[-1]

    def process(self, index):

        record_index = self.data_indexes[index]

        hyp = self.hyp[record_index]
        data = self.data[record_index]

        #print(f"Third time: {time.time() - start}")
        num_epochs = hyp.shape[0]

        # Get how many windows we are offset for the given record
        window_offset = index - self.window_counts[record_index]

        # Calculate how many "extra" epochs there are for the given record
        rest_epochs = num_epochs - (math.floor(num_epochs/self.window_length)*self.window_length)

        # Calculate a random offset in epochs
        random_epoch_offset = random.randint(0, rest_epochs)

        # Calculate the first and last epoch to pick out for this sample
        y_start_idx = (window_offset*self.window_length) + random_epoch_offset
        y_end_idx = y_start_idx + self.window_length

        # Do the same, but for the data
        x_start_idx = y_start_idx*30*128
        x_end_idx = x_start_idx + (self.window_length*128*30)

        # Pick the data and return it.
        x_sample = data[:,x_start_idx:x_end_idx]
        y_sample = hyp[y_start_idx:y_end_idx]

        return x_sample, y_sample, ""
    
    def __get_data(self):
        with h5py.File(self.file_path, "r") as hdf5:

            if self.split_file != None:
                with open(self.split_file, "r") as splitfile:
                    splitdata = json.load(splitfile)
                    try:
                        # Try finding the correct split
                        sets = splitdata[self.dataset_name]
                        subs = sets["train"]
                    except:
                        # If none is configured, take all subjects
                        exit()
            else:
                subs = list(hdf5.keys())

            record_data = []
            record_hyps = []
            window_count = [0]
            data_indexes = []

            record_counter = 0
            
            for subj_key in subs:
                try:
                    subj = hdf5[subj_key]

                    rec_keys = subj.keys()

                    if self.records_to_pick != None:
                        rec_keys = [k for k in rec_keys if k in self.records_to_pick]
                        print(f"Sampling from records {rec_keys}")

                    for rec_key in rec_keys:
                        rec = subj[rec_key]

                        hyp = rec["hypnogram"][()]
                        psg = rec["psg"]

                        data = []

                        psg_keys = psg.keys()

                        if self.channels_to_pick != None:
                            psg_keys = [k for k in psg_keys if k in self.channels_to_pick]

                        for c in psg_keys:
                            
                            channel_data = psg[c][()]

                            whole_windows = math.floor(len(channel_data)/128/30/self.window_length)

                            data.append(channel_data)
                        
                        record_data.append(torch.tensor(np.array(data)))
                        record_hyps.append(torch.tensor(hyp, dtype=torch.int64))

                        window_count.append(whole_windows+window_count[-1])

                        data_indexes = data_indexes + ([record_counter] * whole_windows)

                        record_counter += 1
                except:
                    print(f"Did not find subject {subj_key} in dataset EESM with split type TRAIN")
                    continue
  
        return record_data, record_hyps, window_count, data_indexes