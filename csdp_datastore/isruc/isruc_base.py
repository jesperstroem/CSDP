import os
from abc import abstractmethod

import numpy as np
import scipy.io

from csdp_datastore.base import BaseDataset

from ..models import Labels, Mapping, TTRef


class Isruc_base(BaseDataset):
    """
    ABOUT THIS DATASET

    """

    def sample_rate(self):
        return 200

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    def label_mapping(self):
        return {
            "0": Labels.Wake,
            "1": Labels.N1,
            "2": Labels.N2,
            "3": Labels.N3,
            "5": Labels.REM,
        }

    def channel_mapping(self):
        return {
            "F3_A2": Mapping(TTRef.F3, TTRef.RPA),
            "C3_A2": Mapping(TTRef.C3, TTRef.RPA),
            "F4_A1": Mapping(TTRef.F4, TTRef.LPA),
            "C4_A1": Mapping(TTRef.C4, TTRef.LPA),
            "O1_A2": Mapping(TTRef.O1, TTRef.RPA),
            "O2_A1": Mapping(TTRef.O2, TTRef.LPA),
            "ROC_A1": Mapping(TTRef.ER, TTRef.LPA),
            "LOC_A2": Mapping(TTRef.EL, TTRef.RPA),
        }

    def list_records(self, basepath):
        paths_dict = {}

        record_paths = os.listdir(basepath)

        for path in record_paths:
            # Fucking ugly and hacky, delete ASAP
            if "ipynb_checkpoints" in path:
                continue

            recordpath = basepath + path + "/"
            datapath = recordpath + "subject" + path + ".mat"
            labelpath = recordpath + path + "_" + "1.txt"

            paths_dict[path] = [("1", datapath, labelpath)]

        return paths_dict

    def read_psg(self, record):
        datapath, labelpath = record

        x = dict()

        mat = scipy.io.loadmat(datapath)
        for key in self.channel_mapping().keys():
            # 30 epochs of data was removed due to noise resulting in more labels
            chnl = np.array(mat[key]).flatten()
            x_len = len(chnl)
            x[key] = (chnl, self.sample_rate())

        with open(labelpath, "r") as f:
            y = list(map(lambda x: x[0], f.readlines()))
            y_trunc = y[: int(x_len / self.sample_rate() / 30)]
            trunc_len = len(y) - len(y_trunc)
            if trunc_len > 31:
                self.log_warning(f"Length of truncated y was: {trunc_len}.")
                return None

        return x, y_trunc
