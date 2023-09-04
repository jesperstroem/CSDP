import os
from h5py import File
import scipy.io
import numpy as np
import pandas as pd
import mne
import re
import math

from .isruc_base import Isruc_base

class ISRUC_SG2(Isruc_base):
    """
    ABOUT THIS DATASET 
    
    """
        
    def dataset_name(self):
        return "isruc_sg2"
    
    # Overridden because of filenames being different.
    def list_records(self, basepath):
        paths_dict = {}

        record_paths = os.listdir(basepath)

        for path in record_paths:
            if "ipynb_checkpoints" in path:
                continue

            recordpath = basepath+path+'/'
            datapath = recordpath+"subject"+path+".mat"
            labelpath = recordpath+"1_1.txt"

            paths_dict[path] = [(datapath, labelpath)]

        return paths_dict