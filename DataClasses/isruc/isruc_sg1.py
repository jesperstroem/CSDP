import os
from h5py import File
import scipy.io
import numpy as np
import pandas as pd
import mne
import re
import math

from .isruc_base import Isruc_base

class ISRUC_SG1(Isruc_base):
    """
    ABOUT THIS DATASET 
    
    """
        
    def dataset_name(self):
        return "isruc_sg1"