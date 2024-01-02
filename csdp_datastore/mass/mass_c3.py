import os
from h5py import File
import scipy.io
import numpy as np
import pandas as pd
import mne
import re
import math

from .mass_base import Mass_base

class MASS_C3(Mass_base):
    """
    ABOUT THIS DATASET 
    
    """
        
    def dataset_name(self):
        return "mass_c3"