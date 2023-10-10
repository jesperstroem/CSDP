# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import torch
import numpy as np

from common_sleep_data_pipeline.pipeline_elements.pipe import IPipe
from common_sleep_data_pipeline.preprocessing.spectrogram import create_spectrogram_images

class Spectrogram(IPipe):
    def __init__(self, 
                 win_size = 2,
                 fs_fourier = 100,
                 overlap = 1,
                 sample_rate = 100):
        self.win_size = win_size
        self.fs_fourier = fs_fourier
        self.sample_rate = sample_rate
        self.overlap = overlap
    
    def spectrograms_for_collection(self, coll):
        specs = []
        for i in range(len(coll)):
            _,_,spectrograms = create_spectrogram_images(coll[i],
                                                         self.sample_rate, 
                                                         self.win_size, 
                                                         self.fs_fourier, 
                                                         self.overlap)
            spectrograms = torch.tensor(np.array(spectrograms))
            specs.append(spectrograms)
            
        specs = torch.stack(specs)
        return specs
    
    def process(self, x):
        eegs = x[0]
        eogs = x[1]
        labels = x[2]
        tags = x[3]
        
        # Channels, samples
        assert eegs.dim() == 2 and eogs.dim() == 2

        eegs = eegs.numpy()
        eogs = eogs.numpy()
        
        eegs = self.spectrograms_for_collection(eegs)
        eogs = self.spectrograms_for_collection(eogs)

        return eegs, eogs, labels, tags