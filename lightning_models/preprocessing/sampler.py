# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import torch
import os
import numpy as np

from pathlib import Path
import pandas as pd
from .storage import StorageAPI

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class Sampler():
    def __init__(self, rootpath):
        self.rootpath = rootpath
        self.files = pd.DataFrame(self.__list_files())
    
    def random_sample(self):
        '''
        Chooses a random record from each dataset and draws num_samples from the file.
        Changes the read_offset of the file
        '''
        
        #No more data
        if(len(self.files) == 0):
            return None
        
        df = self.files.groupby('dataset', group_keys=False).apply(lambda x: x.sample(frac=0.6))

        dirs = df['dir']
        filenames = df['filename']
        offsets = df['read_offset']
        indexes = df.index
        
        data = []
        labels = []
        
        for i in range(len(filenames)):
            dir = dirs.iloc[i]
            filename = filenames.iloc[i]
            offset = offsets.iloc[i]
            index = indexes[i]
            
            (x, y) = StorageAPI.load_data(dir, filename)

            number_of_epochs = len(x) - offset
            recordsToPull = 200*1
            
            x = torch.tensor(np.array(x))
            y = torch.tensor(y)
            
            x = x[range(offset, offset+recordsToPull)]
            y = y[range(offset, offset+recordsToPull)]
            
            remaining_epochs = number_of_epochs - recordsToPull
            
            data.append(x)
            labels.append(y)
            
            if remaining_epochs < recordsToPull:
                self.files = self.files.drop(index)
            else:
                self.files.loc[index, 'read_offset'] = offset + recordsToPull

        data = torch.cat(data, dim = 0)
        labels = torch.cat(labels, dim = 0)
        
        return data, labels
    
    def __list_files(self):
        f = []
    
        for dir, subdir, files in os.walk(self.rootpath):
            for filename in files:
                appendix = remove_prefix(dir, self.rootpath)
                
                groups = appendix.split("/")
                
                dataset = groups[1]
                subject = groups[2]
                record = groups[3]

                f.append({'dataset': dataset,
                          'subject': subject,
                          'record': record,
                          'dir': dir,
                          'filename': filename,
                          'read_offset': 0})
                
        return f
                
if __name__ == '__main__':
    s = Sampler("../../../spectrograms")