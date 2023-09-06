# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:46:23 2023

@author: Jesper Strøm
"""

from .sampler import Sampler
from .storage import StorageAPI
import torch

class Chunker():
    def __init__(self,
                 source_path,
                 target_path,
                 chunk_size):
        self.target_path = target_path    
        self.sampler = Sampler(source_path)
        self.chunk_size = chunk_size
        
    def create_chunks(self):
        
        data = torch.tensor([])
        labels = torch.tensor([])
        chunk_no = 0
        
        while True:
            
            s = self.sampler.random_sample()
            
            if s == None:
                print("No more data to sample from, chunking is done")
                self.__save_chunk(data, labels, chunk_no)
                break
            
            (x,Y) = s
            x = torch.reshape(x, (-1, 200, 29, 129))
            Y = torch.reshape(Y, (-1, 200))
                              
            if (len(data) + len(x)) > self.chunk_size:
                remainder_x, remainder_Y = self.__finalize_chunk(x, Y, data, labels, chunk_no)
                
                chunk_no += 1
                
                # Clear memory and add the saved remainder
                data = torch.tensor([])
                labels = torch.tensor([])
                data = torch.cat((data, remainder_x), dim=0)
                labels = torch.cat((labels, remainder_Y), dim=0)
            else:
                data = torch.cat((data,x), dim=0)
                labels = torch.cat((labels,Y), dim=0)
            
            print('Loaded data to chunk {chunk_no}, size is now {size}'.format(chunk_no=chunk_no, size=len(data)))
            
    def __save_chunk(self, data, labels, chunk_no):
        # 3 times only for testing..
        StorageAPI.save_data(self.target_path+'/train', str(chunk_no),(data, labels))
        StorageAPI.save_data(self.target_path+'/validation', str(chunk_no),(data, labels))
        StorageAPI.save_data(self.target_path+'/test', str(chunk_no),(data, labels))
        print('Chunk {} saved!'.format(chunk_no))
        
    def __finalize_chunk(self, sample_x, sample_Y, chunk_x, chunk_Y, chunk_no):
        chunk_remainder = self.chunk_size - len(chunk_x)
        
        # Save what can be added and what goes to next chunk
        chunk1_x = sample_x[range(0,chunk_remainder)]
        chunk2_x = sample_x[range(chunk_remainder, len(sample_x))]
        chunk1_Y = sample_Y[range(0,chunk_remainder)]
        chunk2_Y = sample_Y[range(chunk_remainder, len(sample_Y))]
        
        # Add what can be added
        chunk_x = torch.cat((chunk_x, chunk1_x), dim=0)
        chunk_Y = torch.cat((chunk_Y, chunk1_Y), dim=0)
        
        # Save chunk to disc
        self.__save_chunk(chunk_x, chunk_Y, chunk_no)
        
        return chunk2_x, chunk2_Y

if __name__ == '__main__':
    c = Chunker("C:/repos/Speciale/lseqsleepnet-pytorch/preprocessing/data/spectrograms",
                "C:/repos/Speciale/lseqsleepnet-pytorch/preprocessing/data/chunks",
                chunk_size = 32)
    c.create_chunks()