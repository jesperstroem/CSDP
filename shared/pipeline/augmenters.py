# Based on augmenters from U-Time/U-Sleep model
# Source: https://github.com/perslev/U-Time/blob/main/utime/augmentation/augmenters.py

from abc import ABC, abstractmethod
import numpy as np
import torch
import os
import sys
import random
sys.path.append(os.path.abspath('..'))
from shared.pipeline.pipe import IPipe # This only works from usleep folder or equivalent
from shared.pipeline import pipe # TODO: Delete


# TODO: aug_weight!

class OldGlobalAugmenter(ABC):
    """
    Applying augmentation to random elements across the whole batch
    """
    def __init__(self, apply_prob) -> None:
        self.apply_prob = apply_prob
    

    def augment(self, x):
        aug_mask = np.random.rand(len(x)) <= self.apply_prob

        for i, aug in enumerate(aug_mask):
            if not aug:
                continue
                
            x[i] = self.augmentation_function(x[i])
            
            # TODO: aug_weight?
        return x

    
    @abstractmethod
    def augmentation_function(self, x):
        pass
    
    
class GlobalAugmenter(ABC):
    """
    Applying augmentation to all elements in exactly one channel, chosen randomly
    """
    def __init__(self, apply_prob) -> None:
        self.apply_prob = apply_prob
    

    def augment(self, x):
        aug_idx = random.randint(0, len(x) - 1)
        
        augmented_chan = x[aug_idx]
        
        self.augmentation_function(augmented_chan)
        
        r = x[aug_idx]
        r[:] = self.augmentation_function(r)
        
        return x

    
    @abstractmethod
    def augmentation_function(self, x):
        pass
    

class GlobalGaussianNoise(GlobalAugmenter):
    def __init__(self, apply_prob, sigma, mean) -> None:
        super().__init__(apply_prob)
        
        self.mean = mean
        self.sigma = sigma


    def augmentation_function(self, x):
        gaussian_noise = torch.Tensor(np.random.normal(loc=self.mean, scale=self.sigma, size=x.shape))
        
        return gaussian_noise


class RegionalAugmenter(ABC):
    """
    Applying augmentation to all channels in a subset of batch
    """
    def __init__(self, min_frac, max_frac, apply_prob) -> None:
        self.apply_prob = apply_prob
        self.min_frac = min_frac
        self.max_frac = max_frac

        assert self.min_frac > 0, "min_frac must be > 0."
        assert self.max_frac <= 1, "max_frac must be <= 1."


    def get_aug_length(self, x_length):
        min_idx = int(self.min_frac * x_length)
        max_idx = int(self.max_frac * x_length)

        aug_length = int(np.random.uniform(min_idx, max_idx)) # Random length between min and max (but should 0 be included?

        return aug_length
    

    def get_start_point(self, x_length):
        return np.random.randint(0, x_length - 1)
    

    def augment(self, x):  
        x_length = len(x[0])
        
        start = self.get_start_point(x_length)
        aug_length = self.get_aug_length(x_length)
        
        if aug_length == 0: # Should aug length even have the possibility to be 0?
            return x
        
        for chan in x:
            r = chan[start:start+aug_length]
            r[:] = self.augmentation_function(r)
        
        return x
    

    @abstractmethod
    def augmentation_function():
        pass


class RegionalGaussianNoise(RegionalAugmenter):
    def __init__(self, min_frac, max_frac, apply_prob, sigma, mean) -> None:
        super().__init__(min_frac, max_frac, apply_prob)

        self.mean = mean
        self.sigma = sigma

    def augmentation_function(self, x):
        noise = torch.Tensor(np.random.normal(loc=self.mean, scale=self.sigma, size=x.shape))
        
        return noise
    
    
class Augmenter(IPipe):
    def __init__(self, min_frac, max_frac, apply_prob, sigma, mean):
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.apply_prob = apply_prob
        self.sigma = sigma
        self.mean = mean
        
        self.global_gaussian_noise = GlobalGaussianNoise(self.apply_prob, self.sigma, self.mean)
        self.regional_gaussian_noise = RegionalGaussianNoise(self.min_frac, self.max_frac, self.apply_prob, self.sigma, self.mean)
        
    def process(self, batch):
        x_eeg, x_eog, y, tags = batch
        x = torch.cat((x_eeg, x_eog), dim=0)
        
        if random.random() < self.apply_prob:
            self.global_gaussian_noise.augment(x)
        else:
            self.regional_gaussian_noise.augment(x)
        
        eeg = torch.unsqueeze(x[0], 0)
        eog = torch.unsqueeze(x[1], 0)
        return (eeg, eog, y, tags)

class LSeq_Augmenter(Augmenter):
    def adapt(self, x):
        eeg, eog, y, tags = x
        eeg = torch.unsqueeze(eeg, 0)
        eog = torch.unsqueeze(eog, 0)
        return (eeg, eog, y, tags)
    
if __name__ == "__main__":
    print("Hello world aug")
    x_batch = torch.rand(2, 16)
    #print(x_batch)
    batch = (x_batch, None, None)

    pipes = [
        Augmenter(min_frac=0.001, max_frac=0.3, apply_prob=0.1, sigma=1, mean=0)
    ]

    p = pipe.Pipeline(pipes, batch)
    
    out_batch = p.get_batch()

    #print(out_batch[0])

