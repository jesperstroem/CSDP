#%% 
# Import libraries
import sys
# sys.path.append('C:\\Users\\kaare\\OneDrive - Aarhus universitet\\forskning\\customPythonScripts')
# from kbm_spectrogram import spectrogram_matlabCopy as specMatlab

import mne
import os
from utils_kaare import interpolateOverNans,time_freq_images
import numpy as np
import torch

import scipy.io
import hdf5storage

#%% load  data
raw = mne.io.read_raw_eeglab('C:\\Users\\au207178\\OneDrive - Aarhus universitet\\forskning\\sleepInOrbit\\sleepInOrbitPilots\\derivatives\\cleaned_1\\sub-001\\ses-002\\EEGcleaned1B_2.set', preload=True)
raw._data=raw._data*1e6

targetFolder='C:\\Users\\au207178\\OneDrive - Aarhus universitet\\forskning\\sleepInOrbit\\sleepInOrbitPilots\\derivatives\\spectrograms_1\\sub-001\\ses-002\\'

#%% 

assert raw.info['sfreq']==250

data=raw.get_data(picks=[*range(0,8)]) #"*" unpacks the range and makes it a real list

###### determine single ear epochs: ######
epochLength=int(30*raw.info['sfreq'])
nEpochs=int(np.floor(data.shape[1]/epochLength))
leftnans=np.all(np.isnan(data[0:4,0:int(epochLength*nEpochs)]),axis=0) #true if all channels are nan
leftnans=np.all(np.reshape(leftnans,(epochLength,nEpochs)),axis=0) #true if all samples in an epoch are nan
rightnans=np.all(np.isnan(data[4:8,0:int(epochLength*nEpochs)]),axis=0) #true if all channels are nan
rightnans=np.all(np.reshape(rightnans,(epochLength,nEpochs)),axis=0) #true if all samples in an epoch are nan
singleEarEpochs=np.any(np.concatenate(((leftnans.reshape(1,-1)),rightnans.reshape(1,-1)),axis=0),axis=0) #true if any epoch is missing an entire ear

####### interpolate over nans ######
data=interpolateOverNans(data,raw.info['sfreq'])

####### downsample from 250 hz to 100: ######
data=scipy.signal.resample_poly(data,2,5,axis=1)
srate=100 

lr=np.nanmean(data[0:4,:],axis=0)-np.nanmean(data[4:,:],axis=0)

#epoch data:
epochLength=srate*30
nEpochs=int(np.floor(data.shape[1]/epochLength))
lr=lr[0:nEpochs*epochLength]
lr=np.reshape(lr,(epochLength,nEpochs),order='F') 

#calculate spectrograms:
eeg_spectrograms,freqs,times =time_freq_images(lr, fs_fourier=srate, overlap=1,win_size=2)
assert nEpochs==eeg_spectrograms.shape[0]
eeg_spectrograms[np.isinf(eeg_spectrograms)] = 0

#cast to float32:
eeg_spectrograms=np.single(eeg_spectrograms)

#discard single-ear epochs or epochs with missing labels:
epochsToKeep=np.logical_not(singleEarEpochs)
eeg_spectrograms=eeg_spectrograms[epochsToKeep,:,:]

#save data:
os.makedirs(targetFolder,exist_ok=True)
# labels=np.ones((nEpochs,1))*-1

#discard single-ear epochs or epochs with missing labels:
# labels=labels[epochsToKeep]
eeg_spectrograms=eeg_spectrograms[epochsToKeep,:,:]

# y=torch.nn.functional.one_hot(torch.tensor(labels)-1, num_classes=5).numpy()

hdf5storage.write({'X':eeg_spectrograms,'freqs':freqs,'times':times,'epochsKept':epochsToKeep}, '.', targetFolder+'/eeg_lr_mat.mat', matlab_compatible=True)


