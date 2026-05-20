'''data set class for sending sleep data to usleep'''

#pylint: disable=invalid-name



#%% set up dataset

import types
import mne
import torch
import numpy as np
import h5py

import pandas as pd
from ..preprocessing.usleep_prep_steps import scale_channel, filter_channel, remove_dc, resample_channel, clip_channel, scale_channel_manual, clip_channels

class sleep_dataset_from_paths(torch.utils.data.Dataset):
    ''' Creates torch dataset from list of EEG files and scoring files.
    Applies standard preprocessing to match usleep requirements.
    Syntax:
    sleep_dataset_from_paths( EEG_paths,L=1,scoring_paths=[], derivations=None,scoring_preprocess=None,hdf5File=None)

    hdf5File is a previous instance of sleep_dataset_from_paths that has been saved to a hdf5 file.
    If hdf5File is not None, the dataset is loaded from the hdf5 file, ignoring the other arguments.
    '''
    def __init__(self, EEG_paths,L=1,scoring_paths=[], ch_names=None,
        derivations=None,scoring_preprocess=None,hdf5File=None,fullRecords=False):
        if hdf5File is None:
            self.constructFromPaths( EEG_paths,L,scoring_paths,ch_names, derivations,scoring_preprocess,fullRecords)
        else:
            self.constructFromHDF5(hdf5File)


    def checkDerivations(self):
        '''checks that the derivations have the correct format'''

        if self.derivations is not None:
            for deriv in self.derivations:
                #derivations should be a list of 2-element tuples
                assert len(deriv)==2

    def get_available_channels(filePaths):
        filePaths=[str(fp) for fp in filePaths]
        names = []

        for path in filePaths:
            raw = sleep_dataset_from_paths.open_eeg_file(path, preload=False)
            ch_names = raw.ch_names
            names.append(ch_names)

        return names

    def constructFromPaths(self, EEG_paths,L,scoring_paths=[], ch_names = None, derivations=None,scoring_preprocess=None,fullRecords=False):
        '''Standard constructor'''

        self.file_paths=[str(fp) for fp in EEG_paths]
        self.scoring_paths=[str(fp) for fp in scoring_paths]
        self.derivations=derivations
        self.L=L
        self.epochLength=128*30 #30 seconds
        self.fullRecords=fullRecords

        if scoring_preprocess is not None:
            self.preprocess_scoring=lambda idx: scoring_preprocess(self,idx)
        else:
            self.preprocess_scoring=self.preprocess_scoring_default

        self.checkDerivations()

        self.data_arrays=[]
        self.nansamples=[]
        if len(self.scoring_paths)>0:
            self.scoring_arrays=[]

        for idx,path in enumerate(self.file_paths):
            tempRaw= sleep_dataset_from_paths.open_eeg_file(path)

            if self.derivations is not None:
                data=np.zeros((len(self.derivations),tempRaw.get_data().shape[1]))
                for didx,deriv in enumerate(self.derivations):
                    data[didx,:]=np.nanmean(tempRaw.get_data(picks=deriv[0]),axis=0)-np.nanmean(tempRaw.get_data(picks=deriv[1]),axis=0)
            elif ch_names is not None:
                data=tempRaw.get_data(picks=ch_names)
            else:
                data=tempRaw.get_data()

            data=self.preprocess_data(data,sfreq=tempRaw.info['sfreq'])
            self.data_arrays.append(data)

            if len(self.scoring_paths)>0:
                #appends to scoring_arrays internally:
                self.preprocess_scoring(idx)

                #make sure that the scoring array matches the data array before moving on:
                assert len(self.scoring_arrays[idx])==self.data_arrays[idx].shape[1]//self.epochLength
            else:
                nSamples=data.shape[1]
                nSamples=(nSamples//self.epochLength)*self.epochLength
                self.data_arrays[idx]=data[:,:nSamples]

        #extract nansamples again:
        #it has to be done after preprocess_scoring, because that might remove some samples
        self.extract_nansamples()

        #create data draws - assumes we will draw L epochs at a time:
        self.create_data_draws()

        #send everything to torch tensors:
        self.data_arrays=[torch.tensor(data, dtype=torch.float32)
                          for data in self.data_arrays]
        if len(self.scoring_paths)>0:
            self.scoring_arrays=[torch.tensor(scoring, dtype=torch.float32)
                                  for scoring in self.scoring_arrays]

    def extract_nansamples(self):
        '''Extracts 'nansamples' from the data arrays to bring them back
        to correct size, and keep track of all-nan epochs. If data is not integer
        number of epochs, the trailing samples are ignored.'''
        nansamples_list=[]
        for idx,data in enumerate(self.data_arrays):
            nDeriv=data.shape[0]//2
            assert nDeriv*2==data.shape[0]
            self.data_arrays[idx]=data[0:nDeriv,:]
            nansamples_list.append(data[nDeriv:,:])

        #determine all-nan epochs based on nanvals:
        self.nanEpochs=[]
        for nansamples in nansamples_list:
            nansamples=nansamples>0
            nEpochs=nansamples.shape[1]//self.epochLength
            nanEpochs=np.zeros((nansamples.shape[0],nEpochs),dtype=bool)
            for iChannel in range(nansamples.shape[0]):

                nanEpochs[iChannel,:]=np.all(nansamples[iChannel,:(nEpochs*self.epochLength)].reshape(self.epochLength,-1),axis=0).reshape(1,-1)

            self.nanEpochs.append(nanEpochs)


    def create_data_draws(self):
        '''Creates a list of indices for drawing data from the dataset.
        Allows __getitem__ to ignore epochlength and number of files'''
        self.dataDraws=[]
        for file_idx,_ in enumerate(self.data_arrays):
            nSamples=self.data_arrays[file_idx].shape[1]
            for i in range(0,nSamples-self.L*self.epochLength,self.epochLength):
                self.dataDraws.append([file_idx,i])

        self.dataDraws=np.array(self.dataDraws)

    def preprocess_data(self,data,sfreq):
        '''Preprocess data to be in line with usleep'''

        nansamples=np.isnan(data)
        data[nansamples]=0
        data=resample_channel(data, 128, sfreq, axis=1)
        nansamples=resample_channel(nansamples.astype(float), 128, sfreq, axis=1)>.5
        data=scale_channel_manual(data)
        data=clip_channels(data)

        #return data with nansamples. nansamples are removed again later:
        return np.vstack((data,nansamples))

    def cutData(self,idx,start,end):
        '''Cuts data to a specific range'''
        self.data_arrays[idx]=self.data_arrays[idx][:,start:end]

    def preprocess_scoring_default(self,idx):
        '''Default scoring loading function'''
        scoring=pd.read_csv(self.scoring_paths[idx],sep='/t')

        #cut data array to match scored duration:
        scoringStart=scoring.iloc[0,0]
        self.data_arrays[idx]=self.data_arrays[idx][:,scoringStart:]

        self.scoring_arrays.append(scoring.iloc[:,2].values)

    def open_eeg_file(filename, preload=True):
        '''Opens data files. Add more cases if needed.'''
        if filename.endswith('.set'):
            return mne.io.read_raw_eeglab(filename,preload=preload,verbose=False)
        elif filename.endswith('.edf'):
            return mne.io.read_raw_edf(filename,preload=preload,verbose=False)
        elif filename.endswith(".vhdr"):
            return mne.io.read_raw_brainvision(filename, preload=preload, verbose=False)
        else:
            print('Unknown file type for file ' + filename)
            raise ValueError('Unknown file type')

    def __len__(self):
        if self.fullRecords:
            return len(self.data_arrays)
        else:
            return self.dataDraws.shape[0]

    def get_minibatch(self, idx):
        '''Returns a minibatch of data. Used for training.'''
        fileIdx=self.dataDraws[idx,0]
        sampleIdx=self.dataDraws[idx,1]
        epochIdx=sampleIdx//self.epochLength

        x = self.data_arrays[fileIdx][:,sampleIdx:sampleIdx+self.epochLength*self.L]

        assert x.shape[1]==self.epochLength*self.L

        if len(self.scoring_paths)>0:
            y = self.scoring_arrays[fileIdx][epochIdx:epochIdx+self.L]
            return x,y,[fileIdx,sampleIdx]
        else:
            return x,{'dataSet':0,'file':fileIdx,'sample':sampleIdx}

    def get_full_record(self, fileIdx):
        '''Returns a full record of data. Used for validation and testing.'''

        x = self.data_arrays[fileIdx]

        if len(self.scoring_paths)>0:
            y = self.scoring_arrays[fileIdx]
            return x,y,fileIdx
        else:
            return x,fileIdx

    def __getitem__(self, idx):
        if self.fullRecords:
            return self.get_full_record(idx)
        else:
            return self.get_minibatch(idx)

    def saveToHDF5(self,hdf5File):
        """
            Save the dataset to a hdf5 file.
            Adds '.hdf5' to the end of the filename if it is not already there.

        """
        #it turns out to be convenvient to save the sizes of the data arrays as well:
        self.data_sizes=[data.shape for data in self.data_arrays]

        with h5py.File(hdf5File, "w") as f:
            for key,value in self.__dict__.items():
                if isinstance(value,list):
                    for idx, data in enumerate(value):
                        f.create_dataset(key+'/'+str(idx),data=data)
                elif isinstance(value,int):
                    f.create_dataset(key,data=value,shape=(1,))
                elif isinstance(value,range):
                    temp=np.asarray(value)
                    f.create_dataset(key,data=temp,shape=temp.shape)
                elif value is None:
                    f.create_dataset(key,data=value,shape=(0,))
                elif isinstance(value,types.FunctionType):
                    pass
                else:
                    f.create_dataset(key,data=value,shape=value.shape)

    def constructFromHDF5(self,hdf5File):
        '''If the constructor is fed an hdf5-file'''
        # load all keys from hdf5 file into dataset:
        with h5py.File(hdf5File, "r") as f:
            for key,value in f.items():
                if isinstance(value,h5py.Group):
                    self.__dict__[key]=[]
                    for subkey in value .keys():
                        self.__dict__[key].append(f[key][subkey][...])
                else:
                    if value.shape[0]==1:
                        self.__dict__[key]=value[0]
                    else:
                        self.__dict__[key]=value[...]

        #convert to torch tensors:
        self.data_arrays=[torch.tensor(data, dtype=torch.float32) for data in self.data_arrays]
        if len(self.scoring_paths)>0:
            self.scoring_arrays=[torch.tensor(scoring, dtype=torch.float32) for scoring in self.scoring_arrays]

        #check that data array sizes are still correct:
        for idx,data in enumerate(self.data_arrays):
            assert (data.shape==self.data_sizes[idx]).all()