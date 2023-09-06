
#%%
#import os
#import mne_bids as mb
import numpy as np
from scipy.interpolate import interp1d 
import scipy.signal

import sys


#%%


def spectrogram_matlabCopy(x=None, win=None, noverlap=None, nfft=None,fs=None, sides='onesided'):
    """
    light weight approach to spectrogram calculation. basically a slightly polished version of 
    scipy.signal._fft_helper
    Syntax: times,  freqs, Sxx = spectrogram_lightweight(x=None, win=None, noverlap=None, nfft=None,fs=None, sides='onesided')
    sides='onesided' or 'twosided'
    noverlap is in samples
    nfft is in samples (if  nfft is larger than win, then the signal is zero padded)
    """

    #make sure everything got passed:
    assert x is not None
    assert win is not None
    assert noverlap is not None
    assert nfft is not None
    assert fs is not None

    if np.isscalar(win):
        nperseg=win
        win=np.ones(nperseg)
    else:
        nperseg=len(win)

    assert len(win)==nperseg

    # make sure x is a 1D array, with optional singular dimensions
    assert len(x)==x.shape[-1]

    #make sure data fits cleanly into segments
    assert (len(x)-nperseg) % (nperseg-noverlap) ==0

    # Created strided array of data segments
    # https://stackoverflow.com/a/5568169
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                strides=strides)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == 'twosided':
        func = np.fft.fft
        freqs = np.fft.fftfreq(nfft, 1/fs)
    elif sides == 'onesided':
        result = result.real
        func = np.fft.rfft
        freqs = np.fft.rfftfreq(nfft, 1/fs)
    else:
        raise ValueError('sides must be twosided or onesided')

    result = func(result, nfft)
    time = np.arange(nperseg/2,
                     x.shape[-1] - nperseg/2 + 1,
                     nperseg - noverlap)/float(fs)

    return result,freqs,time



def returnFilePaths(bidsDir,subjectIds=None,sessionIds=None,taskIds=None,extension='.set'):
    #wrapper for get_entity_vals and BIDSPath, to get all files matching certain ID's
     
    # 

    def debugMessage(input,inputName):
        if type(input) is not list:
            raise Exception( "returnFilepaths expects a list or None for " + inputName + " Id's. Consider enclosing id in '[]'" )


    #list of subjects:
    if not subjectIds:
        subjectIds=mb.get_entity_vals(bidsDir,'subject')
        if len(subjectIds)==0:
            subjectIds=[None]
    debugMessage(subjectIds,'subject')


    #list of sessions:
    if not sessionIds:
        sessionIds=mb.get_entity_vals(bidsDir,'session')
        if len(sessionIds)==0:
            sessionIds=[None]
    debugMessage(sessionIds,'session')

    #list of tasks:
    if not taskIds:
        taskIds=mb.get_entity_vals(bidsDir,'task')
        if len(taskIds)==0:
            taskIds=[None]
    debugMessage(taskIds,'task')

    print('Subject Ids:',subjectIds)
    print('Session Ids:',sessionIds)
    print('Task ids:',taskIds)

    #and here we just check and add all possible combinations:
    filePaths=[]
    for sub in subjectIds:
        for ses in sessionIds:
            for task in taskIds:
                try:
                    temp=mb.BIDSPath(root=bidsDir,subject=sub,session=ses,task=task,datatype='eeg',extension=extension,check=False)
                    if os.path.isfile(str(temp)):
                        filePaths.append(str(temp))
                except Exception as error:
                    print(error)
                    print(sub,ses,task)



    return filePaths


def findRuns(input):
    sequence=np.asarray(input)
    assert ~(sequence.all() | ((1-sequence).all())),'Sequence is all 0 or all 1'

    changes=np.diff([0, *sequence, 0])
    runStarts=(changes>0).nonzero()[0]
    runEnds=(changes<0).nonzero()[0]
    runLengths=runEnds-runStarts
    assert all(runLengths>0)

    return runStarts, runLengths

def interpolateOverNans(allDeriv,fs):
    #we can't have nans at the end:
    allDeriv[np.isnan(allDeriv[:,0]),0]=0
    allDeriv[np.isnan(allDeriv[:,-1]),-1]=0


    for iDeriv in range(allDeriv.shape[0]):
        
        nanSamples=np.isnan(allDeriv[iDeriv,:]).nonzero()[0]

        if nanSamples.size>0:
            [nanStart, nanDur]=findRuns(np.isnan(allDeriv[iDeriv,:]))
            nanDur=nanDur-1
            realSamples=np.unique([nanStart-1, (nanStart+nanDur)+1])
            
            distanceToReal=nanSamples*0
            counter=0
            for iRun in range(len(nanDur)):
                distanceToReal[range(counter,counter+nanDur[iRun])]=[*range(int(np.floor(nanDur[iRun]/2))), *range(int(np.ceil(nanDur[iRun]/2)),0,-1) ]
                counter=counter+nanDur[iRun]
           
            interpValues=interp1d(realSamples,allDeriv[iDeriv,realSamples])(nanSamples)
            interpValues=interpValues*np.exp(-distanceToReal/(fs*1))
            
            allDeriv[iDeriv,nanSamples]=interpValues

    return allDeriv

def next_power_of_2(x):
    return 1 if x == 0 else int(2**(np.ceil(np.log2(x))))

def time_freq_images(signal, fs_fourier, overlap,win_size):
    #print(fs_fourier)
    N = signal.shape[1]
    nfft = next_power_of_2(win_size*fs_fourier)
    X_eeg = []

    window=scipy.signal.windows.hamming(int(win_size *fs_fourier))

    for i in range(N):       
        sxx,freqs,times =spectrogram_matlabCopy(signal[:,i], win=window,nfft=nfft,
        fs=fs_fourier, noverlap=overlap*fs_fourier)

        sxx = 20*np.log10(np.abs(sxx)) #+1 to avoid log(0)
        # sxx = sxx.T
        X_eeg.append(sxx)

    return np.array(X_eeg),freqs,times

def markMerger(markings,threshold):
    #threshold is the ratio of gap size to the size of the surrounding markings
    #if the gap is much smaller than the surrounding markings, it is ignored
    #and the markings are merged

    #markings should be only 0s and 1s:
    assert ((markings==0) | (markings==1)).all()

    #the same logic is applied twice, to merge any gaps that are created by the first pass
    for iMerge in range(2):    
        if np.min(markings)==np.max(markings):
            return markings

        markStarts, markDurations=findRuns(markings)
        markEnds=markStarts+markDurations-1
        gapSizes=markStarts[1:]-markEnds[0:-1]-1
        gapStarts=markEnds[0:-1]+1
        
        gapRatios=gapSizes/(np.minimum(markDurations[0:-1],markDurations[1:]))
        
        gapsToPaveOver=gapRatios<=threshold
        #the logic here is that if the size of  the gap is much smaller than
        #the surrounding markings, the gap should be ignored, merging the
        #markings
        
        #pave over gaps:    
        for iGap in (gapsToPaveOver).nonzero()[0]:
            markings[gapStarts[iGap]:(gapStarts[iGap]+gapSizes[iGap])]=1
    
    return markings

import re

def directory_spider(input_dir, path_pattern="", file_pattern="", maxResults=500):
    file_paths = []
    if not os.path.exists(input_dir):
        raise FileNotFoundError("Could not find path: %s"%(input_dir))
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if re.search(path_pattern, dirpath):
            file_list = [item for item in filenames if re.search(file_pattern,item)]
            file_path_list = [os.path.join(dirpath, item) for item in file_list]
            file_paths += file_path_list
            if len(file_paths) > maxResults:
                break
    return file_paths[0:maxResults]

#%%

if __name__ == "__main__":

    # e=np.sin(np.arange(0,2*np.pi,0.1))
    # e=np.expand_dims(e,0)
    # import matplotlib.pyplot as plt

    # plt.plot(e[0,:])
    # e[0,10:30]=np.nan
    # plt.plot(e[0,:])
    # plt.plot(interpolateOverNans(e,10)[0,:])



    merged=markMerger(np.array([0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0]),1)
    print(merged)

    # a,b=findRuns([0,1,1,1,0,1])
    # print(a,b)


    # def mytransform(raw):
    #     raw.filter(0.1,40)
    #     raw._data=raw._data*1e6
    #     return raw

    # bidsPath="C:/Users/au207178/OneDrive - Aarhus Universitet/forskning/EEGprediction/localData/train/"


    # subjectIds=mb.get_entity_vals(bidsPath,'subject',with_key=False)
    # trainIds=subjectIds.copy()
    # trainIds.pop(1)
    # trainPaths=returnFilePaths(bidsPath,trainIds,sessionIds=['001', '002', '003', '004'])



        