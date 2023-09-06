# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:27:04 2023

@author: Jesper Strøm
"""

import numpy as np
import scipy.signal
#import matlab.engine
from scipy.signal import resample_poly
from scipy.interpolate import interp1d 
import torch

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

def create_spectrogram_images(x, sample_rate, win_size, fs_fourier, overlap):
    #if original_sample_rate != target_sample_rate:
    #    x = resample_poly(x, target_sample_rate, original_sample_rate, axis=0)

    epoch_length = sample_rate*30    
    nEpochs = int(np.floor(x.shape[0]/epoch_length))
    x=x[0:nEpochs*epoch_length]
    
    x = np.reshape(x, (-1, epoch_length))
    
    spectrograms = []
    
    for i in range(nEpochs):
        t,f,sxx = create_spectrogram(x[i], win_size, fs_fourier, overlap)

        sxxabs = np.abs(sxx)
        #sxx = sxxabs
        #print(sxx.dtype)
        #sxx = 20*np.log10(sxxabs, where=sxxabs > 0.0)
        sxx = 20*np.log10(sxxabs+0.001)
        spectrograms.append(sxx)
    
    return t, f,spectrograms

def create_spectrogram_matlab(x, win_size, fs_fourier, overlap):
    nfft = next_power_of_2(win_size*fs_fourier)

    window=scipy.signal.windows.hamming(int(win_size *fs_fourier))
    window=window*0+1
    
    eng=matlab.engine.start_matlab()
    
    eng.workspace['signal']=x
    eng.workspace['fs_fourier']=fs_fourier
    eng.workspace['overlap']=overlap 
    eng.workspace['win_size']=win_size
    eng.workspace['nfft']=nfft
    eng.workspace['window']=window
    
    eng.eval('[sxx,f,t]=spectrogram(signal,window,overlap*fs_fourier,nfft,fs_fourier);',nargout=0)
    sxx=eng.workspace['sxx']
    sxxmatlab=np.array(sxx).T
    
    return sxxmatlab

def create_spectrogram(x, win_size, fs_fourier, overlap):
    """
    Takes a one-dimensional time-series signal and converts it to a time-frequency spectrogram image representation
    """
    nfft = next_power_of_2(win_size*fs_fourier)

    window=scipy.signal.windows.hamming(int(win_size *fs_fourier))
    window=window*0+1
    
    t,f,sxx = spectrogram(x,
                          window,
                          overlap*fs_fourier,
                          nfft,
                          fs_fourier)
    
    return t,f,sxx

def next_power_of_2(x):
    return 1 if x == 0 else int(2**(np.ceil(np.log2(x))))

def spectrogram(x=None, win=None, noverlap=None, nfft=None,fs=None, sides='onesided'):
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
        
    
    freqs*=fs
    
    result = func(result, nfft)
    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                    nperseg - noverlap)/float(fs)

    return time,freqs,result