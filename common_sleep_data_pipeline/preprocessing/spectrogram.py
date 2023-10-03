# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:27:04 2023

@author: Jesper Strøm
"""

import numpy as np
import scipy.signal
#import matlab.engine

def create_spectrogram_images(x, sample_rate, win_size = 2, fs_fourier = 100, overlap = 1):
    #if original_sample_rate != target_sample_rate:
    #    x = resample_poly(x, target_sample_rate, original_sample_rate, axis=0)

    epoch_length = sample_rate*30    
    nEpochs = int(np.floor(x.shape[0]/epoch_length))
    x=x[0:nEpochs*epoch_length]
    
    x = np.reshape(x, (-1, epoch_length))
    
    spectrograms = []
    
    for i in range(nEpochs):
        t,f,sxx = __create_spectrogram(x[i], win_size, fs_fourier, overlap)

        sxxabs = np.abs(sxx)
        #sxx = sxxabs
        #print(sxx.dtype)
        #sxx = 20*np.log10(sxxabs, where=sxxabs > 0.0)
        sxx = 20*np.log10(sxxabs+0.001)
        spectrograms.append(sxx)
    
    return t, f,spectrograms

def __create_spectrogram(x, win_size, fs_fourier, overlap):
    """
    Takes a one-dimensional time-series signal and converts it to a time-frequency spectrogram image representation
    """
    nfft = __next_power_of_2(win_size*fs_fourier)

    window=scipy.signal.windows.hamming(int(win_size *fs_fourier))
    window=window*0+1
    
    t,f,sxx = __spectrogram(x,
                          window,
                          overlap*fs_fourier,
                          nfft,
                          fs_fourier)
    
    return t,f,sxx

def __next_power_of_2(x):
    return 1 if x == 0 else int(2**(np.ceil(np.log2(x))))

def __spectrogram(x=None, win=None, noverlap=None, nfft=None,fs=None, sides='onesided'):
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