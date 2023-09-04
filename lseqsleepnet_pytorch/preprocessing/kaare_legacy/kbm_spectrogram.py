


#%%
import numpy as np


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
        
    
    freqs*=fs
    
    result = func(result, nfft)
    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                    nperseg - noverlap)/float(fs)

    return time,freqs,result




#%% compare with matlab spectrogram:

if __name__=='__main__':
    import scipy.signal
    import matplotlib.pyplot as plt
    
    
    def next_power_of_2(x):
        return 1 if x == 0 else int(2**(np.ceil(np.log2(x))))

    win_size=2
    signal=np.random.randn(200)
    fs_fourier=100
    overlap=1

    nfft = next_power_of_2(win_size*fs_fourier)
    X_eeg = []

    window=scipy.signal.windows.hamming(int(win_size *fs_fourier))
    window=window*0+1

    f,t,sxx = scipy.signal.spectrogram(signal,
        fs = fs_fourier,  window=window,
        noverlap = overlap*fs_fourier, nfft = 256,detrend=False)

    print('Check Parseval\'s theorem:')
    print('x^2  :', np.sum(np.abs(signal)**2))
    print('scipy:',np.sum(np.abs(sxx)**2))
    print('fft^2:',np.sum(np.abs(np.fft.fft(signal))**2)/(len(signal)))
    # print(np.sum(np.abs(np.fft.rfft(signal))**2)*2/(len(signal)))


    t,f,mysxx=spectrogram_matlabCopy(signal, win=window,noverlap=overlap*fs_fourier, nfft=nfft,fs=fs_fourier,sides='twosided')
    mysxx=np.abs(mysxx)**2
    print('mine:',(np.sum(mysxx))/nfft)



    import matlab.engine
    eng=matlab.engine.start_matlab()

    signal=np.random.randn(2000)



    eng.workspace['signal']=signal
    eng.workspace['fs_fourier']=fs_fourier
    eng.workspace['overlap']=1 
    eng.workspace['win_size']=2
    eng.workspace['nfft']=256
    eng.workspace['window']=window

    eng.eval('[sxx,f,t]=spectrogram(signal,window,overlap*fs_fourier,nfft,fs_fourier);',nargout=0)
    sxx=eng.workspace['sxx']
    sxxmatlab=np.array(sxx).T


    t,f,mysxx=spectrogram_matlabCopy(signal, win=200, noverlap=100, nfft=nfft,fs=fs_fourier)



    plt.figure()
    plt.subplot(3,1,1)
    plt.title('Python')
    plt.imshow(np.abs(mysxx)**2)
    plt.subplot(3,1,2)
    plt.title('Matlab')
    plt.imshow(np.abs(sxxmatlab)**2)

    plt.subplot(3,1,3)
    plt.plot(np.abs(mysxx)**2,np.abs(sxxmatlab)**2,'r*');
    plt.plot([0,2000],[0,2000],'k--',label='y=x')
    plt.legend()
    plt.xlabel('python')
    plt.ylabel('matlab')