import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def plotHypnoGram(labels,ax,title="Hypnogram", xlabel="Hours", title_size=16, y_size=12, x_size=12, xlabel_size=14, datetimeList=None):
    transformdic = {0: 0,1: 2,2: 3,3: 4,4: 1,5: -1}
    
    labels = [transformdic[l] for l in labels]

    labels = np.array(labels)
    
    if datetimeList is None:
        recordingStart=np.datetime64('2023-01-01')
        datetimeList=np.datetime64(recordingStart).astype('datetime64[s]') #convert to datenum with seconds precision
        datetimeList=np.array(datetimeList+np.arange(0,len(labels))*30, dtype='datetime64[s]') #convert to array

    ax.plot(datetimeList, labels, color="Black")
    
    #marking REM epochs:
    ax.plot(datetimeList[np.where(labels==1)[0]],labels[np.where(labels==1)[0]],'r.')
    ax.plot(datetimeList[np.where(labels == -1)[0]], labels[np.where(labels == -1)[0]], 'b.')

    #reverse ydir:
    ax.invert_yaxis()

    ax.set_yticks([-1,0,1,2,3,4])
    ax.set_yticklabels(['U','W','R','N1','N2', 'N3'], fontdict={"size": y_size})

    ax.set_xlabel(xlabel, fontdict={"size": xlabel_size})
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%#H'))
    ax.xaxis.set_tick_params(labelsize=x_size)
    ax.set_title(title, fontdict={"size": title_size})