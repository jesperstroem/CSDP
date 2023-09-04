import h5py
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import xml.etree.ElementTree as ET
import os
import mne


def plot_hypnogram():
    print("Hello world from plot_hypnogram()")
    
    # ----------- Hypnogram -----------
    hyp_filepath = "/home/alec/repos/data/abc/polysomnography/annotations-events-profusion/baseline/abc-baseline-900001-profusion.xml"
    
    tree = ET.parse(hyp_filepath)
        
    root = tree.getroot()
    sleep_stages = root.find("SleepStages")
    
    labels = []
    
    for stage in sleep_stages:
        if int(stage.text) == 4: # 3 and 4 are both same class (N3)
            lbl = 3
        elif int(stage.text) == 5: # Shifting REM to 4 because pyplot...
            lbl = 4
        else:
            lbl = int(stage.text)
        
        labels.append(lbl)
        
    hyp_steps = np.arange(len(labels))*30/60/60 # Time in hours                     
    
    # ----------- Raw data -----------
    psg_filepath = "/home/alec/repos/data/abc/polysomnography/edfs/baseline/abc-baseline-900001.edf"
    sample_rate = 256 # samples per second
    epoch_len = 30 # seconds
    epoch_samples = sample_rate*epoch_len
    
    data = mne.io.read_raw_edf(psg_filepath, verbose=False)
    
    # Channels measured against Fpz
    # Measured in volt
    
    eeg_channel = data.get_data("F3")[0] # F3
    eog_channel = data.get_data("E1")[0] # Left eye
    
    start_epoch = 28 # N1
    end_epoch = 29
    
    eeg_snippet = eeg_channel[start_epoch*epoch_samples:end_epoch*epoch_samples]
    eog_snippet = eog_channel[start_epoch*epoch_samples:end_epoch*epoch_samples]
    
    print(eeg_snippet.shape)
    
    steps = np.arange(len(eeg_snippet))/256 # Time in seconds
    
    # ----------- Plotting raw data -----------
    plot_path = "/home/alec/repos/Speciale2023/shared/performance_tests/"
    
    fig, axs = plt.subplots(2)
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax1.plot(steps, eeg_snippet)
    ax2.plot(steps, eog_snippet, color="red")
    
    ax1.set_ylim(-0.00025, 0.00025)
    ax2.set_ylim(-0.00025, 0.00025)
    
    ax1.set_title('EEG channel')
    ax2.set_title('EOG channel')
    fig.tight_layout(pad=1.5)
    
    ax2.set_xlabel('Seconds')
    fig.text(0.01, 0.5, 'Volt', va='center', rotation='vertical')
    
    plt.savefig(plot_path + "epoch_raw_data.png")
    
    
    # ----------- Plotting hypnogram -----------
    fig, ax1 = plt.subplots()
    
    ax1.plot(hyp_steps, labels)
    
    ax1.set_xlabel('Hours')
    fig.text(0.05, 0.5, 'Sleep Stage', va='center', rotation='vertical')
    
    plt.sca(ax1)
    plt.yticks(range(5), ['Wake', 'N1', 'N2', 'N3', 'REM'])
    
    fig.set_size_inches(8, 4)
    plt.gca().invert_yaxis()
    plt.savefig(plot_path + "record_hypnogram.png")
                
    exit()
    
    #------ OLD PLOT ------
    
    dataset_path = "/home/alec/repos/data/testing_split_delete_later/"
    dataset_name = "cfs"
    plot_path = "/home/alec/repos/Speciale2023/shared/performance_tests/"
    
    hyp_start = 1040
    hyp_len = 10
    
    signal_start = hyp_start * 128 *30
    signal_len = hyp_len * 128 * 30
    
    with h5py.File(dataset_path + dataset_name + ".hdf5", "r") as f:
        subj_key = list(f.keys())[0]
        subj = f[subj_key]
        
        rec_key = list(subj.keys())[0]
        rec = subj[rec_key]
        
        psg = rec["psg"]
        hyp = rec["hypnogram"]
            
        # Getting relevant data for all channels and corresponding hypnogram
        eeg_channels = [psg[k][signal_start:signal_start+signal_len] for k in psg.keys() if k.startswith("EEG")]
        eog_channels = [psg[k][signal_start:signal_start+signal_len] for k in psg.keys() if k.startswith("EOG")]
        
        hypnogram = hyp[hyp_start:hyp_start+hyp_len]
        hypnogram = list(np.repeat(hypnogram, 128 * 30))
    
    
    fig, ax1 = plt.subplots(layout='constrained')
    ax2 = ax1.twinx()
    
    ax1.plot(eeg_channels[0], color="g", label="EEG channel")
    ax1.plot(eog_channels[0], color="b", label="EOG channel")
    ax1.set_ylabel('mV')
    ax1.set_xlabel('Samples')
    
    ax2.plot(hypnogram, linewidth=3, color="r", label="True label")
    ax2.set_ylabel('Labels', color='r')
    ax2.set_ylim(0, 4)
    ax2.set_yticks(np.arange(0, 5, step=1), color='r')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    ax1.set_title('Hypnogram')
    
    plt.savefig(plot_path + "hypnogram_plot.png")
        
        

if __name__ == "__main__":
    plot_hypnogram()
    