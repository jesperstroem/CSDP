import h5py
import collections
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    print("Train Hello World")
    datapath = f"/users/engholma/mnt/transformed/"
    
    dataset_names, wakes, n1s, n2s, n3s, rems = get_label_counts(datapath)
    
    print(f"dataset_names: {dataset_names}")
    print(f"wakes: {wakes}")
    print(f"n1s: {n1s}")
    print(f"n2s: {n2s}")
    print(f"n3s: {n3s}")
    print(f"rems: {rems}")
    
    total__label_occurences = [sum(wakes), sum(n1s), sum(n2s), sum(n3s), sum(rems)]
    label_types = ["Wake", "N1", "N2", "N3", "REM"]
    
    plot_bar_chart("./total_label_count.png", label_types, total__label_occurences)
    
    values = {
        "Wake": wakes,
        "N1": n1s,
        "N2": n2s,
        "N3": n3s,
        "REM": rems
    }
    
    plot_grouped_bar_chart("./label_counts_per_dataset.png", dataset_names, values)


def plot_bar_chart(filename, x_labels, values):
    y_pos = np.arange(len(x_labels))
    
    fig, ax = plt.subplots(layout='constrained')
    
    ax.bar(y_pos, values, align='center', alpha=0.5, zorder=1)
    ax.set_xticks(y_pos, x_labels)
    ax.set_ylabel('Number of occurences')
    ax.set_xlabel("Class")
    
    plt.savefig(filename)
    

def plot_grouped_bar_chart(filename, dataset_names, values):
    perc_values = {
        "Wake": [],
        "N1": [],
        "N2": [],
        "N3": [],
        "REM": []
    }

    for i in range(len(dataset_names)):
        wake = values["Wake"][i]
        n1 = values["N1"][i]        
        n2 = values["N2"][i]
        n3 = values["N3"][i]
        rem = values["REM"][i]
       
        total = wake+n1+n2+n3+rem
        wake_perc = wake/total
        n1_perc = n1/total
        n2_perc = n2/total
        n3_perc = n3/total
        rem_perc = rem/total
        #print(wake_perc+n1_perc+n2_perc+n3_perc+rem_perc)
        #assert wake_perc+n1_perc+n2_perc+n3_perc+rem_perc == 1
        
        perc_values["Wake"].append(wake_perc*100)
        perc_values["N1"].append(n1_perc*100)
        perc_values["N2"].append(n2_perc*100)
        perc_values["N3"].append(n3_perc*100)
        perc_values["REM"].append(rem_perc*100)

    x = np.arange(len(dataset_names))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    #ax.grid(zorder=0)
    plt.grid(which='major', axis='y', zorder=-1.0)

    for attribute, measurement in perc_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, zorder=3)
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Datasets")
    ax.set_xticks(x + width, dataset_names, rotation=90)
    
    ax.legend(bbox_to_anchor=(1.0, 1.0))

    plt.savefig(filename, bbox_inches="tight")
                           

def get_label_counts(datapath):
    dataset_names = []
    wakes = []
    n1s = []
    n2s = []
    n3s = []
    rems = []
    
    for dataset_filename in os.listdir(datapath):
        print(dataset_filename)
        if "eesm" in dataset_filename:
            print(f"skipping {dataset_filename}")
            continue
        wake_labels = 0
        n1_labels = 0
        n2_labels = 0
        n3_labels = 0
        rem_labels = 0
    
        with h5py.File(datapath + dataset_filename, "r") as f:
            subj_ids = f.keys()

            for subj_id in subj_ids:
                subj = f[subj_id]

                record_ids = subj.keys()

                for record_id in record_ids:
                    record = subj[record_id]
                    hypnogram = list(record["hypnogram"])

                    label_counter = dict(collections.Counter(hypnogram))

                    try:
                        wake_labels = wake_labels + label_counter[0]
                    except:
                        pass

                    try:
                        n1_labels = n1_labels + label_counter[1]
                    except:
                        pass

                    try:
                        n2_labels = n2_labels + label_counter[2]
                    except:
                        pass

                    try:
                        n3_labels = n3_labels + label_counter[3]
                    except:
                        pass

                    try:
                        rem_labels = rem_labels + label_counter[4]
                    except:
                        pass

        dataset_names.append(dataset_filename.split(".")[0].upper())
        wakes.append(wake_labels)
        n1s.append(n1_labels)
        n2s.append(n2_labels)
        n3s.append(n3_labels)
        rems.append(rem_labels)
        
    #return results
    return dataset_names, wakes, n1s, n2s, n3s, rems
    

if __name__ == "__main__":
    main()
