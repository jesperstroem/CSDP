# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:26:04 2023

@author: Jesper Strøm
"""

import functools
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics
import sys

import pickle
import torch
import io
import numpy as np
sys.path.append(os.path.abspath('..'))
from shared.utility import acc, kappa, f1, create_confusionmatrix, plot_confusionmatrix

hold_out_sets = ["isruc_sg1", "isruc_sg2", "isruc_sg3", "mass_c1", "mass_c3", "svuh", "dod-h", "dod-o"]
non_hold_out_sets = ["abc", "ccshs", "cfs", "chat", "dcsm", "homepap", "mesa", "mros", "phys", "sedf_sc", "sedf_st", "shhs", "sof"]
all_sets = ["abc", "ccshs", "cfs", "chat", "dcsm", "homepap", "mesa", "mros", "phys", "sedf_sc", "sedf_st", "shhs", "sof", "isruc_sg1", "isruc_sg2", "isruc_sg3", "mass_c1", "mass_c3", "svuh", "dod-h", "dod-o"]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def plot_trainval_loss_graph(train_path, val_path, output_path):
    '''
    Plots training and validation loss for a training session
    
    train_path (str): Path to the training loss csv file
    val_path (str): Path to the validation loss csv file
    '''
    
    train_losses = pd.read_csv(train_path)
    train_losses.columns =['Epoch', '-', 'Training']
    train_losses = train_losses.drop('-', axis=1)

    val_losses = pd.read_csv(val_path)
    val_losses.columns =['-', '-', 'Validation']
    val_losses = val_losses.drop('-', axis=1)
    
    df = pd.concat([train_losses, val_losses], axis=1)
    
    sns.set(rc={'figure.figsize':(10,6)})
    ax = sns.lineplot(
        data=df[["Training", "Validation"]],
        palette=["royalblue", "firebrick"]
    )

    plt.legend(fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    fig = ax.get_figure()
    fig.savefig(output_path, bbox_inches="tight")

def plot_acc_kapp_graph(kappa_path, acc_path, ylim, output_path):
    '''
    Plots the accuracy and kappa for a training session
    
    kappa_path (str): Path to the kappa csv file
    acc_path (str): Path to the accuracy csv file
    
    '''
    val_kap = pd.read_csv(kappa_path)
    val_kap.columns =['Epoch', '-', 'Kappa']
    val_kap = val_kap.drop('-', axis=1)

    val_acc = pd.read_csv(acc_path)
    val_acc.columns =['-', '-', 'Accuracy']
    val_acc = val_acc.drop('-', axis=1)
    
    df = pd.concat([val_kap, val_acc], axis=1)
    
    sns.set(rc={'figure.figsize':(10,6)})
    ax = sns.lineplot(
        data=df[["Kappa", "Accuracy"]],
        palette=["royalblue", "firebrick"]
    )

    plt.legend(fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim(ylim[0], ylim[1])
    
    fig = ax.get_figure()
    fig.savefig(output_path, bbox_inches="tight")

def get_f1_dataframe(base,classnumber,classtype):
    f1 = pd.read_csv(base+f"training_f1_c{classnumber}.csv")
    f1.columns =['Epoch', '-', classtype]
    f1 = f1.drop('-', axis=1)
    return f1

def plot_f1_graph(base, output_path):
    '''
    Plots the development of f1 validation scores during training
    
    base (str): path to the directory expecting files on this form: training_f1_c<classnumber>.csv
    '''
    
    c1 = get_f1_dataframe(base,1, "Wake")
    c2 = get_f1_dataframe(base,2, "N1")
    c3 = get_f1_dataframe(base,3, "N2")
    c4 = get_f1_dataframe(base,4, "N3")
    c5 = get_f1_dataframe(base,5, "REM")
    
    df = pd.concat([c1, c2,c3,c4,c5], axis=1)
    
    sns.set(rc={'figure.figsize':(10,6)})
    
    ax = sns.lineplot(
        data=df[["Wake", "N1","N2","N3","REM"]],
        palette=["royalblue", "firebrick", "red", "blue", "green"]
    )

    plt.legend(fontsize=12)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim(0, 1)
    
    fig = ax.get_figure()
    fig.savefig(output_path, bbox_inches="tight")

def plot_trainval_speed(output_path):
    '''
    Function that plots training speeds. Hardcoded data, nothing should change
    '''
    
    # Hardcoded data from table in thesis
    data = pd.DataFrame({'Time': [437.47, 234.37,140.52,86.77],
                         'Time x GPUs': [(437.47*1), (234.37*2),(140.52*4),(86.77*8)]},
                        index=['1', '2', '4', '8'])
    
    # Create the pandas DataFrame
    ax = data.plot(kind='line', stacked=False, color=['red', 'pink'])
    
    plt.xticks(rotation = 0)
    plt.xlabel('GPUs')
    plt.ylabel('Time (s)')
    fig = ax.get_figure()
    fig.savefig(output_path, bbox_inches="tight")
    
def plot_learningrate_graph(kappa_csv_path, lr_csv_path, output_path):
    '''
    Plots the learning rate correlation to the validation kappa
    
    kappa_csv_path (str): Path to the kappa csv file
    lr_csv_path (str): Path to the learning rate csv file
    '''
    
    val_kap = pd.read_csv(kappa_csv_path)
    val_kap.columns =['Epoch', '-', 'Kappa']
    val_kap = val_kap.drop('-', axis=1)
    
    lr = pd.read_csv(lr_csv_path)
    lr.columns =['Epoch', '-', 'LR']
    lr = lr.drop('-', axis=1)
    
    ax1 = sns.lineplot(data=val_kap, x="Epoch", y="Kappa", color="royalblue")
    ax1.set_ylabel('Kappa', color='royalblue')
    ax1.tick_params(axis='y', colors='royalblue')
    ax2 = plt.twinx()

    a = sns.lineplot(data=lr, x="Epoch", y="LR", color="firebrick", ax=ax2)
    ax2.set_ylabel('LR', color='firebrick')
    ax2.tick_params(axis='y', colors='firebrick')
    
    fig = a.get_figure()
    fig.savefig(output_path, bbox_inches="tight")
    
def compare(item1, item2):
    d1 = item1.split(".")[0]
    d2 = item2.split(".")[0]
    
    try:
        i1 = all_sets.index(d1)
        i2 = all_sets.index(d2)
    except:
        return 0
    
    if i1 < i2:
        return -1
    elif i1 > i2:
        return 1
    else:
        return 0
    
def plot_kappa_boxplot(result_path_1, result_path_2, model_1_name, model_2_name, datasets, output_path):
    """
    Function for comparing two models kappa values for each dataset on a boxplot.
    
    ============= ARGS =============
    result_path_1: Path to test_metrics folder of model 1 results.
    result_path_2: Path to test_metrics folder of model 2 results.
    model_1_name: Name of model 1, string.
    model_2_name: Name of model 2, string.
    datasets: List of strings of datasets that is to be plotted. E.g. ["dod-h", "dod-o", "svuh"]
    output_path: Path where plot is saved.
    ================================
    """
    
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    
    resultfiles_1 = os.listdir(result_path_1)
    resultfiles_2 = os.listdir(result_path_2)
    
    resultfiles_1 = sorted(resultfiles_1, key=functools.cmp_to_key(compare))
    resultfiles_2 = sorted(resultfiles_2, key=functools.cmp_to_key(compare))
    
    # Creating dataframe for first model results
    for filename in resultfiles_1:
        f = os.path.join(result_path_1, filename)
        
        if os.path.isfile(f) and filename in datasets:
            file = open(f,'rb')
            
            object_file = pickle.load(file)
            
            kappa_scores = object_file["kappa_scores"]
            kappa_scores = kappa_scores.tolist()
            
            if "sedf_sc_physionet" == filename:
                filename = "sedf_sc"
            if "sedf_st_physionet" == filename:
                filename = "sedf_st"
            
            additional_df = pd.DataFrame(kappa_scores, columns=[filename])

            df = pd.concat([df, additional_df], axis=1)
    
    # Creating dataframe for second model results
    for filename in resultfiles_2:
        f = os.path.join(result_path_2, filename)
        
        if os.path.isfile(f) and filename in datasets:
            file = open(f,'rb')
            
            object_file = pickle.load(file)
            
            kappa_scores = object_file["kappa_scores"]
            kappa_scores = kappa_scores.tolist()
            
            if "sedf_sc_physionet" == filename:
                filename = "sedf_sc"
            if "sedf_st_physionet" == filename:
                filename = "sedf_st"
            
            additional_df = pd.DataFrame(kappa_scores, columns=[filename])

            df2 = pd.concat([df2, additional_df], axis=1)
    
    # Merging the two dataframes
    df = pd.melt(df)
    df["model"] = model_1_name
    
    df2 = pd.melt(df2)
    df2["model"] = model_2_name
    
    final_df = pd.concat([df, df2])
    final_df.columns = ["datasets", "kappa", "model"]
    
    #print(final_df.to_string())
    #first_half = final_df.iloc[]
        #final_df = final_df.append({'datasets': "", kappa: None}, ignore_index=True)
    final_df.loc[1910.5] = {'datasets': "", kappa: None}
    final_df = final_df.sort_index()
    #print(final_df.to_string())
   
    # Plotting dataframe
    plt.figure(figsize=(9,6))
    boxplot = sns.boxplot(x="datasets", hue="model", y="kappa", data=final_df)
    plt.legend()
    fig = boxplot.get_figure()
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylim(0, 1)
    
    fig.savefig(output_path)
    
def plot_mean_f1_differences(models,
                             output_path,
                             lim=(0.0,1.0),
                             holdout_only = False): 
    
    '''
    Plots the mean F1 scores for each sleep stage across all available datasets.
    
    models (array): Tuple - 1: path to result files
                            2: name of model
                            3: color on the barplot
    holdout_only (bool): If only the results for the hold-out sets should be plotted.
    '''
    
    model_files = []
    
    for m in models:
        if(holdout_only):
            files = [m[0]+f for f in os.listdir(m[0]) if os.path.isfile(m[0]+f) and f in hold_out_sets]
        else:
            files = [m[0]+f for f in os.listdir(m[0]) if os.path.isfile(m[0]+f)]
        
        model_files.append(files)
    
    results = []
    
    for i, model in enumerate(model_files):
        results.append([[],[],[],[],[]])
        for file in model:
            with open(file, 'rb') as f:
                data = CPU_Unpickler(f).load()
                results[i][0].append(float(data['f1_c1']))
                results[i][1].append(float(data['f1_c2']))
                results[i][2].append(float(data['f1_c3']))
                results[i][3].append(float(data['f1_c4']))
                results[i][4].append(float(data['f1_c5']))
        
        results[i] = np.mean(results[i], axis=1)
            
    final_data = {}
    
    for i, r in enumerate(results):
        final_data[models[i][1]] = r
        
    data = pd.DataFrame(final_data,
                        index=['Wake', 'N1', 'N2', 'N3', 'REM'])
    
    print(final_data)
    # Create the pandas DataFrame
    colors = [c[2] for c in models]
    ax = data.plot(kind='bar', stacked=False, color=colors)
    
    plt.xticks(rotation = 0)
    plt.xlabel('Sleep stages')
    plt.ylabel('Mean F1')

    plt.ylim(lim[0], lim[1])
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig = ax.get_figure()
    fig.savefig(output_path, bbox_inches="tight")

def plot_cf_per_dataset(results_path):
    '''
    Plots confusion matrices for each dataset
    
    results_path (str): Path to result files
    '''
    
    files = [results_path+f for f in os.listdir(results_path) if os.path.isfile(results_path+f)]
    
    for file in files:
        with open(file, 'rb') as f:
            data = CPU_Unpickler(f).load()
            labels = data['labels']
            preds = data['preds']
            
            cf_matrix = create_confusionmatrix(preds, labels)
            plot_confusionmatrix(cf_matrix, data['dataset_name'])
            
def plot_overall_cf(results_path, save_figure=False, save_path=None):
    '''
    Plots one confusion matrix for all datasets combined
    
    results_path (str): Path to result files
    '''
    
    files = [results_path+f for f in os.listdir(results_path) if os.path.isfile(results_path+f) and ".png" not in f]
    
    labels = []
    preds = []
    
    for file in files:
        with open(file, 'rb') as f:
            data = CPU_Unpickler(f).load()
            labels.append(data['labels'])
            preds.append(data['preds'])
    
    labels = torch.cat(labels)
    preds = torch.cat(preds)

    cf_matrix = create_confusionmatrix(preds, labels)
    plot_confusionmatrix(cf_matrix, "", save_figure=save_figure, save_path=save_path)

def plot_kappaf1_diffs_dataset_type(first_base, first_name, second_base, second_name, outpath):
    
    penguins = sns.load_dataset("penguins")
    print(penguins)
    
    hold_out_kappas_first = []
    hold_out_f1s_first = []
    hold_out_kappas_second = []
    hold_out_f1s_second = []
    non_hold_out_kappas_first = []
    non_hold_out_f1s_first = []
    non_hold_out_kappas_second = []
    non_hold_out_f1s_second = []
    
    for d in hold_out_sets:
        with open(f"{first_base}{d}", "rb") as f:
            data = CPU_Unpickler(f).load()
            kappa_first = data['kappa']
            f1_mean_first = data['f1_mean']
            hold_out_kappas_first.append(kappa_first)
            hold_out_f1s_first.append(f1_mean_first)
        
        with open(f"{second_base}{d}", "rb") as f:
            data = CPU_Unpickler(f).load()
            kappa_second = data['kappa']
            f1_mean_second = data['f1_mean']
            hold_out_kappas_second.append(kappa_second)
            hold_out_f1s_second.append(f1_mean_second)
    
    for d in non_hold_out_sets:
        with open(f"{first_base}{d}", "rb") as f:
            data = CPU_Unpickler(f).load()
            kappa_first = data['kappa']
            f1_mean_first = data['f1_mean']
            non_hold_out_kappas_first.append(kappa_first)
            non_hold_out_f1s_first.append(f1_mean_first)
        
        with open(f"{second_base}{d}", "rb") as f:
            data = CPU_Unpickler(f).load()
            kappa_second = data['kappa']
            f1_mean_second = data['f1_mean']
            non_hold_out_kappas_second.append(kappa_second)
            non_hold_out_f1s_second.append(f1_mean_second)
    
    non_hold_out_kappa_first_mean = sum(non_hold_out_kappas_first)/len(non_hold_out_kappas_first)
    non_hold_out_kappa_second_mean = sum(non_hold_out_kappas_second)/len(non_hold_out_kappas_second)
    hold_out_kappa_first_mean = sum(hold_out_kappas_first)/len(hold_out_kappas_first)
    hold_out_kappa_second_mean = sum(hold_out_kappas_second)/len(hold_out_kappas_second)
    
    non_hold_out_f1_first_mean = sum(non_hold_out_f1s_first)/len(non_hold_out_f1s_first)
    non_hold_out_f1_second_mean = sum(non_hold_out_f1s_second)/len(non_hold_out_f1s_second)
    hold_out_f1_first_mean = sum(hold_out_f1s_first)/len(hold_out_f1s_first)
    hold_out_f1_second_mean = sum(hold_out_f1s_second)/len(hold_out_f1s_second)
    
    
    final_data = {}
    final_data[first_name] = [float(non_hold_out_kappa_first_mean),
                              float(hold_out_kappa_first_mean),
                              float(non_hold_out_f1_first_mean),
                              float(hold_out_f1_first_mean)]
    final_data[second_name] = [float(non_hold_out_kappa_second_mean),
                               float(hold_out_kappa_second_mean),
                               float(non_hold_out_f1_second_mean),
                               float(hold_out_f1_second_mean)]
 
    data = pd.DataFrame(final_data,
                        index=['Non-holdout mean Kappa', 'Holdout mean Kappa', 'Non-holdout mean F1', 'Holdout mean F1'])
    
    print(data)
    
    # Create the pandas DataFrame
    #colors = [c[2] for c in models]
    ax = data.plot(kind='bar', stacked=False, color=["royalblue", "firebrick"])
    
    plt.xticks(rotation = 45)
    #plt.xlabel('Sleep stages')
    plt.ylabel('Value')

    plt.ylim(0.6, 0.85)
    
    #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig = ax.get_figure()
    fig.savefig(outpath, bbox_inches="tight")
    
    
def plot_kappameans_diffs(first_model_name, first_bases, second_model_name, second_bases, colors, lim, output_path, datasets):
    final_data = {}
    
    final_data[first_model_name] = []
    final_data[second_model_name] = []

    for d in datasets:    
        with open(first_bases[0]+d, 'rb') as f:
            data = CPU_Unpickler(f).load()
            first_kappa1 = float(data['kappa'])
                
        with open(first_bases[1]+d, 'rb') as f:
            data = CPU_Unpickler(f).load()
            first_kappa2 = float(data['kappa'])
            
        diff = first_kappa1-first_kappa2
        final_data[first_model_name].append(diff)
        
        with open(second_bases[0]+d, 'rb') as f:
            data = CPU_Unpickler(f).load()
            first_kappa1 = float(data['kappa'])
                
        with open(second_bases[1]+d, 'rb') as f:
            data = CPU_Unpickler(f).load()
            first_kappa2 = float(data['kappa'])    
            
        diff = first_kappa1-first_kappa2
        final_data[second_model_name].append(diff)
        
    data = pd.DataFrame(final_data,
                        index=datasets)
    
    # Create the pandas DataFrame
    ax = data.plot(kind='bar', stacked=False, color=colors)
    
    plt.xticks(rotation = 90)
    plt.xlabel('Datasets')
    plt.ylabel('Mean Kappa differences')
    #plt.ylim(lim[0],lim[1])
    
    #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig = ax.get_figure()
    fig.savefig(output_path, bbox_inches="tight")
    
    #plt.ylim(0, 1)
    

def plot_test_metrics(test_metrics_paths=["/home/alec/repos/test_metrics_BIG-122/", "/home/alec/repos/test_metrics_BIG-78/"], run_names=["Article split", "Random split"]):
    """
    Comparing kappa values and f1 mean values per dataset for listed runs and saving to plots.
    """
    dataset_files = ["abc", "ccshs", "cfs", "chat", "dcsm", "homepap", "mesa", "mros", "phys", "sedf_sc", "sedf_st", "shhs", "sof", "isruc_sg1", "isruc_sg2", "isruc_sg3", "mass_c1", "mass_c3", "svuh", "dod-h", "dod-o"]
    
    kappa_dict = {"dataset": dataset_files}
    f1_mean_dict = {"dataset": dataset_files}
    
    for test_metric_path, run_name in zip(test_metrics_paths, run_names):
        dataset_list = []
        kappa_list = []
        f1_mean_list = []
        
        for f in dataset_files:                
            with open(test_metric_path + f, 'rb') as fp:
                obj = pickle.load(fp)
                dataset_list.append(f)
                kappa_list.append(float(obj.get("kappa")))
                f1_mean_list.append(float(obj.get("f1_mean")))
        
        assert len(dataset_list) == len(kappa_list) == len(kappa_list)
        
        kappa_dict[run_name] = kappa_list
        
        f1_mean_dict[run_name] = f1_mean_list
        
    kappa_df = pd.DataFrame(kappa_dict)
    print(kappa_df)
    kappa_df = pd.melt(kappa_df, id_vars=['dataset'], value_vars=run_names, var_name='run ID', value_name='kappa')
    
    print(kappa_df)
    exit()
    
    f1_mean_df = pd.DataFrame(f1_mean_dict)
    f1_mean_df = pd.melt(f1_mean_df, id_vars=['dataset'], value_vars=run_names, var_name='run ID', value_name='F1 mean')
        
    for idx, (outfile, df, y_label) in enumerate(zip(
        ["./split_comparison_kappa.png", "./split_comparison_f1_mean.png"], 
        [kappa_df, f1_mean_df],
        ["kappa", "F1 mean"]
    )):
    
        plt.figure(idx)
        ax = sns.barplot(data=df, x="dataset", y=y_label, hue="run ID")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.ylim(0.5, 0.9)
        plt.xticks(rotation = 90)
        plt.grid()
        plt.savefig(outfile, bbox_inches="tight")
    
    
def plot_f1_mean_primary_results():
    f1_mean = {
        'Dataset': ["ABC", "CCSHS", "CFS", "CHAT", "DCSM", "HPAP", "MESA", "MROS", "PHYS", "SEDF-SC", "SEDF-ST", "SHHS", "SOF", "ISRUC-SG1", "ISRUC-SG2", "ISRUC-SG3", "MASS-C1", "MASS-C3", "SVUH", "DOD-H", "DOD-O"],
        'L-SeqSleepNet': [0.69, 0.82, 0.77, 0.79, 0.77, 0.72, 0.69, 0.71, 0.70, 0.73, 0.78, 0.75, 0.71, 0.68, 0.61, 0.68, 0.65, 0.76, 0.66, 0.77, 0.73],
        'U-Sleep': [0.74, 0.84, 0.79, 0.82, 0.81, 0.74, 0.73, 0.73, 0.75, 0.76, 0.79, 0.77, 0.74, 0.71, 0.66, 0.70, 0.68, 0.76, 0.71, 0.80, 0.74],
        'U-Sleep (orig.)': [0.77, 0.85, 0.82, 0.85, 0.81, 0.78, 0.79, 0.77, 0.79, 0.79, 0.76, 0.80, 0.78, 0.77, 0.76, 0.77, 0.73, 0.80, 0.73, 0.82, 0.79]
    }
    df = pd.DataFrame(f1_mean)
    
    df = pd.melt(df, id_vars=['Dataset'], value_vars=["L-SeqSleepNet", "U-Sleep", "U-Sleep (orig.)"], var_name='Model', value_name='F1 Mean')
    
    sns.set(rc={'figure.figsize':(10,6)})
    
    plt.ylim(0.5, 0.9)
    plt.grid(zorder=0)
    plt.xticks(rotation = 90)
    
    ax = sns.barplot(data=df, x="Dataset", y="F1 Mean", hue="Model", zorder=3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    plt.savefig("./f1_mean_sota_comparison.png", bbox_inches="tight")
    
    
def plot_avg_f1_class_score():
    lseq_wake = statistics.fmean([0.87, 0.96, 0.96, 0.93, 0.96, 0.89, 0.94, 0.95, 0.74, 0.95, 0.83, 0.94, 0.95, 0.83, 0.71, 0.82, 0.83, 0.87, 0.78, 0.85, 0.82])
    usleep_wake = statistics.fmean([0.89, 0.96, 0.96, 0.95, 0.99, 0.90, 0.95, 0.96, 0.80, 0.98, 0.82, 0.94, 0.95, 0.77, 0.65, 0.69, 0.93, 0.90, 0.80, 0.87, 0.81])
    usleep_orig_wake = statistics.fmean([0.87, 0.93, 0.93, 0.93, 0.97, 0.91, 0.92, 0.93, 0.84, 0.93, 0.80, 0.93, 0.93, 0.89, 0.85, 0.90, 0.94, 0.93, 0.80, 0.91, 0.90])
    
    lseq_n1 = statistics.fmean([0.36, 0.49, 0.42, 0.45, 0.45, 0.40, 0.36, 0.34, 0.46, 0.45, 0.64, 0.39, 0.22, 0.32, 0.24, 0.36, 0.35, 0.53, 0.37, 0.47, 0.45])
    usleep_n1 = statistics.fmean([0.50, 0.57, 0.48, 0.56, 0.54, 0.45, 0.49, 0.40, 0.56, 0.51, 0.70, 0.51, 0.36, 0.40, 0.36, 0.44, 0.33, 0.49, 0.38, 0.60, 0.48])
    usleep_orig_n1 = statistics.fmean([0.53, 0.63, 0.52, 0.64, 0.48, 0.48, 0.59, 0.46, 0.60, 0.57, 0.58, 0.51, 0.45, 0.52, 0.49, 0.55, 0.41, 0.54, 0.37, 0.60, 0.52])
    
    lseq_n2 = statistics.fmean([0.80, 0.88, 0.87, 0.83, 0.85, 0.80, 0.83, 0.88, 0.83, 0.82, 0.90, 0.88, 0.84, 0.75, 0.69, 0.74, 0.78, 0.84, 0.77, 0.86, 0.87])
    usleep_n2 = statistics.fmean([0.83, 0.89, 0.88, 0.84, 0.88, 0.81, 0.84, 0.88, 0.85, 0.86, 0.89, 0.88, 0.85, 0.76, 0.72, 0.74, 0.79, 0.85, 0.80, 0.86, 0.85])
    usleep_orig_n2 = statistics.fmean([0.84, 0.91, 0.89, 0.87, 0.86, 0.84, 0.87, 0.87, 0.84, 0.86, 0.88, 0.87, 0.86, 0.79, 0.78, 0.78, 0.81, 0.87, 0.81, 0.87, 0.86])
    
    lseq_n3 = statistics.fmean([0.54, 0.86, 0.78, 0.88, 0.71, 0.63, 0.45, 0.50, 0.66, 0.57, 0.65, 0.64, 0.70, 0.70, 0.67, 0.70, 0.45, 0.65, 0.60, 0.76, 0.67])
    usleep_n3 = statistics.fmean([0.57, 0.87, 0.77, 0.88, 0.73, 0.65, 0.49, 0.54, 0.69, 0.59, 0.66, 0.63, 0.69, 0.79, 0.79, 0.78, 0.45, 0.65, 0.75, 0.73, 0.68])
    usleep_orig_n3 = statistics.fmean([0.72, 0.88, 0.84, 0.90, 0.83, 0.78, 0.65, 0.68, 0.81, 0.71, 0.64, 0.76, 0.77, 0.77, 0.83, 0.74, 0.61, 0.75, 0.78, 0.79, 0.74])
    
    lseq_rem = statistics.fmean([0.91, 0.91, 0.85, 0.86, 0.88, 0.87, 0.85, 0.88, 0.82, 0.84, 0.91, 0.88, 0.84, 0.80, 0.73, 0.79, 0.84, 0.90, 0.79, 0.91, 0.83])
    usleep_rem = statistics.fmean([0.92, 0.92, 0.86, 0.88, 0.90, 0.88, 0.88, 0.89, 0.85, 0.87, 0.91, 0.89, 0.85, 0.81, 0.78, 0.83, 0.88, 0.91, 0.83, 0.92, 0.88])
    usleep_orig_rem = statistics.fmean([0.90, 0.93, 0.91, 0.90, 0.89, 0.90, 0.90, 0.88, 0.87, 0.88, 0.91, 0.92, 0.92, 0.88, 0.86, 0.85, 0.88, 0.91, 0.88, 0.94, 0.92])
    
    f1_class_wise = {
        'Class': ["Wake", "N1", "N2", "N3", "REM"],
        'L-SeqSleepNet': [lseq_wake, lseq_n1, lseq_n2, lseq_n3, lseq_rem],
        'U-Sleep': [usleep_wake, usleep_n1, usleep_n2, usleep_n3, usleep_rem],
        'U-Sleep (orig.)': [usleep_orig_wake, usleep_orig_n1, usleep_orig_n2, usleep_orig_n3, usleep_orig_rem]
    }
    
    df = pd.DataFrame(f1_class_wise)
    
    df = pd.melt(df, id_vars=['Class'], value_vars=["L-SeqSleepNet", "U-Sleep", "U-Sleep (orig.)"], var_name='Model', value_name='F1 score')
    
    sns.set(rc={'figure.figsize':(10,6)})
    
    plt.ylim(0.3, 1)
    plt.grid(zorder=0)
    #plt.xticks(rotation = 90)
    
    ax = sns.barplot(data=df, x="Class", y="F1 score", hue="Model", zorder=3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    plt.savefig("./f1_class_wise_sota_comparison.png", bbox_inches="tight")
    

def main():
    print("Hello world from plots.py")
    
    
if __name__ == '__main__':
    main()
    
