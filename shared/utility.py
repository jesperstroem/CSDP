import torch.nn as nn
import torch

from torchmetrics.classification import MulticlassCohenKappa, MulticlassF1Score
from torchmetrics.functional import accuracy # pytorch_lightning.metrics.Accuracy does not work anymore

from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import h5py
import pickle
from neptune.utils import stringify_unsupported
import os


def dump(t):
    print(t)
    print(t.shape)


def log_step(logger, logpath, global_rank, stage, tags, preds, labels, **kwargs):
    """
    Used for logging raw predictions and true labels for a single step. Extra logging to Neptune happens through kwargs.

    Logging to file at location specified by logpath

    Naming convention of file: {model_name}_{run_id}_{global_rank}
    """
    tag = tags[0].split('<HDF5')[0]
    dataset, record = tag.split(".hdf5")
    dataset = dataset.split("/")[-1]
    tag = dataset + record

    step_data = { # TODO: Maybe log other things? global_step?
        "stage": stage,
        "record": tag,
        "preds": preds,
        "labels": labels
    }

    filename = f"{logpath}{logger.name}_{logger.version}_{global_rank}"

    with open(filename, "ab") as f:
        pickle.dump(step_data, f)

    for key, value in kwargs.items():
        logger.experiment[f"{stage}/{tag}/{key}"].log(stringify_unsupported(value))


def log_test_step(base, run_id, dataset, subject, record, **kwargs):
        """
        Used for logging raw predictions and true labels for a single step. Extra logging to Neptune happens through kwargs.
        Logging to file at location: ???
        Naming convention of file: {model_name}_{run_id} ???
        """

        print(f"logging for: {dataset}/{subject}/{record}")
        
        print(f"kwargs: {kwargs}")
        

        #step_data = {
        #    "ensemble_preds": ensemble_preds,
        #    "single_preds": single_preds,
        #    "labels": labels
        #}

        path = f"{base}/{run_id}"

        if not os.path.exists(path):
            os.makedirs(path)

        filename = f"{path}/{dataset}.{subject}_{record}"

        print(f"log preds and labels to file: {filename}")

        with open(filename, "ab") as f:
            pickle.dump(kwargs, f)


def log_dataset_splits(datasets, logger):
    # Takes a list of paths to hdf5 datasets, together with a neptune logger
    for d in datasets:
        with h5py.File(d, 'r') as f:
            trainrecords = [k for k in f.keys() if k.startswith('train_')]
            valrecords = [k for k in f.keys() if k.startswith('val_')]
            testrecords = [k for k in f.keys() if k.startswith('test_')]
            if not trainrecords:
                print(f"No train records for dataset {d}")
            else:
                logger.experiment[f"datasets/{d}/train"].log(trainrecords)
            if not valrecords:
                print(f"No validation records for dataset {d}")
            else:
                logger.experiment[f"datasets/{d}/val"].log(valrecords)
            if not testrecords:
                print(f"No test records for dataset {d}")
            else:
                logger.experiment[f"datasets/{d}/test"].log(testrecords)
    
def filter_unknowns(predictions, labels):
    mask = labels != 5
    labels = torch.masked_select(labels, mask)
    predictions = torch.masked_select(predictions, mask)
    
    assert len(labels) == len(predictions)
    
    return predictions, labels
    
def kappa(predictions, labels, num_classes=5):
    predictions, labels = filter_unknowns(predictions, labels)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    labels = labels.to(device)
    
    predictions = predictions.to(device)
    
    metric = MulticlassCohenKappa(num_classes=num_classes).to(device)
    
    kappa = metric(predictions, labels)
    return kappa

def acc(predictions, labels):
    predictions, labels = filter_unknowns(predictions, labels)
    
    accu = accuracy(task="multiclass", num_classes=5, preds=predictions, target=labels)
    
    return accu

def f1(predictions, labels, average=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, labels = filter_unknowns(predictions, labels)
    
    predictions = predictions.to(device)
    labels = labels.to(device)
    
    if average:
        metric = MulticlassF1Score(num_classes=5).to(device)
    else:
        metric = MulticlassF1Score(num_classes=5, average=None).to(device)
        
    score = metric(predictions, labels)
    
    return score

def create_confusionmatrix(pred, labels):
    pred, labels = filter_unknowns(pred, labels)
    
    return confusion_matrix(labels,pred)

def plot_confusionmatrix(conf, title, percentages = True, formatting = '.2f', save_figure=False, save_path=None):
    if percentages:
        conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
        
    df_cm = pd.DataFrame(conf,
                         index = [i for i in ["Wake", "N1", "N2", "N3", "REM"]],
                         columns = [i for i in ["Wake", "N1", "N2", "N3", "REM"]])
    plt.figure(figsize=(10,7))
    plt.title(title)
    f = sn.heatmap(df_cm, annot=True, fmt=formatting)
    f.set(xlabel='Predicted', ylabel='Truth')
    if(save_figure):
        plt.savefig(save_path)
    
def majority_vote(preds):
    # Expects preds to be (Num_votes, num_epochs)
    #assert preds.dim() == 2
    
    preds = preds.swapaxes(0,1)
    
    mode, _ = torch.mode(preds,1)
   
    return mode


def ensemble_vote(votes):
    """
    Function to choose highest voted label for each epoch
    
    Input should be a list of lists
    
    # TODO: In case of tie, which element should we return?
    """
    preds = []
    
    for epoch in votes:
        data = Counter(epoch)
        chosen_pred = max(epoch, key=data.get)
        preds.append(chosen_pred)
    return preds
    

if __name__ == '__main__':
    ensemple_voting("lol")
    #preds = torch.tensor([1,1,2,0,1,1,1,0,1,2,0])
    #
    #labels = torch.tensor([1,1,2,0,1,1,1,0,1,2,0])
    #
    #score = f1(preds,
    #           labels,
    #           average=False)
    #print(score)
