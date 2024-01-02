import torch

from torchmetrics.classification import MulticlassCohenKappa, MulticlassF1Score
from torchmetrics.functional import accuracy # pytorch_lightning.metrics.Accuracy does not work anymore

from sklearn.model_selection import train_test_split
import h5py
import pickle
import os
import json

def dump(t):
    print(t)
    print(t.shape)

def log_test_step(base, run_id, dataset, subject, record, **kwargs):
        """
        Used for logging raw predictions and true labels for a single step. Extra logging to Neptune happens through kwargs.
        Logging to file at location: ???
        Naming convention of file: {model_name}_{run_id} ???
        """

        print(f"logging for: {dataset}/{subject}/{record}")
        
        print(f"kwargs: {kwargs}")

        path = f"{base}/{run_id}/{dataset}"

        if not os.path.exists(path):
            os.makedirs(path)

        filename = f"{path}/{dataset}.{subject}_{record}"

        print(f"log preds and labels to file: {filename}")

        with open(filename, "ab") as f:
            pickle.dump(kwargs, f)

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

def create_split_file(hdf5_basepath):
    output_name = "random_split.json"

    hdf5_paths = os.listdir(hdf5_basepath)

    output_dic = dict()

    for path in hdf5_paths:
        with h5py.File(f"{hdf5_basepath}/{path}", "r") as hdf5:
            subs = list(hdf5.keys())
            dataset_name = path.replace(".hdf5", "")

            train, test = train_test_split(subs, train_size=0.80, test_size=0.20)
            val, test = train_test_split(test, train_size=0.5, test_size=0.5)

            output_dic[dataset_name] = dict()

            output_dic[dataset_name]["train"] = train
            output_dic[dataset_name]["val"] = val
            output_dic[dataset_name]["test"] = test

    json_object = json.dumps(output_dic, indent=4)
    print(json_object)

    with open("random_split.json", 'w') as fp:
        fp.write(json_object)
    
    return output_name


if __name__ == '__main__':
    create_split_file("C:/Users/au588953/test_hdf5")
    