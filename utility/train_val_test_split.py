import h5py
from sklearn.model_selection import train_test_split
import argparse
import os
import json

def split_dataset(dataset, 
                  train_ratio, 
                  val_ratio, 
                  test_ratio, 
                  shuffle=True
                 ):
    
    max_val_len = 50 # 50 according to U-Sleep
    max_test_len = 100 # 100 according to U-Sleep
    
    dataset_len = len(dataset)
    
    train_len = round(dataset_len*train_ratio)
    val_len = round(dataset_len*val_ratio)
    test_len = dataset_len - train_len - val_len # We do this to make sure round does not cause us to have values like 6/3/3 for an 11 rec dataset.
    
    if val_len > max_val_len:
        diff = val_len - max_val_len
        train_len += diff
        val_len = max_val_len
    
    if test_len > max_test_len:
        diff = test_len - max_test_len
        train_len += diff
        test_len = max_test_len

    print(f"Number of train subjects: {train_len}")
    print(f"Number of validation subjects: {val_len}")
    print(f"Number of test subjects_ {test_len}")
    
    assert train_len + val_len + test_len == len(dataset)
    
    train_subjects = dataset[:train_len]
    val_subjects = dataset[train_len:train_len+val_len]
    test_subjects = dataset[train_len+val_len:]
    
    return train_subjects, val_subjects, test_subjects


def rename_keys(file, groups, split_type):
    for key in groups:
        base_key = key.split("_")[-1]
        new_key = split_type + "_" + base_key
        
        file.move(key, new_key)
        
    
def remove_keys(file):
    print("Removing prefixes...")
    print(f"Before: {file.keys()}")
    subj_keys = file.keys()
    for key in subj_keys:
        new_key = key.split("_")[-1]
        file.move(key, new_key)        
        
        
def train_val_test_split(basepath, filename, train_ratio, val_ratio, test_ratio, remove_prefix):
    """
    Function to split a single transformed hdf5 dataset into train, validation and test datasets by tagging them in subject group name:
    
    {split_type}_{subject_id}
    
    E.g. train_800002
    
    # TODO: Currently split by number of subjects, but subjects can have different number of records. In future we might want to count number of records for each subject and split based on that.
    """    
    assert os.path.exists(basepath+filename), f"{basepath+filename} does not exist..."

    with h5py.File(basepath + filename, 'a') as file:
        print("Opened: " + basepath+filename)
        subj_keys = list(file.keys())
        
        if remove_prefix:
            remove_keys(file)
        else:
            train, val, test = split_dataset(subj_keys, train_ratio, val_ratio, test_ratio)

            rename_keys(file, train, "train")
            rename_keys(file, val, "val")
            rename_keys(file, test, "test")

def generate_config_file(basepath, filenames, train_ratio, val_ratio, test_ratio):
    data = {}
    
    for filename in filenames:
        with h5py.File(basepath + filename, 'a') as file:
            print("Opened: " + basepath+filename)
            subj_keys = list(file.keys())
            train, val, test = split_dataset(subj_keys, train_ratio, val_ratio, test_ratio)
            data[filename] = {
                    "train": train,
                    "val": val,
                    "test": test
                }
    
    with open(basepath + 'split_config.json', 'w') as f:
        json.dump(data, f)
        print("=======================================================")
        print(f"JSON config saved to: {basepath + 'split_config.json'}")
        print("=======================================================")
    
    
if __name__ == '__main__':
    CLI=argparse.ArgumentParser()
    CLI.add_argument("--basepath", type=str)
    CLI.add_argument("--files", nargs="*", type=str,  default=[]) # any type/callable can be used here
    CLI.add_argument("--train_ratio", type=float, default=0.75)
    CLI.add_argument("--val_ratio", type=float, default=0.10)
    CLI.add_argument("--test_ratio", type=float, default=0.15)    
    CLI.add_argument("--list_subjects", action='store_true')  
    CLI.add_argument('--generate_config_file', action='store_true')
    CLI.add_argument('--rename_keys', action='store_true')
    CLI.add_argument("--remove_prefix", action='store_true')
    args = CLI.parse_args()
    
    bpath = args.basepath
    filenames = args.files
    remove_prefix = args.remove_prefix

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    assert train_ratio+val_ratio+test_ratio == 1
    

    if args.list_subjects:
        for file in filenames:
            # Listing keys for each file
            with h5py.File(bpath + file, 'a') as f:
                print(f"{file} subject keys: {f.keys()}")

    elif args.generate_config_file:
        print("Generating config file...")
        generate_config_file(args.basepath, args.files, args.train_ratio, args.val_ratio, args.test_ratio)
        
    elif args.rename_keys:
        print("Renaming keys...")
        for file in filenames:
            train_val_test_split(bpath, file, train_ratio, val_ratio, test_ratio, remove_prefix)    

            # Testing if keys are changed
            with h5py.File(bpath + file, 'a') as f:
                print(f"After: {f.keys()}")
    
    
    
