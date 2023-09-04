import h5py
import os


def print_dataset_info():
    """
    Method for gathering information for report table X about dataset details.
    
    - Number of subjects
    - Number of records
    - Length of dataset measured in 30s epochs
    """
    print("Hello world from dataset_details.py")
    
    dataset_path = "/users/engholma/mnt/transformed/"
    #dataset_path = "/home/alec/repos/data/testing_split_delete_later/"
    
    datasets = [f.split(".")[0] for f in os.listdir(dataset_path) if ".hdf5" in f]
    
    for dataset_name in datasets:
        with h5py.File(dataset_path + dataset_name + ".hdf5", "r") as f:
            subj_ids = f.keys()
            num_subjects = len(subj_ids)
            num_records = 0
            total_num_epochs = 0

            for subj in subj_ids:
                #print(f"Reading subject ----- {subj} -----")
                subject_object = f[subj]
                
                rec_keys = subject_object.keys()
                num_records += len(rec_keys)
                
                for rec in rec_keys:
                    #print(f"Reading record ----- {rec} -----")
                    rec_len_read = False
                    
                    psg = subject_object[rec]["psg"]
                    
                    channel_keys = psg.keys()
                    
                    for channel_key in channel_keys:
                        #print(f"Reading channel ----- {channel_key} -----")
                        chan = psg[channel_key]
                        rec_epochs = int(chan.shape[0] / 128 / 30)
                        
                        if not rec_len_read: 
                            total_num_epochs += rec_epochs
                            rec_len_read = True

        print("--------------------------------------------------------------------------------")
        print(f"Succesfully read dataset: {dataset_name}")
        print(f"Number of subjects: {num_subjects}")
        print(f"Number of records: {num_records}")
        print(f"Total number of epochs in dataset: {total_num_epochs}")
        print("--------------------------------------------------------------------------------")
        

if __name__ == "__main__":
    print_dataset_info()
    
