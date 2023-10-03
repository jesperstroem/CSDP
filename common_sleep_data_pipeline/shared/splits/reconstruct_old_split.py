import os
import sys
import json
import neptune

def main():
    run = neptune.init_run(project="NTLAB/bigsleep",api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzViZjJlYy00NDNhLTRhN2EtOGZmYy00NDEzODBmNTgxYzMifQ==", with_id="BIG-71")

    sets = ["abc", "ccshs", "cfs", "chat", "dcsm", "homepap", "mesa", "mros", "phys", "sedf_sc_physionet", "sedf_st_physionet", "shhs", "sof"]
    
    base = "datasets/users/engholma/mnt/transformed/"
    
    dic = {}
    
    for s in sets:
        train_data = list(run[f"datasets/users/engholma/mnt/transformed/{s}.hdf5/train"].fetch_values(include_timestamp=False)["value"])
        val_data = list(run[f"datasets/users/engholma/mnt/transformed/{s}.hdf5/val"].fetch_values(include_timestamp=False)["value"])
        test_data = list(run[f"datasets/users/engholma/mnt/transformed/{s}.hdf5/test"].fetch_values(include_timestamp=False)["value"])
        
        train_data = [t[6:] for t in train_data]
        val_data = [t[4:] for t in val_data]
        test_data = [t[5:] for t in test_data]
        
        dic[s] = { "train": train_data,
                   "val": val_data,
                   "test": test_data
                 }

    json_object = json.dumps(dic, indent=4)
    with open(f"/home/jose/repo/Speciale2023/shared/random_split.json", "w") as outfile:
        outfile.write(json_object)
        
        
        
if __name__ == '__main__':
    main()
    
