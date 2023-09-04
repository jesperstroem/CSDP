import os
import sys
import json

def get_subs(base, splittype, split):
    with open(f"{base}/{splittype}_list_{split}.txt", "r") as f:
        lines = ["sub-0"+l.split("/")[3][1:] for l in f.readlines()]
    return set(lines)

def main():
    split_list_base = "/home/jose/repo/Speciale2023/shared/eesm_splits/huy_splits/file_list_eareeg/"
    
    files = os.listdir(split_list_base)
    
    files = [f.split("_")[-1].split(".")[0] for f in os.listdir(split_list_base) if f.endswith(".txt")]
    splits = set(files)
    
    for split in splits:
        trains = get_subs(split_list_base, "train", split)
        vals = get_subs(split_list_base, "eval", split)
        tests = get_subs(split_list_base, "test", split)
        
        dic = {
            "eesm": {"train": list(trains),
                     "val": list(vals),
                     "test": list(tests)},
        }

        json_object = json.dumps(dic, indent=4)

        with open(f"/home/jose/repo/Speciale2023/shared/eesm_splits/our_splits/{split}.json", "w") as outfile:
            outfile.write(json_object)
        
        
if __name__ == '__main__':
    main()
    
