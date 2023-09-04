
import h5py
import os
import sys
import argparse

def count_records(base, files):
    for s in files:
        with h5py.File(f"{base}/{s}.hdf5", "r") as f:
            subjects = f.keys()
            train_subjects = [k for k in subjects if k.startswith("train")]            
            print(f"Number of train subjects: {len(train_subjects)}")
            val_subjects = [k for k in subjects if k.startswith("val")]
            print(f"Number of validation subjects: {len(val_subjects)}")
            test_subjects = [k for k in subjects if k.startswith("test")]
            print(f"Number of test subjects: {len(test_subjects)}")

            tot_records = 0
            for sub in subjects:
                records = f[sub]
                tot_records += len(records)
            
            print(f"Number of subjects for {s}: {len(subjects)}")    
            print(f"Number of records for {s}: {tot_records}")

def main():
    CLI=argparse.ArgumentParser()
    
    CLI.add_argument(
      "--basepath",
      type=str
    )
    
    CLI.add_argument(
      "--files",
      nargs="*",
      type=str,  # any type/callable can be used here
      default=[],
    )
    
    args = CLI.parse_args()

    base = args.basepath
    files = args.files
    count_records(base, files)


if __name__ == "__main__":
    main()
