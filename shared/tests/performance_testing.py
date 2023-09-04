import timeit
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
import os
from h5py import File
import h5py
import pickle
import sys

sys.path.append(os.path.abspath('../../..'))
import SleepDataPipeline


"""
Program for performance testing different types of data formats.

The formats will be evaluated on time and memory consumption

The formats that will be tested are: HDF5 and Parquet.
Feather was not considered as it is not expected to be used as a long term file storage


# TODO: Create data files with all needed channels
# Implement choosing random of EEG and EOG
# Test if this has an impact


"""



def test_load_parquet():
    data = pq.read_table("./performance_test_data/psg.parquet").column("C3")
    #print(len(data))
    
    return data
    
    
def test_load_hdf5():
    f = h5py.File("./performance_test_data/psg.hdf5", "r")
    data = f["C3"][()]
    f.close()
    #print(len(data))
    
    return data


def test_load_pickle():
    with open(f"./performance_test_data/psg.pickle", "rb") as f:
        data = pickle.load(f)["C3"]
        
        return data
    
    
def write_record_to_database_parquet(output_path, x):
    psg_table = pa.table(
        {
            'C3': x['C3'][0],
            'C4': x['C4'][0],
            'M1': x['M1'][0],
            'M2': x['M2'][0],
            'LOC': x['LOC'][0],
            'ROC': x['ROC'][0],
        }
    )
    pq.write_table(psg_table, output_path + "psg.parquet")
    
    
def write_record_to_database_hdf5(output_path, x):
    with h5py.File(f"{output_path}psg.hdf5", "w") as f:
        for channel_name in x.keys():
            f.create_dataset(channel_name, data=x[channel_name][0])
            
        
     
    
def write_record_to_database_pickle(output_path, x):
    with open(f"{output_path}psg.pickle", "wb") as f:
        pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
def init_test_data():
    cfs = SleepDataPipeline.Cfs(max_num_subjects=1,dataset_path="../../data/cfs/", output_path="../../data/parquet_test/", port_on_init=False )
    
    
    x, y = cfs.read_psg(("/home/alec/repos/data/cfs/cfs/polysomnography/edfs/cfs-visit5-800002.edf/","/home/alec/repos/data/cfs/cfs/polysomnography/annotations-events-profusion/cfs-visit5-800002-profusion.xml"))

    # region Setup test databases
    
    # Write to Parquet
    write_record_to_database_parquet("./performance_test_data/", x)
    
    # Write to HDF5
    write_record_to_database_hdf5("./performance_test_data/", x)
    
    # Write to Pickle
    write_record_to_database_pickle("./performance_test_data/", x)
    # endregion
    
    
def main():
    # region Run performance test
    num_iterations = 100
    
    time_data = {}
    mem_data = {}
    
    
    tracemalloc.start()
    t = timeit.timeit("test_load_parquet()", globals=globals(), number=num_iterations)
    avg_time = t/num_iterations
    peak_mem = tracemalloc.get_traced_memory()[1]
    print(f"Parquet loading time based on {num_iterations} iterations: {avg_time}")
    print(f"    Memory consumption peak at: {peak_mem}")
    tracemalloc.stop()
    
    time_data["Parquet"] = avg_time
    mem_data["Parquet"] = peak_mem
    
    
    tracemalloc.start()
    t = timeit.timeit("test_load_hdf5()", globals=globals(), number=num_iterations)
    avg_time = t/num_iterations
    peak_mem = tracemalloc.get_traced_memory()[1]
    print(f"HDF5 loading time based on {num_iterations} iterations: {avg_time}")
    print(f"    Memory consumption peak at: {peak_mem}")
    tracemalloc.stop()
    
    time_data["HDF5"] = avg_time
    mem_data["HDF5"] = peak_mem
    
    
    tracemalloc.start()
    t = timeit.timeit("test_load_pickle()", globals=globals(), number=num_iterations)
    avg_time = t/num_iterations
    peak_mem = tracemalloc.get_traced_memory()[1]
    print(f"Pickle loading time based on {num_iterations} iterations: {avg_time}")
    print(f"    Memory consumption peak at: {peak_mem}")
    tracemalloc.stop()
    
    time_data["Pickle"] = avg_time
    mem_data["Pickle"] = peak_mem
    # endregion
    
    fig = plt.figure(figsize = (5, 5))
    plt.bar(list(time_data.keys()), list(time_data.values()))
    plt.xlabel("Data format")
    plt.ylabel("Seconds")
    plt.savefig("./performance_test_data/time_data.png")
    
    plt.bar(list(mem_data.keys()), list(mem_data.values()))
    plt.xlabel("Data format")
    plt.ylabel("Bytes")
    plt.savefig("./performance_test_data/mem_data.png")

    
if __name__ == "__main__":
    #init_test_data()    
    #exit()
    main()
    