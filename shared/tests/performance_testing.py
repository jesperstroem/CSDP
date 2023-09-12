import timeit
import tracemalloc

import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import pickle
import statistics

from datastore_classes import CFS

"""
Program for performance testing different types of data formats.

The formats will be evaluated on time and memory consumption

The formats that will be tested are: HDF5 and Parquet.
Feather was not considered as it is not expected to be used as a long term file storage


# TODO: Create data files with all needed channels
# Implement choosing random of EEG and EOG
# Test if this has an impact
"""

def test_load_parquet(path):
    data = pq.read_table(f"{path}/psg.parquet").column("C3")
    
    return data
    
def test_load_hdf5(path, iterations):
    f = h5py.File(f"{path}/psg.hdf5", "r")
    data = f["C3"][()]
    f.close()
    
    return data

def test_load_pickle(path):
    with open(f"{path}/psg.pickle", "rb") as f:
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
    with h5py.File(f"{output_path}/psg.hdf5", "w") as f:
        for channel_name in x.keys():
            f.create_dataset(channel_name, data=x[channel_name][0])
            
def write_record_to_database_pickle(output_path, x):
    with open(f"{output_path}/psg.pickle", "wb") as f:
        pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def init_test_data(datapath, outputpath):
    cfs = CFS(max_num_subjects=1,
              dataset_path=datapath,
              output_path=outputpath,
              port_on_init=False)

    x, _ = cfs.read_psg((f"{datapath}polysomnography/edfs/cfs-visit5-800002.edf",
                        f"{datapath}polysomnography/annotations-events-profusion/cfs-visit5-800002-profusion.xml"))

    # Write to Parquet
    write_record_to_database_parquet(outputpath, x)

    write_record_to_database_hdf5(outputpath, x)
    
    # Write to Pickle
    write_record_to_database_pickle(outputpath, x)

def run_performance_testv2(datapath, fileformat):
    num_iterations = 100
    
    time_data = {}
    mem_data = {}

    tracemalloc.start()

    t = timeit.timeit(f"test_load_{fileformat}(o)", setup=f"o='{datapath}'", globals=globals(), number=num_iterations)

    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    time_data[fileformat] = t
    mem_data[fileformat] = peak_mem

    print(t)
    print(peak_mem)

def run_performance_test(datapath,
                         outputpath):
    
    init_test_data(datapath, outputpath)
    
    num_iterations = 100
    
    time_data = {}
    mem_data = {}

    for fileformat in ['parquet', 'hdf5', 'pickle']:
        times = []
        tracemalloc.start()

        for _ in range(num_iterations):
            t = timeit.timeit(f"test_load_{fileformat}(o)", setup=f"o='{outputpath}'", globals=globals(), number=1)
            times.append(t)

        peak_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        time_data[fileformat] = times
        mem_data[fileformat] = peak_mem

    with open(f'{outputpath}/time.pickle', 'wb') as handle:
        pickle.dump(time_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{outputpath}/memory.pickle', 'wb') as handle:
        pickle.dump(mem_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for fileformat in ['parquet', 'hdf5', 'pickle']:
        print(f"Mean of {fileformat}: {statistics.mean(time_data[fileformat])}\n")
        print(f"Std deviation of {fileformat}: {statistics.stdev(time_data[fileformat])}\n")
        print(f"Peak memory of {fileformat}: {mem_data[fileformat]}\n")
        print("\n\n")
