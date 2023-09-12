import h5py
import pickle
import pyarrow.parquet as pq

path_to_data = "C:/Users/au588953/Git Repos/CSDP/"

def test_load_hdf5():
    f = h5py.File(f"{path_to_data}psg.hdf5", "r")
    data = f["C3"][()]
    f.close()

    return data

def test_load_parquet():
    data = pq.read_table(f"{path_to_data}psg.parquet").column("C3")
    
    return data

def test_load_pickle():
    with open(f"{path_to_data}psg.pickle", "rb") as f:
        data = pickle.load(f)["C3"]
        
    return data