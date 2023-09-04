import h5py
import os
import pickle
import statistics


def calculate_mean_values_multiple_runs(results_path):
    acc_scores = []
    kap_scores = []
    f1_wake_scores = []
    f1_n1_scores = []
    f1_n2_scores = []
    f1_n3_scores = []
    f1_rem_scores = []
    f1_mean_scores = []
    
    runs = [x for x in os.listdir(results_path)]
    
    print(f"Calculating values based off of following {len(runs)} runs: {runs}")
    
    for run in runs:        
        pickled_test_results = [x for x in os.listdir(results_path + run + "/test_metrics/")]
        
        assert len(pickled_test_results) == 1
        
        with open(f"{results_path}{run}/test_metrics/{pickled_test_results[0]}",'rb') as fp:
            obj = pickle.load(fp)
            
            acc_scores.append(float(obj.get("acc")))
            kap_scores.append(float(obj.get("kappa")))
            f1_wake_scores.append(float(obj.get("f1_c1")))
            f1_n1_scores.append(float(obj.get("f1_c2")))
            f1_n2_scores.append(float(obj.get("f1_c3")))
            f1_n3_scores.append(float(obj.get("f1_c4")))
            f1_rem_scores.append(float(obj.get("f1_c5")))
            f1_mean_scores.append(float(obj.get("f1_mean")))
    
    assert len(acc_scores) == 20
    assert len(kap_scores) == 20
    assert len(f1_wake_scores) == 20
    assert len(f1_n1_scores) == 20
    assert len(f1_n2_scores) == 20
    assert len(f1_n3_scores) == 20
    assert len(f1_rem_scores) == 20
    assert len(f1_mean_scores) == 20
    
    print("===============================================")
    print(f"acc mean: {statistics.fmean(acc_scores)}")
    print(f"kap mean: {statistics.fmean(kap_scores)}")
    print(f"f1 wake mean: {statistics.fmean(f1_wake_scores)}")
    print(f"f1 n1 mean: {statistics.fmean(f1_n1_scores)}")
    print(f"f1 n2 mean: {statistics.fmean(f1_n2_scores)}")
    print(f"f1 n3 mean: {statistics.fmean(f1_n3_scores)}")
    print(f"f1 rem mean: {statistics.fmean(f1_rem_scores)}")
    print(f"f1 mean mean: {statistics.fmean(f1_mean_scores)}")
    print("===============================================")

    
def calculate_mean_values_single_run(results_path):
    acc_scores = []
    kap_scores = []
    f1_wake_scores = []
    f1_n1_scores = []
    f1_n2_scores = []
    f1_n3_scores = []
    f1_rem_scores = []
    f1_mean_scores = []
    
    runs = sorted([x for x in os.listdir(results_path)])
    
    print(f"Calculating values based off of following {len(runs)} subjects: {runs}")
    
    for run in runs:     
        
        with open(f"{results_path}{run}",'rb') as fp:
            obj = pickle.load(fp)
            print("=====================================")
            print(f"Subject: {run}")
            print(f"Accurracy: {float(obj.get('acc'))}")
            print(f"Kappa: {float(obj.get('kappa'))}")
            print(f"F1 wake: {float(obj.get('f1_c1'))}")
            print(f"F1 N1: {float(obj.get('f1_c2'))}")
            print(f"F1 N2: {float(obj.get('f1_c3'))}")
            print(f"F1 N3: {float(obj.get('f1_c4'))}")
            print(f"F1 REM: {float(obj.get('f1_c5'))}")
            print(f"F1 mean: {float(obj.get('f1_mean'))}")
            print("=====================================")
            
            acc_scores.append(float(obj.get("acc")))
            kap_scores.append(float(obj.get("kappa")))
            f1_wake_scores.append(float(obj.get("f1_c1")))
            f1_n1_scores.append(float(obj.get("f1_c2")))
            f1_n2_scores.append(float(obj.get("f1_c3")))
            f1_n3_scores.append(float(obj.get("f1_c4")))
            f1_rem_scores.append(float(obj.get("f1_c5")))
            f1_mean_scores.append(float(obj.get("f1_mean")))
    
    assert len(acc_scores) == 20
    assert len(kap_scores) == 20
    assert len(f1_wake_scores) == 20
    assert len(f1_n1_scores) == 20
    assert len(f1_n2_scores) == 20
    assert len(f1_n3_scores) == 20
    assert len(f1_rem_scores) == 20
    assert len(f1_mean_scores) == 20
    
    print("===============================================")
    print(f"acc mean: {statistics.fmean(acc_scores)}")
    print(f"kap mean: {statistics.fmean(kap_scores)}")
    print(f"f1 wake mean: {statistics.fmean(f1_wake_scores)}")
    print(f"f1 n1 mean: {statistics.fmean(f1_n1_scores)}")
    print(f"f1 n2 mean: {statistics.fmean(f1_n2_scores)}")
    print(f"f1 n3 mean: {statistics.fmean(f1_n3_scores)}")
    print(f"f1 rem mean: {statistics.fmean(f1_rem_scores)}")
    print(f"f1 mean mean: {statistics.fmean(f1_mean_scores)}")
    print("===============================================")
    
    
if __name__ == "__main__":
    #path = "/home/alec/repos/usleep_results_finetune_crossval/"
    path = "/home/alec/repos/raw_BIG-263/test_metrics/"
    calculate_mean_values_single_run(path)
    
