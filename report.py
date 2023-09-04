# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:52:26 2023

@author: Jesper Strøm
"""

import sys
import os
import pickle
import torch
import io
import numpy as np
sys.path.append(os.path.abspath('../..'))
from shared.utility import acc, kappa, f1

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def main():
    run_id = sys.argv[1]
    
    base = f"results/{run_id}/"
  
    files = os.listdir(base)
    
    files = [f for f in os.listdir(base) if os.path.isfile(base+f)]
    
    scores = {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    acc_overall = []
    kap_overall = []
    f1_1_overall = []
    f1_2_overall = []
    f1_3_overall = []
    f1_4_overall = []
    f1_5_overall = []
    f1_mean_overall = []
    
    for file in files:
        with open(base+file, 'rb') as f:
            data = CPU_Unpickler(f).load()
            ensemble_preds = data['channel_pred']
            labels = data['labels']
            
            a = acc(ensemble_preds, labels)
            k = kappa(ensemble_preds, labels)
            f = f1(ensemble_preds, labels, average=False)
            f_mean = f1(ensemble_preds, labels, average=True)
            
            f_1 = f[0]
            f_2 = f[1]
            f_3 = f[2]
            f_4 = f[3]
            f_5 = f[4]                
            
            filename_split = file.split(".")
 
            dataset = filename_split[0]
            

            if dataset not in scores:
                print(dataset)
                scores[dataset] = {'a': [],'k': [], 'f1_mean': [],'f1_c1': [],'f1_c2': [],'f1_c3': [],'f1_c4': [],'f1_c5': [], 'preds':[], 'labels':[]}
                
            scores[dataset]['a'].append(a)
            scores[dataset]['k'].append(k)
            scores[dataset]['f1_c1'].append(f_1)
            scores[dataset]['f1_c2'].append(f_2)
            scores[dataset]['f1_c3'].append(f_3)
            scores[dataset]['f1_c4'].append(f_4)
            scores[dataset]['f1_c5'].append(f_5)
            scores[dataset]['preds'].append(torch.squeeze(ensemble_preds))
            scores[dataset]['labels'].append(torch.squeeze(labels))
            scores[dataset]['f1_mean'].append(f_mean)
            
            acc_overall.append(a)
            kap_overall.append(k)
            f1_1_overall.append(f_1)
            f1_2_overall.append(f_2)
            f1_3_overall.append(f_3)
            f1_4_overall.append(f_4)
            f1_5_overall.append(f_5)
            f1_mean_overall.append(f_mean)
                
    for key in scores:
        acc_scores = torch.stack(scores[key]['a'])
        kappa_scores = torch.stack(scores[key]['k'])
        f1_c1 = torch.stack(scores[key]['f1_c1'])
        f1_c2 = torch.stack(scores[key]['f1_c2'])
        f1_c3 = torch.stack(scores[key]['f1_c3'])
        f1_c4 = torch.stack(scores[key]['f1_c4'])
        f1_c5 = torch.stack(scores[key]['f1_c5'])
        f1_mean = torch.stack(scores[key]['f1_mean'])
        
        preds = scores[key]['preds']
        labels = scores[key]['labels']

        preds = torch.cat(preds)
        labels = torch.cat(labels)
        
        acc_mean = torch.sum(acc_scores)/len(acc_scores)
        k_mean = torch.sum(kappa_scores)/len(kappa_scores)
        f1_c1_mean = torch.sum(f1_c1)/len(f1_c1)
        f1_c2_mean = torch.sum(f1_c2)/len(f1_c2)
        f1_c3_mean = torch.sum(f1_c3)/len(f1_c3)
        f1_c4_mean = torch.sum(f1_c4)/len(f1_c4)
        f1_c5_mean = torch.sum(f1_c5)/len(f1_c5)
        f1_mean = torch.sum(f1_mean)/len(f1_mean)

        isExist = os.path.exists(base+"test_metrics")
        if not isExist:
            os.makedirs(base+"test_metrics")
            
        with open(base+"test_metrics/"+key, "ab") as f:
            pickle.dump({
                            'dataset_name': key,
                            'labels': labels,
                            'preds': preds,
                            'kappa_scores': kappa_scores,
                            'acc': acc_mean,
                            'kappa': k_mean,
                            'f1_c1': f1_c1_mean,
                            'f1_c2': f1_c2_mean,
                            'f1_c3': f1_c3_mean,
                            'f1_c4': f1_c4_mean,
                            'f1_c5': f1_c5_mean,
                            'f1_mean': f1_mean,
                        }, f)
        
        print(f"Report for dataset: {key}")
        print(f"Acc: {acc_mean}")
        print(f"Kappa: {k_mean}")
        print(f"F1 class 1: {f1_c1_mean}")
        print(f"F1 class 2: {f1_c2_mean}")
        print(f"F1 class 3: {f1_c3_mean}")
        print(f"F1 class 4: {f1_c4_mean}")
        print(f"F1 class 5: {f1_c5_mean}")
        print(f"F1 mean: {f1_mean}")
        print("\n")
        
    print(f"Report overall:")
    print(f"Acc: {sum(acc_overall)/len(acc_overall)}")
    print(f"Kappa: {sum(kap_overall)/len(kap_overall)}")
    print(f"F1 class 1: {sum(f1_1_overall)/len(f1_1_overall)}")
    print(f"F1 class 2: {sum(f1_2_overall)/len(f1_2_overall)}")
    print(f"F1 class 3: {sum(f1_3_overall)/len(f1_3_overall)}")
    print(f"F1 class 4: {sum(f1_4_overall)/len(f1_4_overall)}")
    print(f"F1 class 5: {sum(f1_5_overall)/len(f1_5_overall)}")
    print(f"F1 mean: {sum(f1_mean_overall)/len(f1_mean_overall)}")
    print("\n")

if __name__ == '__main__':
    main()
