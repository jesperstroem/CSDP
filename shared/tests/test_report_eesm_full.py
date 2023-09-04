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
    base = "/home/jose/repo/lseq_eesm_before_finetune/BIG-75/"
    
    records = os.listdir(base)
    scores = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    a_overall = []
    k_overall = []
    f1_overall = []
    f2_overall = []
    f3_overall = []
    f4_overall = []
    f5_overall = []
    fmean_overall = []
    
    for rec in records:
        subject = rec.split(".")[1].split("_")[0].split("-")[1]
        
        if subject not in scores.keys():
            scores[subject]= {'a': [],'k': [], 'f1': [],'f2': [],'f3': [],'f4': [],'f5': [],'fmean': []}
        
        with open(base+rec, 'rb') as f:
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
            
            a_overall.append(a)
            k_overall.append(k)
            f1_overall.append(f_1)
            f2_overall.append(f_2)
            f3_overall.append(f_3)
            f4_overall.append(f_4)
            f5_overall.append(f_5)
            fmean_overall.append(f_mean)

            scores[subject]['a'].append(a)
            scores[subject]['k'].append(k)
            scores[subject]['f1'].append(f_1)
            scores[subject]['f2'].append(f_2)
            scores[subject]['f3'].append(f_3)
            scores[subject]['f4'].append(f_4)
            scores[subject]['f5'].append(f_5)
            scores[subject]['fmean'].append(f_mean)
                
    
    for key in scores.keys():
        sub = scores[key]
        
        a = sub['a']
        k = sub['k']
        f_1 = sub['f1']
        f_2 = sub['f2']
        f_3 = sub['f3']
        f_4 = sub['f4']
        f_5 = sub['f5']
        fmean = sub['fmean']
        
        assert len(a) == 4
        assert len(k) == 4
        assert len(f_1) == 4
        assert len(f_2) == 4
        assert len(f_3) == 4
        assert len(f_4) == 4
        assert len(f_5) == 4
        assert len(fmean) == 4
        
        print(f"Subject: {key}")
        print(f"Acc: {round(float(sum(a)/len(a)),3)}")
        print(f"Kap: {round(float(sum(k)/len(k)),3)}")
        print(f"F1: {round(float(sum(f_1)/len(f_1)),3)}")
        print(f"F2: {round(float(sum(f_2)/len(f_2)),3)}")
        print(f"F3: {round(float(sum(f_3)/len(f_3)),3)}")
        print(f"F4: {round(float(sum(f_4)/len(f_4)),3)}")
        print(f"F5: {round(float(sum(f_5)/len(f_5)),3)}")
        print(f"Fmean: {round(float(sum(fmean)/len(fmean)),3)}")
        print("\n\n")
    
    print(f"Overall:")
    print(f"Acc: {round(float(sum(a_overall)/len(a_overall)),3)}")
    print(f"Kap: {round(float(sum(k_overall)/len(k_overall)),3)}")
    print(f"F1: {round(float(sum(f1_overall)/len(f1_overall)),3)}")
    print(f"F2: {round(float(sum(f2_overall)/len(f2_overall)),3)}")
    print(f"F3: {round(float(sum(f3_overall)/len(f3_overall)),3)}")
    print(f"F4: {round(float(sum(f4_overall)/len(f4_overall)),3)}")
    print(f"F5: {round(float(sum(f5_overall)/len(f5_overall)),3)}")
    print(f"Fmean: {round(float(sum(fmean_overall)/len(fmean_overall)),3)}")
    print("\n\n")
    

if __name__ == '__main__':
    main()
