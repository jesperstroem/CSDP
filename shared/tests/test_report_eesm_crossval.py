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
    print("WORKS")
    exit()
    base = "/home/jose/repo/lseq_eesm_after_finetune/"
    
    direc = os.listdir(base)
    scores = {}    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    accmean = []
    kapmean = []
    f_1mean = []
    f_2mean = []
    f_3mean = []
    f_4mean = []
    f_5mean = []
    f_m = []
    
    for subj in direc:
        records = os.listdir(base+subj)
        
        assert len(records) == 4
        
        accs = []
        kaps = []
        f_1s = []
        f_2s = []
        f_3s = []
        f_4s = []
        f_5s = []
        f_means = []
        
        for record in records:
            with open(base+subj+"/"+record, 'rb') as f:
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

                accs.append(a)
                kaps.append(k)
                f_1s.append(f_1)
                f_2s.append(f_2)
                f_3s.append(f_3)
                f_4s.append(f_4)
                f_5s.append(f_5)
                f_means.append(f_mean)
                
                
                accmean.append(a)
                kapmean.append(k)
                f_1mean.append(f_1)
                f_2mean.append(f_2)
                f_3mean.append(f_3)
                f_4mean.append(f_4)
                f_5mean.append(f_5)
                f_m.append(f_mean)
                
                
        
        print(f"Report for subject: {subj}")
        print(f"Acc: {round(float(sum(accs)/len(accs)),3)}")
        print(f"Kappa: {round(float(sum(kaps)/len(kaps)),3)}")
        print(f"F1 class 1: {round(float(sum(f_1s)/len(f_1s)),3)}")
        print(f"F1 class 2: {round(float(sum(f_2s)/len(f_2s)),3)}")
        print(f"F1 class 3: {round(float(sum(f_3s)/len(f_3s)),3)}")
        print(f"F1 class 4: {round(float(sum(f_4s)/len(f_4s)),3)}")
        print(f"F1 class 5: {round(float(sum(f_5s)/len(f_5s)),3)}")
        print(f"F1 mean: {round(float(sum(f_means)/len(f_means)),3)}")
        print("\n\n")
        
    print(f"Overall means:")
    print(f"Acc: {round(float(sum(accmean)/len(accmean)),3)}")
    print(f"Kappa: {round(float(sum(kapmean)/len(kapmean)),3)}")
    print(f"F1 class 1: {round(float(sum(f_1mean)/len(f_1mean)),3)}")
    print(f"F1 class 2: {round(float(sum(f_2mean)/len(f_2mean)),3)}")
    print(f"F1 class 3: {round(float(sum(f_3mean)/len(f_3mean)),3)}")
    print(f"F1 class 4: {round(float(sum(f_4mean)/len(f_4mean)),3)}")
    print(f"F1 class 5: {round(float(sum(f_5mean)/len(f_5mean)),3)}")
    print(f"F1 mean: {round(float(sum(f_m)/len(f_m)),3)}")
    print("\n\n")
        
        
    

if __name__ == '__main__':
    main()
