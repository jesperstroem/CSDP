# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:32:03 2023

@author: Jesper Strøm
"""

import pickle
import os

class Storage(type):
    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, 'save_data') and 
                callable(subclass.save_data) and 
                hasattr(subclass, 'load_data') and 
                callable(subclass.load_data))

class PickleStorage(metaclass=Storage):
    @staticmethod
    def save_data(dir, data):
        with open(dir, 'wb') as file:
            pickle.dump(data, file)
    
    @staticmethod
    def load_data(dir):
        with open(dir, 'rb') as file:
            data = pickle.load(file)
            return data

class StorageAPI(metaclass=Storage):
    @staticmethod
    def save_data(dir, filename, data):
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        PickleStorage.save_data(dir+'/'+filename, data)
    
    @staticmethod
    def load_data(dir, filename):
        return PickleStorage.load_data(dir+'/'+filename)