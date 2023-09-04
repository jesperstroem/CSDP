import DataClasses
import yaml
from yaml.loader import SafeLoader
import importlib

def main():
    target = "/home/alec/repos/data/hdf5"
    num_sub = None
    
    with open('preprocessing_args.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        datasets = data['datasets']
        target = data['target_path']
        parameters = data['parameters']
        scale_and_clip = parameters['scale_and_clip']
        output_sample_rate = parameters['output_sample_rate']
    
    for dataset in datasets:
        name = dataset['name']
        path = dataset['path']
        
        module = importlib.import_module("DataClasses")
        class_ = getattr(module, name)
        instance = class_(num_sub, path, target, scale_and_clip, output_sample_rate)
        
if __name__ == "__main__":
    main() 
