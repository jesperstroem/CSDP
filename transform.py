import yaml
from yaml.loader import SafeLoader
import importlib

def main():
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
        
        module = importlib.import_module("common_sleep_data_store.datastore_classes")
        class_ = getattr(module, name)
        _ = class_(None, path, target, scale_and_clip, output_sample_rate)
        
if __name__ == "__main__":
    main() 
