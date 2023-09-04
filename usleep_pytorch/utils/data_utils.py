import usleep_pytorch.dataset_classes as dataset_classes

def get_data(args):
    dm = dataset_classes.available_datasets[args.dataset_type](**vars(args))

    return dm