from lightning_models.factories.pipeline_factory import Pipeline_Factory
from shared.pipeline.sampler import Sampler
from shared.pipeline.augmenters import Augmenter
from shared.pipeline.resampler import Resampler
from shared.pipeline.spectrogram import Spectrogram
from shared.pipeline.determ_sampler import Determ_sampler

class USleep_Pipeline_Factory(Pipeline_Factory):
    def create_training_pipeline(self, train_args, dataset_args):
        aug = train_args["augmentation"]

        if aug["use"] == True:
            print("Running with augmentation")
            train_pipes = [Sampler(dataset_args["base_path"],
                                   dataset_args["train"],
                                   dataset_args["datasplit_path"],
                                   split_type="train",
                                   num_epochs=35,
                                   subject_percentage = train_args["subject_percentage"]),
                           Augmenter(
                               min_frac=aug["min_frac"], 
                               max_frac=aug["max_frac"], 
                               apply_prob=aug["apply_prob"], 
                               sigma=aug["sigma"],
                               mean=aug["mean"]
                           )]
        else:
            train_pipes = [Sampler(dataset_args["base_path"],
                                   dataset_args["train"],
                                   train_args["datasplit_path"],
                                   split_type="train",
                                   num_epochs=35,
                                   subject_percentage = train_args["subject_percentage"])]
        return train_pipes

    def create_validation_pipeline(self, train_args, dataset_args):
        val_pipes = [Determ_sampler(dataset_args["base_path"],
                                    dataset_args["val"],
                                    train_args["datasplit_path"],
                                    split_type="val",
                                    num_epochs=35,
                                    subject_percentage = train_args["subject_percentage"])]
        
        return val_pipes

    def create_test_pipeline(self, train_args, dataset_args):
        test_pipes = [Determ_sampler(dataset_args["base_path"],
                        dataset_args["test"],
                        train_args["datasplit_path"],
                        split_type="test",
                        num_epochs=35)]
        
        return test_pipes

class LSeqSleepNet_Pipeline_Factory(Pipeline_Factory):
    def create_training_pipeline(self, train_args, dataset_args):
        aug = train_args["augmentation"]
        
        if aug["use"] == True:
            train_pipes = [Sampler(dataset_args["base_path"],
                                   dataset_args["train"],
                                   train_args["datasplit_path"],
                                   split_type="train",
                                   num_epochs=200,
                                   subject_percentage = train_args["subject_percentage"]),
                           Augmenter(
                               min_frac=aug["min_frac"], 
                               max_frac=aug["max_frac"], 
                               apply_prob=aug["apply_prob"], 
                               sigma=aug["sigma"],
                               mean=aug["mean"]
                           ),
                           Resampler(128, 100),
                           Spectrogram()]
        else:
            train_pipes = [Sampler(dataset_args["base_path"],
                                   dataset_args["train"],
                                   train_args["datasplit_path"],
                                   split_type="train",
                                   num_epochs=200,
                                   subject_percentage = train_args["subject_percentage"]),
                           Resampler(128, 100),
                           Spectrogram()]
        return train_pipes

    def create_validation_pipeline(self, train_args, dataset_args):
        val_pipes = [Determ_sampler(dataset_args["base_path"],
                                    dataset_args["val"],
                                    train_args["datasplit_path"],
                                    split_type="val",
                                    num_epochs=200,
                                    subject_percentage = train_args["subject_percentage"]),
                    Resampler(128, 100),
                    Spectrogram()]
        return val_pipes
    
    def create_test_pipeline(self, train_args, dataset_args):
        test_pipes = [Determ_sampler(dataset_args["base_path"],
                                     dataset_args["test"],
                                     train_args["datasplit_path"],
                                     split_type="test",
                                     num_epochs=200),
                    Resampler(128, 100),
                    Spectrogram()]
        return test_pipes