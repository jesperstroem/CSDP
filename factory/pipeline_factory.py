from abc import ABC, abstractmethod

class Pipeline_Factory(ABC):
    @abstractmethod
    def create_training_pipeline(self):
        pass

    @abstractmethod
    def create_validation_pipeline(self):
        pass
    
    @abstractmethod
    def create_test_pipeline(self):
        pass

class USleep_Pipeline_Factory(Pipeline_Factory):
    def create_training_pipeline(self):
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

    def create_validation_pipeline(self):
        val_pipes = [Determ_sampler(dataset_args["base_path"],
                                    dataset_args["val"],
                                    train_args["datasplit_path"],
                                    split_type="val",
                                    num_epochs=35,
                                    single_channels = True,
                                    subject_percentage = train_args["subject_percentage"])]
        
        return val_pipes

    def create_test_pipeline(self):
        test_pipes = [Determ_sampler(dataset_args["base_path"],
                        dataset_args["test"],
                        train_args["datasplit_path"],
                        split_type="test",
                        num_epochs=35)]
        
        return test_pipes

class LSeqSleepNet_Pipeline_Factory(Pipeline_Factory):
    def create_training_pipeline(self):
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

    def create_validation_pipeline(self):
        val_pipes = [Determ_sampler(dataset_args["base_path"],
                                    dataset_args["val"],
                                    train_args["datasplit_path"],
                                    split_type="val",
                                    num_epochs=200,
                                    subject_percentage = train_args["subject_percentage"]),
                    Resampler(128, 100),
                    Spectrogram()]
        return val_pipes
    
    def create_test_pipeline(self):
        test_pipes = [Determ_sampler(dataset_args["base_path"],
                                     dataset_args["test"],
                                     train_args["datasplit_path"],
                                     split_type="test",
                                     num_epochs=200),
                    Resampler(128, 100),
                    Spectrogram()]
        return test_pipes