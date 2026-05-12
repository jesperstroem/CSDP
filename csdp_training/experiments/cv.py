import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
import h5py
from sklearn.model_selection import train_test_split
from csdp_pipeline.pipeline_elements.samplers import Random_Sampler, Determ_sampler, SamplerConfiguration
from csdp_pipeline.pipeline_elements.pipeline import PipelineConfiguration
from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
from csdp_training.lightning_models.usleep import USleep_Lightning
from copy import deepcopy
from csdp_pipeline.pipeline_elements.models import Split, Dataset_Split
from sklearn.model_selection import KFold
import os

class CV_Experiment:

    def __init__(self,
                 base_net: USleep_Lightning,
                 base_data_path: str,
                 datasets: list[str],
                 training_epochs: int,
                 batch_size: int,
                 num_folds: int,
                 earlystopping_patience: int,
                 logging_folder: str,
                 num_validation_subjects: int = 1,
                 batches_per_epoch: int = 100,
                 pick_all_channels: [bool] = [False, False, True],
                 test_first: bool = False,
                 pipeline_configuration: PipelineConfiguration = PipelineConfiguration(),
                 continue_existing: bool = False,
                 mlflow_run_id: str | None = None,
                 split_filepath = None):
        """_summary_

        Args:
            base_net (USleep_Lightning): A U-Sleep lightning model
            dataset_paths (list[str]): A list of exact filepaths to the datasets used for the experiment
            training_epochs (int): Number of training epochs per fold
            batch_size (int): Batchsize during training
            num_val_subjects (int, optional): The number of validation subjects per fold. Defaults to 1.
            batches_per_epoch (int, optional): Number of mini-batches per epoch. Defaults to 100.
            pick_all_channels (bool, optional): If True, the sampled data will contain all available channels. If False, one random EEG and EOG is picked. Defaults to False.
            test_first (bool, optional): If True, the base model will be tested first on all records. Defaults to False.
            pipeline_configuration (PipelineConfiguration, optional): A desired pipeline configuration. The default parameter has no pipes. Defaults to PipelineConfiguration().
            experiment_name (str, optional): The name of the experiment and the name of the test output folder. Defaults to "LOSO".
            mlflow_run_id (str | None, optional): The run ID of an active MLflow run to log into. Defaults to None.
        """

        self.continue_existing = continue_existing
        self.batches_per_epoch = batches_per_epoch
        self.pick_all_channels = pick_all_channels
        self.base_data_path = base_data_path
        self.dataset_paths = [f"{base_data_path}/{p}" for p in datasets]
        self.pipeline_configuration = pipeline_configuration
        self.logging_folder = logging_folder
        self.training_epochs = training_epochs
        self.mlflow_run_id = mlflow_run_id
        self.base_net = base_net
        self.batch_size = batch_size
        self.earlystopping_patience = earlystopping_patience
        self.num_folds = num_folds
        self.num_validation_subjects = num_validation_subjects

        os.makedirs(logging_folder, exist_ok=True)

        if split_filepath != None:
            splits = []
            filepaths = os.listdir(split_filepath)
            filepaths = [f"{split_filepath}/{p}" for p in filepaths]
            
            for path in filepaths:
                splits.append(Split.file(path))

            self.split_data = splits
        else:
            self.split_data: [Split] = self.__create_split(self.dataset_paths,
                                                           num_folds=num_folds,
                                                           num_validation_subjects=num_validation_subjects)
        
            self.__save_split_data(self.split_data, logging_folder)

        self.test_first = test_first
        self.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        if self.test_first == True:
            global_split: Split = self.__create_global_split(self.dataset_paths)

            wrapper = self.__create_wrapper(global_split, self.batch_size)

            trainer = self.__init_trainer(max_epochs=self.training_epochs,
                                          split_data = global_split,
                                          split_name="Global Test")
            
            base_net = deepcopy(self.base_net)
            
            self.__test(trainer, wrapper, base_net, split_name="Global Test", load_best_model=False)

        for _, split in enumerate(self.split_data):
            split_name = f"Split_{split.id}"

            wrapper = self.__create_wrapper(split, self.batch_size)

            trainer = self.__init_trainer(max_epochs=self.training_epochs,
                                          split_data=split,
                                          split_name=split_name)

            base_net = deepcopy(self.base_net)

            net, trainer = self.__train(base_net,
                                         wrapper,
                                         trainer)

            self.__test(trainer,
                        wrapper,
                        net=net,
                        split_name=split_name)

    def __train(self,
                net: USleep_Lightning,
                wrapper: Dataloader_Factory,
                trainer: pl.Trainer):
        
        train_loader = wrapper.training_loader(num_workers=8)
        val_loader = wrapper.validation_loader(num_workers=1)

        trainer.fit(net, train_loader, val_loader)

        return net, trainer


    def __test(self,
               trainer: pl.Trainer,
               wrapper: Dataloader_Factory,
               net: USleep_Lightning,
               split_name: str,
               load_best_model = True):        
        loader = wrapper.testing_loader(num_workers=1)

        net.run_test(trainer,
                     loader, 
                     output_folder_prefix=f"{self.logging_folder}/sleep_scorings/split_{split_name}",
                     load_best_model=load_best_model)

    def __init_trainer(self,
                       max_epochs: int,
                       split_data: Split,
                       split_name: str) -> pl.Trainer:
        
        checkpoint_callback = ModelCheckpoint(dirpath=f"{self.logging_folder}/weights", filename=f"best-{split_name}", monitor="valKap", mode="max")
        
        early_stopping = EarlyStopping(
            monitor="valKap",
            min_delta=0.00,
            patience=self.earlystopping_patience,
            verbose=True,
            mode="max"
        )

        callbacks = [checkpoint_callback, early_stopping]

        if self.mlflow_run_id is not None:
            mlflow.MlflowClient().log_dict(
                run_id=self.mlflow_run_id,
                dictionary=split_data.get_dict(),
                artifact_file=f"{split_name}_split.json",
            )
            logger = MLFlowLogger(run_id=self.mlflow_run_id, prefix=split_name)
        else:
            logger = None
        
        trainer = pl.Trainer(logger=logger,
                             max_epochs=max_epochs,
                             callbacks=callbacks,
                             accelerator=self.accelerator,
                             devices=1,
                             num_nodes=1)
        
        return trainer
    
    def __create_wrapper(self,
                         split: Split,
                         batch_size):
        
        train_sampler = Random_Sampler(split,
                                    num_epochs=35,
                                    num_iterations=batch_size*self.batches_per_epoch)
        
        val_sampler = Determ_sampler(split,
                                     get_all_channels=self.pick_all_channels[1],
                                    split_type="val")
        
        test_sampler = Determ_sampler(split,
                                      get_all_channels=self.pick_all_channels[2],
                                      split_type="test")
        
        samplers = SamplerConfiguration(train_sampler,
                                        val_sampler,
                                        test_sampler)
        
        pipes = self.pipeline_configuration

        wrapper = Dataloader_Factory(batch_size,
                                     samplers,
                                     pipes)
    
        return wrapper
    
    def __create_split(self,
                       dataset_filepaths: list[str],
                       num_folds,
                       num_validation_subjects = 1) -> [Split]:
        all_subs = []
        all_split_data: list[Split] = []

        for file in dataset_filepaths:
            with h5py.File(file, "r") as hdf5:
                hdf5 = hdf5["data"]
                subs = list(hdf5.keys())

                subs = [(file, sub) for sub in subs]
                all_subs.extend(subs)
        
        kf = KFold(n_splits=num_folds, shuffle=True)
        
        for i, (train_index, test_index) in enumerate(kf.split(all_subs)):
            split_data = Split(id=i,
                               dataset_splits=[],
                               base_data_path=self.base_data_path)
            
            train_records = [all_subs[i] for i in train_index]

            train_records, val_records = train_test_split(train_records,
                                                          test_size=num_validation_subjects)

            test_records = [all_subs[i] for i in test_index]
            
            for file in dataset_filepaths:
                dataset_split = Dataset_Split(file,
                                              train=[],
                                              val=[],
                                              test=[])
                
                dataset_split.train = [x[1] for x in list(filter(lambda x: file == x[0], train_records))]
                dataset_split.val = [x[1] for x in list(filter(lambda x: file == x[0], val_records))]
                dataset_split.test = [x[1] for x in list(filter(lambda x: file == x[0], test_records))]

                split_data.dataset_splits.append(dataset_split)
            
            all_split_data.append(split_data)

        return all_split_data
    

    def __save_split_data(self, splits: [Split], logging_folder):
        os.makedirs(f"{logging_folder}/splits", exist_ok=True)

        for _, split in enumerate(splits):
            split: Split = split
            split.dump_file(path=f"{logging_folder}/splits")

    def __create_global_split(self, dataset_filepaths: list[str]):
        split_data = Split(id="test",
                           dataset_splits=[],
                           base_data_path=self.base_data_path)
        
        for dataset_filepath in dataset_filepaths:
            with h5py.File(dataset_filepath, "r") as hdf5:
                hdf5 = hdf5["data"]

                subs = list(hdf5.keys())

            split_data.dataset_splits.append(Dataset_Split(dataset_filepath,
                                                        [],
                                                        [],
                                                        subs))
        
        return split_data
