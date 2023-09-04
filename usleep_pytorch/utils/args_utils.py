import argparse
import pytorch_lightning as pl

# BEFORE YOU BEGIN RUNNING ON LUMI
# Change batch size (depending on distributed setup, effective batch size should be 64)
# Change iterations (aka. fac)
# Change max epochs
# When training, use pretrained FALSE
# Change paths, aka result file, datasets path
# Change to run on all train datasets
# Change to run fit or test depending.. (in train_usleep.py)

# OTHER PRETRAINED MODELS ================================================================
# 12 hour with augmentation on LUMI: /users/engholma/repos/.neptune/U-Sleep/BIG-17/checkpoints/epoch=94-step=1900.ckpt
# 12 hour without augmentation on LUMI: /users/engholma/repos/.neptune/U-Sleep/BIG-18/checkpoints/epoch=101-step=2040.ckpt
# 12 hour with augmentation on AU: /home/alec/repos/Speciale2023/.neptune/epoch=94-step=1900.ckpt

# 72 hours with augmentation on LUMI: /users/engholma/repos/.neptune/U-Sleep/BIG-71/checkpoints/epoch=451-step=9040.ckpt
# 72 hours aug finetuned on EESM with augmentation: /users/engholma/repos/.neptune/U-Sleep/BIG-89/checkpoints/epoch=24-step=50.ckpt
# 72 hours with aug on LUMI USLEEP SPLIT: /users/engholma/repos/.neptune/U-Sleep/version_None/checkpoints/epoch=469-step=9400.ckpt # version_None should be fixed - dunno why its there?

class ArgCollection():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(add_help=False)
        
        # Model args ====================================================================== 
        batch_size = 64 # 64 according to U-Sleep
        fac = 8 # 15000 / batch_size according to report # 78 for ~5000 iterations # 8 for small run aka finetune and 10%
        iterations = batch_size * fac
        
        self.parser.add_argument("--num_epochs", default=35, type=int) # Prediction window
        self.parser.add_argument("--lr", default=1.0e-03, type=float) # 1.0e-07 according to U-Sleep
        self.parser.add_argument("--batch_size", default=batch_size, type=int)
        self.parser.add_argument("--ensemble_window_size", default=35, type=int)
        self.parser.add_argument("--iterations", default=iterations, type=int)
        self.parser.add_argument("--model_use_pretrained", default=True, type=bool)
        self.parser.add_argument("--model_pretrained_path", default="/users/engholma/repos/.neptune/U-Sleep/BIG-71/checkpoints/epoch=451-step=9040.ckpt", type=str)
        self.parser.add_argument("--array_id")

        # Neptune args =====================================================================
        self.parser.add_argument("--neptune_api_key", default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NjAwMjZkMC1iZTg1LTQ3ZTgtOGJhMC05MmU0YzcwZDU0ZDMifQ==", type=str)
        self.parser.add_argument("--neptune_project", default="NTLAB/bigsleep", type=str)
        self.parser.add_argument("--neptune_model_name", default="U-Sleep", type=str)
        
        # Trainer args ======================================================================
        self.parser.add_argument("--max_epochs", default=99999, type=int) # Run for max_epochs
        self.parser.add_argument("--accelerator", default="gpu", type=str)
        self.parser.add_argument("--devices", default=1, type=int) # 4 for training
        self.parser.add_argument("--num_nodes", default=1, type=int)
        self.parser.add_argument("--strategy", default="ddp_find_unused_parameters_false", type=str)

        # LR scheduler / early stopping / model checkpoint args ==============================
        self.parser.add_argument("--lr_scheduler_factor", default=0.5, type=float)
        self.parser.add_argument("--lr_scheduler_patience", default=50, type=int) # 25 for 72 hours
        self.parser.add_argument("--monitor_metric", default="valKappa", type=str)
        self.parser.add_argument("--monitor_mode", default="max", type=str)
        self.parser.add_argument("--early_stopping_patience", default=100, type=int) # 55 for 72 hours
        
        # Augmentation args
        self.parser.add_argument("--min_frac", default=0.001, type=float)
        self.parser.add_argument("--max_frac", default=0.3, type=float)
        self.parser.add_argument("--apply_prob", default=0.1, type=float)
        self.parser.add_argument("--sigma", default=1, type=float)
        self.parser.add_argument("--mean", default=0, type=float)
        
        # Result args =========================================================================
        self.parser.add_argument(
            "--result_file_location", 
            #default="./usleep_results" # Not on LUMI
            default="/users/engholma/mnt/usleep_results" # On LUMI
        )
        
        # Datasets =============================================================================
        self.parser.add_argument("--dataset_type", default="tdb", type=str)
        self.parser.add_argument(
            "--datasets_path", 
            default=f"/users/engholma/mnt/transformed/"
            #default=f"/home/alec/repos/data/testing_split_delete_later/" # CFS her
            #default=f"/home/alec/repos/data/hdf5/"
            #default=f"C:/Users/Holme/Desktop/lumi/" # SVUH lokalt
        )
        
        self.parser.add_argument(
            "--split_file", 
            #default=f"/users/engholma/repos/Speciale2023/shared/{self.get_args().array_id}", # On LUMI with array
            default="/users/engholma/repos/Speciale2023/shared/eesm_splits/empty.json", # On LUMI without array
            #default="/home/alec/repos/Speciale2023/shared/usleep_split.json", # On AU cluster
            type=str)
        self.parser.add_argument(
            "--fit_datasets", default=[ 
                #"shhs", "sedf_st", "sedf_sc", "phys", "mros", "mesa", "homepap", "dcsm", "chat", "cfs", "ccshs", "abc", "sof",
                "eesm"
            ]
        )
        self.parser.add_argument(
            "--test_datasets", default=[
                ## NON-HOLD-OUT
                #"shhs", "sedf_st", "sedf_sc", "phys", "mros", "mesa", "homepap", "dcsm", "chat", "cfs", "ccshs", "abc", "sof",
                ## HOLD-OUT 
                #"isruc_sg1", "isruc_sg2", "isruc_sg3", "mass_c1", "mass_c3", "svuh", "dod-h", "dod-o",
                "eesm"
            ]
        )
        # ======================================================================================
        
        
    def get_args(self):
        args = self.parser.parse_args()
        
        return args


    def add_arg(self, name, default):
        self.parser.add_argument(name, default=default)

