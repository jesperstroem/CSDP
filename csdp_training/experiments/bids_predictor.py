import mne_bids as mb
import torch

from csdp_pipeline.pipeline_elements.sleep_dataset_class import sleep_dataset_from_paths

from ..lightning_models.factories.lightning_model_factory import USleep_Factory, USleep_Lightning


class BIDS_USleep_Predictor:
    def __init__(self, data_dir, data_extension, data_task, subjects=None, sessions=None):
        """Helper class that can sleep-stage a full BIDS dataset with a pre-trained version of U-Sleep

        Args:
            data_dir (str): Root directory of the BIDS dataset
            data_extension (str): Extension of the datafiles. Valid types at the moment are .edf, .set, .vhdr
            data_task (str): Task of the BIDS files - "...task-<data task>" - most often it is "sleep"
            subjects (list(str), optional): List of subjects, e.g. ["sub-001", "sub-002"]. Defaults to None which means it will sleep-stage every subject
            sessions (list(str), optional): List of sessions, e.g. ["ses-001"]. Defaults to None, which means it will sleepstage every session.
        """
        assert (data_extension == ".vhdr") or (data_extension == ".set") or (data_extension == ".edf")

        self.data_dir = data_dir
        self.data_extension = data_extension
        self.data_task = data_task
        self.subjects = subjects
        self.sessions = sessions

        self.filepaths = mb.find_matching_paths(
            data_dir, extensions=data_extension, tasks=data_task, subjects=subjects, sessions=sessions
        )

    def get_available_channels(self):
        """Helper function to obtain channel information from the BIDS dataset

        Returns:
            set: Names of the available channels
        """
        ch_names = sleep_dataset_from_paths.get_available_channels(self.filepaths)

        # if overlapping==True:
        return set.intersection(*[set(x) for x in ch_names])
        ##else:
        #   return ch_names

    def predict_all(self, lighting_checkpoint, dataset, eeg_indexes=None, eog_indexes=None):
        """Function to predict on a dataset object, given a U-Sleep checkpoint

        Args:
            lighting_checkpoint (str): Path to a U-Sleep Lightning checkpoint
            dataset (Torch Dataset): A Torch dataset, obtained with the function "build_dataset"
            eeg_indexes (list(int), optional): Indexes for which channels in the dataset are eeg. Can't be None if you have specified a two-channel model. Defaults to None.
            eog_indexes (list(int), optional): Indexes for which channels in the dataset are eog. Can't be None if you have specified a two-channel model. Defaults to None.

        Returns:
            list(torch.Tensor): A list of sleep-stage predictions, one for each entry in the BIDS dataset
        """
        fac = USleep_Factory(lr=0.0001, batch_size=64)
        usleep: USleep_Lightning = fac.create_pretrained_net(lighting_checkpoint)
        num_channels = usleep.num_channels

        if num_channels == 2:
            if (eeg_indexes is None) or (eog_indexes is None):
                raise ValueError("For a model with 2 channels EEG and EOG indexes must be specified")

        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        all_preds = []
        all_confidences = []

        for _, (x, _) in enumerate(dataLoader):
            if num_channels == 1:
                output = usleep.majority_vote_prediction(x_eegs=x)
            elif num_channels == 2:
                eegs = torch.index_select(x, dim=1, index=torch.tensor(eeg_indexes))
                eogs = torch.index_select(x, dim=1, index=torch.tensor(eeg_indexes))

                output = usleep.majority_vote_prediction(x_eegs=eegs, x_eogs=eogs)
            else:
                raise ValueError("Unknown number of channels for U-Sleep")

            num_epochs = output[list(output.keys())[0]].shape[2]
            num_classes = 5

            votes = torch.zeros(num_classes, num_epochs)  # fordi vi summerer løbende

            num_channels = len(list(output.items()))

            for item in output.items():
                pred = item[1]
                votes = torch.add(votes, pred)

            votes = torch.squeeze(votes)
            votes = votes / len(output.items())
            all_confidences.append(votes)
            preds = torch.argmax(votes, axis=0)
            all_preds.append(preds)

        return all_preds, all_confidences

    def build_dataset(self, channel_names: list):
        """Builds a Dataset object from a list of channel names

        Args:
            channel_names (list(str)): List with channel names. Avaiable channel names can be obtained from the function "get_available_channels"

        Returns:
            Torch Dataset: A Torch dataset containing the given channel names
        """
        data_set = sleep_dataset_from_paths(
            self.filepaths, scoring_paths=[], ch_names=channel_names, scoring_preprocess=None, fullRecords=True
        )
        return data_set
