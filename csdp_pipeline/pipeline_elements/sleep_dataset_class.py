"""data set class for sending sleep data to usleep"""

# pylint: disable=invalid-name

# %% set up dataset

import os
import types

import h5py
import mne
import numpy as np
import pandas as pd
import torch

# %% set up preprocessing functions
from ..preprocessing.usleep_prep_steps import clip_channel, filter_channel, remove_dc, resample_channel, scale_channel


class FilterSettings:
    def __init__(self, lcut=0.1, hcut=None, order=2):
        if lcut is not None and hcut is None:
            type = "highpass"
            self.cutoffs = lcut
        elif hcut is not None and lcut is None:
            type = "lowpass"
            self.cutoffs = hcut
        else:
            type = "bandpass"
            self.cutoffs = [lcut, hcut]

        self.order = order
        self.type = type

    cutoffs: list[float]
    order: int
    type: str


fsets = FilterSettings()

# %%


class sleep_dataset_from_paths(torch.utils.data.Dataset):
    """Creates torch dataset from list of EEG files and scoring files.
    Applies standard preprocessing to match usleep requirements.
    Syntax:
    sleep_dataset_from_paths( EEG_paths,L=1,scoring_paths=[], derivations=None,scoring_preprocess=None,sdcFile=None)

    sdcFile is a previous instance of sleep_dataset_from_paths that has been saved to a sdc file.
    If sdcFile is not None, the dataset is loaded from the sdc file, ignoring the other arguments.
    """

    def __init__(
        self,
        EEG_paths,
        L=1,
        scoring_paths=[],
        ch_names=None,
        derivations=None,
        scoring_preprocess=None,
        sdcFile=None,
        fullRecords=False,
    ):
        if sdcFile is None:
            self.constructFromPaths(EEG_paths, L, scoring_paths, ch_names, derivations, scoring_preprocess, fullRecords)
        else:
            self.construct_from_sdc(sdcFile)

    def constructFromPaths(
        self,
        EEG_paths,
        L,
        scoring_paths=[],
        ch_names=None,
        derivations=None,
        scoring_preprocess=None,
        fullRecords=False,
        passNanEpochs=False,
    ):
        """Standard constructor"""

        self.file_paths = [str(fp) for fp in EEG_paths]
        self.scoring_paths = [str(fp) for fp in scoring_paths]
        self.derivations = derivations
        self.L = L
        self.epochLength = 128 * 30  # 30 seconds
        self.fullRecords = fullRecords
        self.passNanEpochs = passNanEpochs

        if scoring_preprocess is not None:
            self.preprocess_scoring = lambda idx: scoring_preprocess(self, idx)
        else:
            self.preprocess_scoring = self.preprocess_scoring_default

        self.data_arrays = []
        self.nansamples = []
        if len(self.scoring_paths) > 0:
            self.scoring_arrays = []

        for idx, path in enumerate(self.file_paths):
            tempRaw = sleep_dataset_from_paths.open_eeg_file(path)

            if self.derivations is not None:
                # check the derivations, make sure they can be accepted by mne:
                # disallowed derivations are changed to 'None' in the output
                tempDerivs = self.checkDerivations(tempRaw.info, self.derivations)

                data = np.zeros((len(self.derivations), tempRaw.get_data().shape[1]))
                for didx, deriv in enumerate(tempDerivs):
                    if deriv is None:
                        data[didx, :] = np.nan
                    else:
                        if np.isscalar(deriv) == 1:
                            data[didx, :] = tempRaw.get_data(picks=deriv)
                        elif len(deriv) == 2:
                            data[didx, :] = np.nanmean(tempRaw.get_data(picks=deriv[0]), axis=0) - np.nanmean(
                                tempRaw.get_data(picks=deriv[1]), axis=0
                            )
                        else:
                            raise ValueError(
                                "Each derivation should either be single channel ID or a tuple of two lists of channels. \n You passed "
                                + str(deriv)
                            )
            elif ch_names is not None:
                data = tempRaw.get_data(picks=ch_names)
            else:
                data = tempRaw.get_data()

            data = self.preprocess_data(data, sfreq=tempRaw.info["sfreq"])
            self.data_arrays.append(data)

            if len(self.scoring_paths) > 0:
                # appends to scoring_arrays internally:
                self.preprocess_scoring(idx)

                # make sure that the scoring array matches the data array before moving on:
                assert len(self.scoring_arrays[idx]) == self.data_arrays[idx].shape[1] // self.epochLength
            else:
                nSamples = data.shape[1]
                nSamples = (nSamples // self.epochLength) * self.epochLength
                self.data_arrays[idx] = data[:, :nSamples]

        # extract nansamples again:
        # it has to be done after preprocess_scoring, because that might remove some samples
        self.extract_nansamples()

        # create data draws - assumes we will draw L epochs at a time:
        self.create_data_draws()

        # send everything to torch tensors:
        self.data_arrays = [torch.tensor(data, dtype=torch.float32) for data in self.data_arrays]
        if len(self.scoring_paths) > 0:
            self.scoring_arrays = [torch.tensor(scoring, dtype=torch.float32) for scoring in self.scoring_arrays]

    def extract_nansamples(self):
        """Extracts 'nansamples' from the data arrays to bring them back
        to correct size, and keep track of all-nan epochs. If data is not integer
        number of epochs, the trailing samples are ignored."""
        nansamples_list = []
        for idx, data in enumerate(self.data_arrays):
            nDeriv = data.shape[0] // 2
            assert nDeriv * 2 == data.shape[0]
            self.data_arrays[idx] = data[0:nDeriv, :]
            nansamples_list.append(data[nDeriv:, :])

        # determine all-nan epochs based on nanvals:
        self.nanEpochs = []
        for nansamples in nansamples_list:  # loop over all recordings
            nansamples = nansamples > 0  # make sure it's booleans
            nEpochs = nansamples.shape[1] // self.epochLength
            nanEpochs = np.zeros((nansamples.shape[0], nEpochs), dtype=bool)
            for iChannel in range(nansamples.shape[0]):
                nanEpochs[iChannel, :] = np.all(
                    nansamples[iChannel, : (nEpochs * self.epochLength)].reshape(self.epochLength, -1, order="f"),
                    axis=0,
                ).reshape(1, -1)

            self.nanEpochs.append(nanEpochs)

    def checkDerivations(self, raw_info, derivations):
        """Checks that the derivations are valid inputs to mne, and removes those that are not."""

        channel_types = [mne.channel_type(raw_info, i) for i in range(len(raw_info["ch_names"]))]
        allowedInputs = raw_info["ch_names"] + channel_types + np.arange(len(raw_info["ch_names"])).tolist()
        # we could also add 'data' and 'all' to the allowed inputs, but that's not necessary for now.

        outputDerivs = []

        for deriv in derivations:
            # make sure len(deriv) is either 1 or 2:
            if np.isscalar(deriv):
                continue
            if len(deriv) == 2:
                continue
            else:
                raise ValueError(
                    "Each derivation should either be single channel ID or a tuple of two lists of channels. \n You passed "
                    + str(deriv)
                )

        def _checkDerivation(channelDescriptors, allowedInputs):
            """Checks a single derivation, and returns a valid one"""
            wasScalar = False
            if np.isscalar(channelDescriptors) == 1:
                channelDescriptors = [channelDescriptors]
                wasScalar = True
            goodDescriptors = [c for c in channelDescriptors if c in allowedInputs]
            badDescriptors = [c for c in channelDescriptors if c not in allowedInputs]

            if len(badDescriptors) > 0:
                print("Warning, these are not valid channel picks, ignoring them:")
                print(badDescriptors)

            # print warning if no valid channel picks:
            if len(goodDescriptors) == 0:
                print("Warning, no valid channel picks in derivation: \n" + str(channelDescriptors))
                goodDescriptors = [None]

            # assert len(goodDescriptors)>0, 'No valid channel picks in derivation: \n'+str(channelDescriptors)

            if wasScalar:
                return goodDescriptors[0]  # return in the same format as it came
            else:
                return goodDescriptors

        for deriv in derivations:
            if np.isscalar(deriv) == 1:
                outputDerivs.append(_checkDerivation(deriv, allowedInputs))
            elif len(deriv) == 2:
                tempSubDev = []
                for subDev in deriv:
                    tempSubDev.append(_checkDerivation(subDev, allowedInputs))
                outputDerivs.append(tempSubDev)

        return outputDerivs

    def create_data_draws(self):
        """Creates a list of indices for drawing data from the dataset.
        Allows __getitem__ to ignore epochlength and number of files"""
        self.dataDraws = []
        for file_idx, _ in enumerate(self.data_arrays):
            nSamples = self.data_arrays[file_idx].shape[1]
            for i in range(0, nSamples - self.L * self.epochLength, self.epochLength):
                self.dataDraws.append([file_idx, i])

        self.dataDraws = np.array(self.dataDraws)

    def preprocess_data(self, data, sfreq):
        """Preprocess data to be in line with usleep"""

        # nansamples=np.isnan(data)
        # data[nansamples]=0
        # data=resample_channel(data, 128, sfreq)
        # nansamples=resample_channel(nansamples.astype(float), 128, sfreq)>.5
        # data=scale_channel(data)
        # data=clip_channel(data)

        output_data = []
        output_nansamples = []

        for i in range(data.shape[0]):
            channel_data = data[i, :]

            nansamples = np.isnan(channel_data)
            channel_data[nansamples] = 0

            channel_data = remove_dc(channel_data)

            # resampling both data and nansamples:
            channel_data = resample_channel(channel_data, output_rate=128, source_sample_rate=sfreq)
            nansamples = resample_channel(nansamples.astype(float), output_rate=128, source_sample_rate=sfreq) > 0.5

            # standard settings are a 2nd order highpass butterworth filter:
            channel_data = filter_channel(channel_data, 128, fsets)

            channel_data = scale_channel(channel_data)
            channel_data = clip_channel(channel_data)

            output_data.append(channel_data)
            output_nansamples.append(nansamples)

        output_data = np.array(output_data)
        output_nansamples = np.array(output_nansamples)

        # return data with nansamples. nansamples are removed again later:
        return np.vstack((output_data, output_nansamples))

    def cutData(self, idx, start, end):
        """Cuts data to a specific range"""
        self.data_arrays[idx] = self.data_arrays[idx][:, start:end]

    def preprocess_scoring_default(self, idx):
        """Default scoring loading function"""
        scoring = pd.read_csv(self.scoring_paths[idx], sep="/t")

        # cut data array to match scored duration:
        scoringStart = int(scoring.iloc[0, 0] * 128)
        self.data_arrays[idx] = self.data_arrays[idx][:, scoringStart:]

        self.scoring_arrays.append(scoring.iloc[:, 2].values)

    def get_available_channels(filePaths):
        filePaths = [str(fp) for fp in filePaths]
        names = []
        for path in filePaths:
            raw = sleep_dataset_from_paths.open_eeg_file(path, preload=False)
            names.append(raw.ch_names)
        return names

    def open_eeg_file(filename, preload=True):
        """Opens data files. Add more cases if needed."""
        if filename.endswith(".set"):
            try:
                raw = mne.io.read_raw_eeglab(filename, preload=preload, verbose=False)
                return raw
            except Exception as e:
                print("Error reading file " + filename + ": " + str(e))
                return None
        elif filename.endswith(".edf"):
            return mne.io.read_raw_edf(filename, preload=preload, verbose=False)
        elif filename.endswith(".vhdr"):
            return mne.io.read_raw_brainvision(filename, preload=preload, verbose=False)
        else:
            print("Unknown file type for file " + filename)
            raise ValueError("Unknown file type")

    def __len__(self):
        if self.fullRecords:
            return len(self.data_arrays)
        else:
            return self.dataDraws.shape[0]

    def get_epoch_sequence(self, idx):
        """Returns L consecutive epochs. Used for training."""
        fileIdx = self.dataDraws[idx, 0]
        sampleIdx = self.dataDraws[idx, 1]
        epochIdx = sampleIdx // self.epochLength

        x = self.data_arrays[fileIdx][:, sampleIdx : sampleIdx + self.epochLength * self.L]

        assert x.shape[1] == self.epochLength * self.L

        if len(self.scoring_paths) > 0:
            y = self.scoring_arrays[fileIdx][epochIdx : epochIdx + self.L]
            returnList = [x, y, [fileIdx, sampleIdx]]
        else:
            returnList = [x, [fileIdx, sampleIdx]]

        if self.passNanEpochs:
            returnList.append(self.nanEpochs[fileIdx][:, epochIdx : epochIdx + self.L])

        return (*returnList,)

    def get_full_record(self, fileIdx):
        """Returns a full record of data. Used for validation and testing."""

        x = self.data_arrays[fileIdx]

        if len(self.scoring_paths) > 0:
            y = self.scoring_arrays[fileIdx]
            returnList = [x, y, fileIdx]
        else:
            returnList = [x, fileIdx]

        if self.passNanEpochs:
            returnList.append(self.nanEpochs[fileIdx])

        return (*returnList,)

    def __getitem__(self, idx):
        if self.fullRecords:
            return self.get_full_record(idx)
        else:
            return self.get_epoch_sequence(idx)

    def save_to_sdc(self, sdcFile):
        """
        Save the dataset to an sdc file.
        Adds '.sdc' to the end of the filename if it is not already there.

        """
        # ensure that the file ends with .sdc:
        if not sdcFile.endswith(".sdc"):
            sdcFile += ".sdc"

        # it turns out to be convenvient to save the sizes of the data arrays as well:
        self.data_sizes = [data.shape for data in self.data_arrays]

        with h5py.File(sdcFile, "w") as f:
            for key, value in self.__dict__.items():
                if isinstance(value, list):
                    for idx, data in enumerate(value):
                        f.create_dataset(key + "/" + str(idx), data=data)
                elif isinstance(value, int):
                    f.create_dataset(key, data=value, shape=(1,))
                elif isinstance(value, range):
                    temp = np.asarray(value)
                    f.create_dataset(key, data=temp, shape=temp.shape)
                elif value is None:
                    f.create_dataset(key, data=value, shape=(0,))
                elif isinstance(value, (types.FunctionType, types.MethodType)):
                    pass
                else:
                    f.create_dataset(key, data=value, shape=value.shape)

    def construct_from_sdc(self, sdcFile):
        """If the constructor is fed an sdc-file"""

        # ensure that the file ends with .sdc:
        if not sdcFile.endswith(".sdc"):
            sdcFile += ".sdc"

        # An sdc-file is just an hdf5-file in disguise, to avoid confusing with hdf5-files from other sources

        # load all keys from hdf5 file into dataset:
        with h5py.File(sdcFile, "r") as f:
            for key, value in f.items():
                if isinstance(value, h5py.Group):
                    self.__dict__[key] = []
                    for subkey in value.keys():
                        self.__dict__[key].append(f[key][subkey][...])
                else:
                    if value.shape[0] == 1:
                        self.__dict__[key] = value[0]
                    else:
                        self.__dict__[key] = value[...]

        if not hasattr(self, "scoring_paths"):
            self.scoring_paths = []

        # convert to torch tensors:
        self.data_arrays = [torch.tensor(data, dtype=torch.float32) for data in self.data_arrays]
        if hasattr(self, "scoring_arrays"):
            self.scoring_arrays = [torch.tensor(scoring, dtype=torch.float32) for scoring in self.scoring_arrays]

        # check that data array sizes are still correct:
        for idx, data in enumerate(self.data_arrays):
            assert (data.shape == self.data_sizes[idx]).all()


# %% testing
if __name__ == "__main__":
    import mne_bids as mb

    # get a list of recording files with corresponding scoring files:
    ROOT = "path/to/data/root"
    derivations = os.path.join(ROOT, "derivatives")

    CLEAN_DERIV_PATH = os.path.join(derivations, "cleaned_2")
    filePaths = mb.find_matching_paths(CLEAN_DERIV_PATH, extensions=".set", tasks="sleep", subjects="005")[:1]

    # make a dataset from the list of files, let the derivation be difference between average left and right ear electrodes:
    dataset = sleep_dataset_from_paths(
        filePaths,
        derivations=[(["EL1", "EL2", "EL3", "EL4", "EL5"], ["ER1", "ER2", "ER3", "ER4", "ER5"])],
        fullRecords=True,
    )

# %%
