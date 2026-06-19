import numpy as np
from scipy import signal
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler


def filter_channel(channel, fs, filtersettings):
    order = filtersettings.order
    cutoffs = filtersettings.cutoffs
    type = filtersettings.type

    sos = signal.butter(order, cutoffs, btype=type, fs=fs, output="sos")
    channel = signal.sosfiltfilt(sos, channel)
    return channel


def clip_channels(channel_data, min_max_times_global_iqr=20):
    """clips each row of the input matrix based on the
    global IQR of that row. assumes median is 0"""
    # https://github.com/perslev/psg-utils/blob/main/psg_utils/preprocessing/quality_control_funcs.py

    clipped = channel_data.copy()
    for i in range(channel_data.shape[0]):
        c = clip_channel(channel_data[i, :], min_max_times_global_iqr=min_max_times_global_iqr)
        clipped[i, :] = c

    return clipped


def clip_channel(chnl, min_max_times_global_iqr=20):
    # https://github.com/perslev/psg-utils/blob/main/psg_utils/preprocessing/quality_control_funcs.py
    iqr = np.subtract(*np.percentile(chnl, [75, 25]))

    threshold = iqr * min_max_times_global_iqr

    clipped = np.clip(chnl, -threshold, threshold)

    return clipped


def scale_channel_manual(chnl):
    medians = np.median(chnl, axis=1)
    iqr = np.subtract(*np.percentile(chnl, [75, 25]))
    scaled = (chnl - medians.reshape(-1, 1)) / iqr
    return scaled


def scale_channel(chnl):
    # https://github.com/perslev/psg-utils/blob/main/psg_utils/preprocessing/scaling.py
    chnl = np.reshape(chnl, (-1, 1))

    assert len(chnl.shape) == 2 and chnl.shape[1] == 1

    transformer = RobustScaler().fit(chnl)

    scaled = transformer.transform(chnl).flatten()

    assert len(scaled.shape) == 1

    return scaled


def resample_channel(channel, output_rate, source_sample_rate, axis=0):
    """
    Function to resample a single data channel to the desired sample rate.
    """

    channel_resampled = resample_poly(channel, output_rate, source_sample_rate, axis=axis)

    return channel_resampled


def remove_dc(data):
    mean = np.mean(data)
    data = np.subtract(data, mean)
    return data
