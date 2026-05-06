import pytest
import torch


@pytest.fixture
def synthetic_batch_128hz():
    """4 epochs of 2-channel EEG + 1-channel EOG at 128 Hz. No data files needed."""
    n_epochs, fs = 4, 128
    samples = n_epochs * fs * 30
    eegs = torch.randn(2, samples, dtype=torch.float64)
    eogs = torch.randn(1, samples, dtype=torch.float64)
    labels = torch.zeros(n_epochs, dtype=torch.long)
    tags = [{"dataset": "mock", "subject": "s0", "record": "r0"}]
    return eegs, eogs, labels, tags


@pytest.fixture
def synthetic_batch_100hz():
    """3 epochs of 2-channel EEG + 1-channel EOG at 100 Hz (Spectrogram default)."""
    n_epochs, fs = 3, 100
    samples = n_epochs * fs * 30
    eegs = torch.randn(2, samples, dtype=torch.float64)
    eogs = torch.randn(1, samples, dtype=torch.float64)
    labels = torch.zeros(n_epochs, dtype=torch.long)
    tags = [{"dataset": "mock", "subject": "s0", "record": "r0"}]
    return eegs, eogs, labels, tags
