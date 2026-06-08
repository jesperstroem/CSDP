import numpy as np
import os
import matplotlib.pyplot as plt

from csdp_pipeline.pipeline_elements.sleep_dataset_class import sleep_dataset_from_paths
from csdp_pipeline.pipeline_elements.plot_hypnogram import plotHypnoGram
from csdp_training.lightning_models.usleep import USleep_Lightning
import torch


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str, device: str = 'cpu'):
    usleep_pretrained = USleep_Lightning.load_from_checkpoint(checkpoint_path)
    usleep_pretrained.eval()
    usleep_pretrained.to(device)
    return usleep_pretrained


def _translate_labels(input_labels: list, input_order: list, output_order: list):
    assert all([x in output_order for x in input_order])
    input_labels = np.array(input_labels)
    output_labels = np.zeros_like(input_labels) - 1
    assert np.max(input_labels) < len(input_order)
    for class_idx, class_name in enumerate(input_order):
        epoch_indexes = input_labels == class_idx
        output_labels[epoch_indexes] = output_order.index(class_name)
    return output_labels


def _generate_derivations_list(eeg_inputs: list = None, eog_inputs: list = None):
    """Given lists of EEG and EOG inputs, generate list of derivation (EEG, EOG) pairs."""
    if eeg_inputs is not None and len(eeg_inputs) == 1:
        raise ValueError("EEG inputs must have at least 2 elements to make derivations.")
    if eog_inputs is not None and len(eog_inputs) == 1:
        raise ValueError("EOG inputs must have at least 2 elements to make derivations.")

    if eeg_inputs is None or len(eeg_inputs) == 0:
        eeg_inputs = [None, None]
    if eog_inputs is None or len(eog_inputs) == 0:
        eog_inputs = [None, None]

    eeg_derivations = []
    for i in range(len(eeg_inputs)):
        for j in range(i + 1, len(eeg_inputs)):
            eeg_derivations.append((eeg_inputs[i], eeg_inputs[j]))

    eog_derivations = []
    for i in range(len(eog_inputs)):
        for j in range(i + 1, len(eog_inputs)):
            eog_derivations.append((eog_inputs[i], eog_inputs[j]))

    derivations = []
    for eeg_deriv in eeg_derivations:
        for eog_deriv in eog_derivations:
            derivations.append((eeg_deriv, eog_deriv))
    return derivations


def _make_derivation(dataset, derivation_pair):
    if derivation_pair[0] is None or derivation_pair[1] is None:
        return None
    return dataset[0][0][derivation_pair[0], :] - dataset[0][0][derivation_pair[1], :]


def _predict_single_pair(model, eeg_signal=None, eog_signal=None):
    """One forward pass for one EEG + one EOG array. Returns softmax (nClasses, nEpochs)."""
    if eeg_signal is None:
        input_signal = eog_signal.reshape(1, 1, -1)
    elif eog_signal is None:
        input_signal = eeg_signal.reshape(1, 1, -1)
    else:
        input_signal = torch.stack([eeg_signal, eog_signal], axis=0).reshape(1, 2, -1)
    output = model(input_signal).detach().numpy()  # (1, nClasses, nEpochs)
    return output[0, :, :]  # (nClasses, nEpochs)


def _predict_all_pairs(model, dataset, eeg_indices, eog_indices):
    """Run _predict_single_pair for every EEG×EOG pair, return averaged softmax (nClasses, nEpochs)."""
    outputs = []
    derivations_list = _generate_derivations_list(eeg_inputs=eeg_indices, eog_inputs=eog_indices)
    for derivation in derivations_list:
        eeg_signal = _make_derivation(dataset, derivation[0])
        eog_signal = _make_derivation(dataset, derivation[1])
        output = _predict_single_pair(model, eeg_signal=eeg_signal, eog_signal=eog_signal)
        outputs.append(output)
    outputs = np.array(outputs)        # (nPairs, nClasses, nEpochs)
    avg_outputs = np.mean(outputs, axis=0)   # (nClasses, nEpochs)
    return avg_outputs


def _plot_hypnogram(labels: np.ndarray) -> plt.Figure:
    """Return a matplotlib Figure of the hypnogram for the given label sequence."""
    stage_order = ["W", "R", "N1", "N2", "N3"]
    label_to_y = {i: -i for i in range(len(stage_order))}

    y = np.array([label_to_y.get(int(l), np.nan) if not np.isnan(l) else np.nan for l in labels])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(np.arange(len(y)), y, where='post', color='black', linewidth=0.8)
    ax.set_yticks(list(label_to_y.values()))
    ax.set_yticklabels(stage_order)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Stage")
    ax.set_title("Hypnogram")
    fig.tight_layout()
    return fig


# ── Public API ─────────────────────────────────────────────────────────────────

def score_file(input_path: str,
               eeg_inputs: list = None,
               eog_inputs: list = None,
               checkpoint: str = None,
               output_path: str = None,
               device: str = 'cpu'):
    """Score a single BIDS-compatible EDF file using a pretrained U-Sleep checkpoint.

    Args:
        input_path: Path to the EDF file to score.
        eeg_inputs: List of EEG channel names (need ≥2 to form bipolar derivations).
        eog_inputs: List of EOG channel names (need ≥2 to form bipolar derivations).
        checkpoint: Path to a USleep_Lightning .ckpt checkpoint file.
        output_path: If given, saves labels + softmax as .npz and a hypnogram PNG
                     next to it. If None, returns results without writing files.
        device: Torch device string, e.g. 'cpu' or 'cuda'.

    Returns:
        labels (np.ndarray): Predicted stage per epoch, NaN where all channels are NaN.
        epochsUsed (np.ndarray): Indices of non-NaN epochs.
        avg_outputs (np.ndarray): Averaged softmax probabilities, shape (5, nEpochs).
    """
    all_inputs = eeg_inputs + eog_inputs
    dataset = sleep_dataset_from_paths([input_path], ch_names=all_inputs, L=35)
    dataset.fullRecords = True

    eeg_indices = np.arange(len(eeg_inputs))
    eog_indices = np.arange(len(eog_inputs)) + len(eeg_inputs)

    assert dataset[0][0].shape[0] == len(all_inputs)

    usleep_pretrained = _load_model(checkpoint, device=device)

    avg_outputs = _predict_all_pairs(usleep_pretrained, dataset, eeg_indices, eog_indices)

    labels = np.argmax(avg_outputs, axis=0)  # (nEpochs,) — avg_outputs is (nClasses, nEpochs)

    labels = _translate_labels(labels,
                                input_order=['W', 'N1', 'N2', 'N3', 'R'],
                                output_order=['W', 'R', 'N1', 'N2', 'N3'])
    labels = labels.astype(float)

    nan_epochs = np.nonzero(np.all(dataset.nanEpochs[0], axis=0))[0]
    labels[nan_epochs] = np.nan

    epochsUsed = np.arange(len(labels))
    epochsUsed = epochsUsed[~np.isnan(labels)]

    if output_path is None:
        return labels, epochsUsed, avg_outputs

    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    np.savez(output_path, labels=labels, avg_outputs=avg_outputs)

    filename = os.path.splitext(os.path.basename(output_path))[0]
    png_path = os.path.join(parent_dir, filename + '.png')

    fig = _plot_hypnogram(labels)
    fig.savefig(str(png_path), dpi=300)
    plt.close(fig)

    return labels, epochsUsed, avg_outputs
