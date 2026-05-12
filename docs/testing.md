# CSDP Test Suite

## Overview

The guiding principle for every test: **no data files, no GPU, no network**.
All 123 tests run in ~3 seconds using synthetic tensors, temporary HDF5 files, and pickle files.
Tests are organised into nine files, each covering a distinct layer of the package.
A minimum coverage of **30 %** is enforced in CI (`--cov-fail-under=30`). The current
aggregate sits at ~35 %. The ceiling is intentionally modest because large modules
(`samplers.py`, `mne_sleep_dataset.py`, the Lightning model internals, experiment
runners) require real data files, a GPU, or a training loop to exercise — all of which
are explicitly excluded by the "no data files, no GPU, no network" policy. The threshold
is set to catch deletions of the unit-testable core, not to demand integration coverage.

```
tests/
├── conftest.py                # Shared pytest fixtures (synthetic signal batches)
├── test_imports.py            # Import smoke tests for all public modules
├── test_pipeline_elements.py  # Resampler + Spectrogram pipeline elements
├── test_preprocessing.py      # FilterSettings + create_spectrogram_images
├── test_training_utils.py     # filter_unknowns
├── test_models.py             # Data model classes (Dataset_Split, Split, ISample, ITag)
├── test_usleep_prep_steps.py  # Signal preprocessing functions (filter, clip, scale, resample)
├── test_metrics.py            # kappa, acc, f1, get_majority_vote_predictions
├── test_augmenters.py         # GlobalGaussianNoise, RegionalGaussianNoise, Augmenter
└── test_split_factories.py    # Split factory classmethods + create_split_file
```

Run the full suite:
2
```bash
pytest tests/ -v
# With coverage:
pytest tests/ --cov --cov-report=term-missing
```

---

## conftest.py — Shared Fixtures

Two pytest fixtures produce a `(eegs, eogs, labels, tags)` tuple matching the `IPipe`
contract that every pipeline element expects:

| Fixture | Sample rate | Epochs | Shape (EEG) | Used by |
|---|---|---|---|---|
| `synthetic_batch_128hz` | 128 Hz | 4 | `(2, 15360)` | Resampler tests |
| `synthetic_batch_100hz` | 100 Hz | 3 | `(2, 9000)` | Spectrogram tests |

The 100 Hz fixture matches the Spectrogram element's default FFT sample rate,
so input does not need resampling before the spectrogram step.

---

## test_imports.py — Import Smoke Tests (11 tests)

**What is tested:** Every public class and function can be imported without error.

**Why:** A broken `__init__.py` or a circular import raises `ImportError` at module
load time, before any test logic runs. These tests catch that immediately.

| Test | Module imported |
|---|---|
| `test_import_datastore` | `csdp_datastore` |
| `test_import_datastore_base` | `csdp_datastore.base.BaseDataset` |
| `test_import_datastore_models` | `FilterSettings`, `Labels`, `Mapping`, `TTRef` |
| `test_import_pipeline` | `csdp_pipeline` |
| `test_import_pipeline_elements` | `IPipe`, `Pipeline`, `PipelineConfiguration`, `Resampler`, `Spectrogram` |
| `test_import_pipeline_models` | `ISample`, `ITag`, `Split`, `Dataset_Split` |
| `test_import_pipeline_preprocessing` | `create_spectrogram_images` |
| `test_import_training` | `csdp_training` |
| `test_import_training_utility` | `filter_unknowns`, `kappa`, `acc`, `f1` |
| `test_import_usleep_lightning` | `USleep_Lightning` from `csdp_training.lightning_models.usleep` |
| `test_import_usleep_factory` | `USleep_Factory` from `csdp_training.lightning_models.factories` |

**ml_architectures:** `USleep` and `LSeqSleepNet` depend on
`ml_architectures` (`gitlab.au.dk/tech_ear-eeg/ml_architectures`), a private package
that is not bundled in this repo.

CI installs it automatically before running the test suite:

```bash
pip install git+https://gitlab.au.dk/tech_ear-eeg/ml_architectures.git@main
```

All 11 import tests therefore pass in CI. In a local environment without `ml_architectures`
the two USleep tests are **skipped** rather than failed — `pytest.importorskip("ml_architectures")`
handles this gracefully.

The matching `try/except ImportError` guard in `csdp_training/__init__.py` ensures
`import csdp_training` itself always succeeds, even without `ml_architectures`, so the
rest of the training tests (metrics, utilities) are unaffected.

---

## test_pipeline_elements.py — Resampler + Spectrogram (8 tests)

### TestResampler (5 tests)

`Resampler` normalises incoming data to a target sample rate regardless of the
recording sample rate (128 Hz, 200 Hz, 256 Hz, etc.).

| Test | What it verifies |
|---|---|
| `test_no_resampling_preserves_shape` | 128→128: output shape identical to input |
| `test_downsample_halves_length` | 256→128: sample dimension halved |
| `test_upsample_doubles_length` | 64→128: sample dimension doubled |
| `test_labels_and_tags_pass_through_unchanged` | Labels and metadata tags are not modified |
| `test_channel_count_preserved_after_resampling` | A 4-EEG + 2-EOG batch keeps 4+2 channels |

### TestSpectrogram (3 tests)

`Spectrogram` converts raw time-series into time-frequency images for USleep.
Default parameters: `win_size=2, fs_fourier=100, overlap=1, sample_rate=100`.
Output tensor shape: `(channels, epochs, freq_bins, time_bins)`.

| Test | What it verifies |
|---|---|
| `test_output_channel_and_epoch_dims` | Channel count and epoch count correct in output |
| `test_spectrogram_bins_are_2d` | Output is 4-dimensional `(C, E, F, T)` |
| `test_labels_and_tags_pass_through` | Labels and tags unchanged by spectrogram transform |

---

## test_preprocessing.py — FilterSettings + create_spectrogram_images (10 tests)

### TestFilterSettings (5 tests)

`FilterSettings` (`csdp_datastore/models.py`) is the configuration object for
Butterworth filtering applied during dataset preprocessing. Constructor:
`FilterSettings(lcut, hcut, order)`. The `type` attribute is auto-derived.

| Inputs | Derived `type` |
|---|---|
| `lcut` only | `"highpass"` |
| `hcut` only | `"lowpass"` |
| Both `lcut` and `hcut` | `"bandpass"` |
| Neither (defaults) | `"highpass"` with `lcut=0.3` |

| Test | What it verifies |
|---|---|
| `test_defaults_are_highpass` | Default construction gives `type="highpass"`, `order=2` |
| `test_bandpass_when_both_cutoffs_given` | Both cutoffs → `type="bandpass"`, `cutoffs=[lcut, hcut]` |
| `test_lowpass_when_only_hcut` | `hcut` only → `type="lowpass"` |
| `test_highpass_when_only_lcut` | `lcut` only → `type="highpass"` |
| `test_custom_order` | `order=5` stored correctly |

### TestCreateSpectrogramImages (5 tests)

`create_spectrogram_images` (`csdp_pipeline/preprocessing/spectrogram.py`) is the
low-level numpy function that slices a 1D signal into 30-second epochs and computes
a 2D spectrogram for each epoch.

| Test | What it verifies |
|---|---|
| `test_returns_one_spectrogram_per_epoch` | One spectrogram per complete 30-s epoch |
| `test_single_epoch` | Edge case: exactly one epoch |
| `test_each_spectrogram_is_2d` | Each output is `(freq_bins, time_bins)` |
| `test_all_epochs_same_shape` | All epoch spectrograms have identical shape (required for batching) |
| `test_truncates_to_full_epochs` | Partial trailing epoch is silently dropped |

---

## test_training_utils.py — filter_unknowns (6 tests)

`filter_unknowns` (`csdp_training/utility.py`) removes AASM label 5 (`UNKNOWN`) from
a `(predictions, labels)` pair before computing metrics. If unknown labels were included,
accuracy and F1 would be artificially deflated.

| Test | What it verifies |
|---|---|
| `test_removes_unknown_labels` | Label 5 removed from both tensors |
| `test_no_unknowns_returns_input_unchanged` | No-op when no label 5 present |
| `test_all_unknown_returns_empty` | All-unknown input → empty tensors |
| `test_preserves_relative_order` | Remaining elements keep their original order |
| `test_output_lengths_match` | Predictions and labels always the same length after filtering |
| `test_single_valid_sample` | Edge case: only one non-unknown label |

`test_output_lengths_match` is the most critical: a length mismatch between
predictions and labels would silently produce wrong metric values.

---

## test_models.py — Data Model Classes (12 tests)

### TestDatasetSplit (3 tests)

`Dataset_Split` holds the train/val/test subject lists for one HDF5 file.

| Test | What it verifies |
|---|---|
| `test_get_subjects_train` | `get_subjects_from_string("train")` returns train list |
| `test_get_subjects_val` | Returns val list |
| `test_get_subjects_test` | Returns test list |

### TestSplit (4 tests)

`Split` is a complete cross-validation fold definition — it wraps one or more
`Dataset_Split` objects and a base data path. These are serialised to JSON files
that researchers pass to the training script when launching experiments on the cluster.

| Test | What it verifies |
|---|---|
| `test_get_dict_has_datasets_key` | `get_dict()` output contains `datasets` and `base_data_path` keys |
| `test_get_dict_dataset_has_all_splits` | Each dataset entry has `train`, `val`, `test` keys |
| `test_dump_and_reload_via_file` | JSON round-trip: dump to file, reload, fields preserved exactly |
| `test_repr_contains_id` | `repr(split)` contains the fold ID string |

`test_dump_and_reload_via_file` is the most important: it validates the full
serialisation/deserialisation cycle used in production experiment runs.

### TestISample / TestITag (5 tests)

`ISample` and `ITag` are the core data-sample classes that flow through the
PyTorch `DataLoader` during training and inference.

| Test | What it verifies |
|---|---|
| `test_index_stored` | `ISample(42).index == 42` |
| `test_fields_assignable` | `eeg`, `eog`, `labels`, `tag` fields are assignable |
| `test_default_construction` | `ITag()` initialises empty string fields |
| `test_full_construction` | All `ITag` fields stored correctly |
| *(implicit)* | `ITag.eeg`, `ITag.start_idx`, `ITag.end_idx` accessible |

---

## test_usleep_prep_steps.py — Signal Preprocessing Functions (24 tests)

These are pure numpy/scipy functions in `csdp_pipeline/preprocessing/usleep_prep_steps.py`
applied to raw EEG channels during dataset preprocessing. No data files or GPU needed.

### TestRemoveDC (3 tests)

`remove_dc(data)` subtracts the signal mean to eliminate DC offset.

| Test | What it verifies |
|---|---|
| `test_output_mean_is_zero` | Mean of output is effectively zero |
| `test_zero_mean_input_unchanged` | No-op on an already zero-mean signal |
| `test_shape_preserved` | Output length matches input |

### TestClipChannel (4 tests)

`clip_channel(chnl, min_max_times_global_iqr)` clips extreme values to ±threshold,
where threshold = IQR × multiplier. Prevents outlier artefacts from dominating scaling.

| Test | What it verifies |
|---|---|
| `test_extreme_values_clipped` | A 1e6 spike is reduced to within threshold |
| `test_normal_signal_largely_unchanged` | A normal distribution is not clipped at the default threshold |
| `test_shape_preserved` | Output length matches input |
| `test_custom_threshold` | Custom multiplier produces the expected clip boundary |

### TestClipChannels (3 tests)

`clip_channels(channel_data)` applies `clip_channel` row-by-row to a 2D array.

| Test | What it verifies |
|---|---|
| `test_shape_preserved` | Output shape `(channels, samples)` unchanged |
| `test_does_not_modify_input` | Input array is not mutated (returns a copy) |
| `test_outlier_clipped_in_correct_channel` | Outlier in channel 1 is clipped; channel 0 is untouched |

### TestScaleChannel (4 tests)

`scale_channel(chnl)` applies sklearn's `RobustScaler` — centres by median, scales by IQR.
Used to normalise each channel to a comparable amplitude range before model input.

| Test | What it verifies |
|---|---|
| `test_output_is_1d` | Output remains a 1D array |
| `test_output_length_matches_input` | Length unchanged |
| `test_output_median_near_zero` | Median of output ≈ 0 regardless of input offset |
| `test_output_iqr_near_one` | IQR of output ≈ 1 |

### TestScaleChannelManual (2 tests)

`scale_channel_manual(chnl)` is a vectorised 2D version — subtracts per-channel median
and divides by the global IQR of the array.

| Test | What it verifies |
|---|---|
| `test_shape_preserved` | Output shape matches input |
| `test_per_channel_median_near_zero` | Each row has median ≈ 0 after scaling |

### TestResampleChannel (4 tests)

`resample_channel(channel, output_rate, source_sample_rate)` resamples a 1D array
using `scipy.signal.resample_poly`.

| Test | What it verifies |
|---|---|
| `test_downsample_halves_length` | 200→100 Hz halves the sample count |
| `test_upsample_doubles_length` | 100→200 Hz doubles the sample count |
| `test_identity_resample_preserves_length` | Same rate in and out: length unchanged |
| `test_output_is_1d` | Output dimensionality is 1 |

### TestFilterChannel (4 tests)

`filter_channel(channel, fs, filtersettings)` applies a Butterworth filter defined by
a `FilterSettings` object. The strategy is to construct signals with known spectral content
and verify the correct frequency bands are attenuated or preserved.

| Test | What it verifies |
|---|---|
| `test_lowpass_attenuates_high_frequency` | A 50 Hz component is removed by a 10 Hz lowpass |
| `test_highpass_attenuates_dc_offset` | DC + 20 Hz signal: DC removed by 1 Hz highpass |
| `test_output_shape_preserved` | Output length matches input |
| `test_bandpass_preserves_in_band_signal` | A 10 Hz signal survives a 1–40 Hz bandpass |

---

## test_metrics.py — Training Metrics (19 tests)

All metric functions live in `csdp_training/utility.py`. Each of `kappa`, `acc`, and `f1`
calls `filter_unknowns` internally, so tests with label 5 verify that filtering is applied
before the metric is computed.

### TestKappa (4 tests)

Cohen's Kappa measures inter-rater agreement, correcting for chance. Range: −1 to 1.

| Test | What it verifies |
|---|---|
| `test_perfect_predictions_give_one` | Identical preds and labels → kappa = 1.0 |
| `test_result_is_scalar` | Output is a 0-dimensional tensor |
| `test_unknowns_are_excluded` | Label-5 epochs ignored; perfect non-unknown preds still give 1.0 |
| `test_all_same_class_gives_zero_or_less` | Predicting one class for all inputs → kappa ≤ 0.0 (exact bound) |

### TestAcc (5 tests)

Multiclass accuracy: fraction of epochs with correct prediction.

| Test | What it verifies |
|---|---|
| `test_perfect_predictions_give_one` | Accuracy = 1.0 when all correct |
| `test_all_wrong_gives_zero` | Accuracy = 0.0 when always one class off |
| `test_unknowns_are_excluded` | Label-5 epochs ignored |
| `test_result_is_scalar` | Output is a 0-dimensional tensor |
| `test_half_correct_gives_half` | 50 correct / 100 total → accuracy = 0.5 |

### TestF1 (5 tests)

Macro-averaged F1 score across all 5 AASM sleep stages.

| Test | What it verifies |
|---|---|
| `test_perfect_predictions_give_one` | F1 = 1.0 when all correct |
| `test_result_is_scalar_when_averaged` | `average=True` → scalar output |
| `test_result_is_per_class_when_not_averaged` | `average=False` → shape `(5,)` |
| `test_per_class_scores_are_one_when_perfect` | All per-class F1 scores = 1.0 |
| `test_unknowns_are_excluded` | Label-5 epochs ignored |

### TestGetMajorityVotePredictions (4 tests)

`get_majority_vote_predictions(path)` reads a pickle file containing per-channel
prediction score tensors of shape `(epochs, classes)`, sums the votes across channels,
and returns the winning class per epoch. Tests write a synthetic pickle to pytest's
`tmp_path` fixture, which is cleaned up automatically after each test.

| Test | What it verifies |
|---|---|
| `test_output_length_matches_labels` | `votes` and `labels` tensors have the same length |
| `test_unanimous_predictions_are_correct` | One channel voting perfectly → correct output |
| `test_majority_wins_with_multiple_channels` | 2 channels vote class 0, 1 votes class 1 → class 0 wins |
| `test_returned_labels_match_stored_labels` | Ground-truth labels are returned unchanged |

---

## test_augmenters.py — Augmentation Pipeline Elements (10 tests)

Augmenters live in `csdp_pipeline/pipeline_elements/augmenters.py` and are applied
during training to improve model generalisation. They are not used during inference.

### TestGlobalGaussianNoise (3 tests)

`GlobalGaussianNoise` selects one random channel and replaces it with Gaussian noise.

| Test | What it verifies |
|---|---|
| `test_output_shape_unchanged` | Channel count and length preserved |
| `test_noise_applied_to_one_channel` | At least one channel differs from the zero baseline |
| `test_zero_sigma_leaves_values_as_mean` | `sigma=0` → output equals the `mean` parameter exactly |

### TestRegionalGaussianNoise (3 tests)

`RegionalGaussianNoise` injects noise into a randomly positioned sub-region
of all channels, of length between `min_frac` and `max_frac` of the signal.

| Test | What it verifies |
|---|---|
| `test_output_shape_unchanged` | Shape preserved after augmentation |
| `test_only_region_is_modified` | A zero signal is not all-zero after augmentation |
| `test_invalid_fractions_raise` | `min_frac=0` or `max_frac>1` raises `AssertionError` |

### TestAugmenter (4 tests)

`Augmenter` implements the `IPipe` interface and randomly applies either
`GlobalGaussianNoise` or `RegionalGaussianNoise` each time `process(batch)` is called.

| Test | What it verifies |
|---|---|
| `test_output_shape_preserved` | EEG and EOG output shapes match expected dimensions |
| `test_labels_pass_through_unchanged` | Labels tensor not modified |
| `test_tags_pass_through_unchanged` | Metadata tags not modified |
| `test_zero_prob_leaves_signal_unchanged` | `apply_prob=0.0` → signal values untouched |

---

## test_split_factories.py — Split Factory Methods + create_split_file (25 tests)

These tests use **minimal synthetic HDF5 files** created in pytest's `tmp_path` fixture —
no real PSG recordings needed. The functions only call `.keys()` on HDF5 groups to list
subject identifiers, so an empty group per subject is structurally equivalent to a real file.

### Fixtures

| Fixture | Structure | Used by |
|---|---|---|
| `hdf5_dir` | Two `.hdf5` files, each with a `"data"` group and subject subgroups | `TestFullTest`, `TestRandom` |
| `hdf5_pair` | Two `.hdf5` files (train + test), `"data"` group structure | `TestTrainAndHoldout` |
| `hdf5_flat_dir` | One `.hdf5` file with subjects at the root level (no `"data"` group) | `TestCreateSplitFile` |

The `hdf5_flat_dir` fixture differs because `create_split_file` reads `hdf5.keys()`
at the root level, while the `Split` factory methods read `hdf5["data"].keys()`.

### TestTrainAndHoldout (7 tests)

`Split.train_and_holdout(base, train_file, test_file)` creates a split where one HDF5
file supplies all training subjects and a second supplies all test subjects.

| Test | What it verifies |
|---|---|
| `test_returns_split_object` | Return type is `Split` |
| `test_split_name_stored` | Custom `split_name` stored in `split.id` |
| `test_training_subjects_in_train_list` | All subjects from training file are in `train` list |
| `test_training_dataset_has_empty_val_and_test` | Training `Dataset_Split` has no val or test subjects |
| `test_test_subjects_in_test_list` | All subjects from test file are in `test` list |
| `test_two_dataset_splits_created` | Exactly 2 `Dataset_Split` objects created |
| `test_base_data_path_stored` | `split.base_data_path` matches the provided path |

### TestFullTest (6 tests)

`Split.full_test(base_path)` assigns every subject in every HDF5 file to the test split.
Used when running a trained model against an unseen holdout set.

| Test | What it verifies |
|---|---|
| `test_returns_split_object` | Return type is `Split` |
| `test_all_subjects_in_test_list` | All 16 subjects across both files appear in test lists |
| `test_train_and_val_are_empty` | No subjects assigned to train or val |
| `test_one_dataset_split_per_file` | One `Dataset_Split` created per HDF5 file |
| `test_default_split_name` | Default `id` is `"full_test"` |
| `test_custom_split_name` | Custom name stored correctly |

### TestRandom (6 tests)

`Split.random(base_path)` randomly partitions subjects into train/val/test using
sklearn's `train_test_split`. Default split: 80% / 10% / 10%.

| Test | What it verifies |
|---|---|
| `test_returns_split_object` | Return type is `Split` |
| `test_no_subject_appears_in_two_splits` | Sets are disjoint — no subject in multiple splits |
| `test_all_subjects_accounted_for` | No subjects lost or duplicated |
| `test_train_is_largest_split` | Train count ≥ val and test counts |
| `test_one_dataset_split_per_file` | One `Dataset_Split` per HDF5 file |
| `test_custom_split_name` | Custom name stored correctly |

### TestCreateSplitFile (6 tests)

`create_split_file(hdf5_basepath)` scans a directory of HDF5 files, randomly splits
subjects 80/10/10, and writes the result to `random_split.json` in the current working
directory. Tests use `monkeypatch.chdir` to redirect the output file into `tmp_path`.

| Test | What it verifies |
|---|---|
| `test_returns_filename` | Return value is `"random_split.json"` |
| `test_json_file_created` | File exists after the call |
| `test_json_has_dataset_entry` | Dataset name (derived from filename) is a key in the JSON |
| `test_json_has_train_val_test_keys` | Each dataset entry contains `train`, `val`, `test` |
| `test_all_subjects_distributed` | Union of train+val+test equals the full subject list |
| `test_no_subject_in_two_splits` | Sets are disjoint — no subject in multiple splits |
