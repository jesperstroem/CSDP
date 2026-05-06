"""Verify that all public modules import cleanly without data files."""

import pytest


def test_import_datastore():
    import csdp_datastore  # noqa: F401


def test_import_datastore_base():
    from csdp_datastore.base import BaseDataset  # noqa: F401


def test_import_datastore_models():
    from csdp_datastore.models import FilterSettings, Labels, Mapping, TTRef  # noqa: F401


def test_import_pipeline():
    import csdp_pipeline  # noqa: F401


def test_import_pipeline_elements():
    from csdp_pipeline.pipeline_elements.pipeline import IPipe, Pipeline, PipelineConfiguration  # noqa: F401
    from csdp_pipeline.pipeline_elements.resampler import Resampler  # noqa: F401
    from csdp_pipeline.pipeline_elements.spectrogram import Spectrogram  # noqa: F401


def test_import_pipeline_models():
    from csdp_pipeline.pipeline_elements.models import ISample, ITag, Split, Dataset_Split  # noqa: F401


def test_import_pipeline_preprocessing():
    from csdp_pipeline.preprocessing.spectrogram import create_spectrogram_images  # noqa: F401


def test_import_training():
    import csdp_training  # noqa: F401


def test_import_training_utility():
    from csdp_training.utility import filter_unknowns, kappa, acc, f1  # noqa: F401


# Tests below require ml_architectures from the private GitLab instance.
# They are skipped automatically in CI where ml_architectures is not available.

def test_import_usleep_lightning():
    ml_arch = pytest.importorskip("ml_architectures", reason="ml_architectures not installed")
    from csdp_training.lightning_models.usleep import USleep_Lightning  # noqa: F401


def test_import_usleep_factory():
    pytest.importorskip("ml_architectures", reason="ml_architectures not installed")
    from csdp_training.lightning_models.factories.lightning_model_factory import USleep_Factory  # noqa: F401
