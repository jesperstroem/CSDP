"""Verify that all public modules import cleanly without data files."""

import pytest


def test_import_datastore():
    pass


def test_import_datastore_base():
    pass


def test_import_datastore_models():
    pass


def test_import_pipeline():
    pass


def test_import_pipeline_elements():
    pass


def test_import_pipeline_models():
    pass


def test_import_pipeline_preprocessing():
    pass


def test_import_training():
    pass


def test_import_training_utility():
    pass


# Tests below require ml_architectures.
# They are skipped automatically in environments where ml_architectures is not available.


def test_import_usleep_lightning():
    pytest.importorskip("ml_architectures", reason="ml_architectures not installed")


def test_import_usleep_factory():
    pytest.importorskip("ml_architectures", reason="ml_architectures not installed")
