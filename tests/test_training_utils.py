"""
Tests for pure-function utilities in csdp_training.utility.
These functions operate on torch tensors with no data-file dependencies.
"""

import torch

from csdp_training.utility import filter_unknowns


class TestFilterUnknowns:
    """Label 5 = UNKNOWN in the AASM mapping; must be excluded from all metrics."""

    def test_removes_unknown_labels(self):
        labels = torch.tensor([0, 1, 5, 3, 5, 4])
        preds = torch.tensor([0, 1, 2, 3, 0, 4])
        fp, fl = filter_unknowns(preds, labels)
        assert len(fl) == 4
        assert 5 not in fl.tolist()

    def test_no_unknowns_returns_input_unchanged(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        preds = torch.tensor([4, 3, 2, 1, 0])
        fp, fl = filter_unknowns(preds, labels)
        assert torch.equal(fl, labels)
        assert torch.equal(fp, preds)

    def test_all_unknown_returns_empty(self):
        labels = torch.tensor([5, 5, 5])
        preds = torch.tensor([0, 1, 2])
        fp, fl = filter_unknowns(preds, labels)
        assert len(fl) == 0
        assert len(fp) == 0

    def test_preserves_relative_order(self):
        labels = torch.tensor([0, 5, 2, 5, 4])
        preds = torch.tensor([1, 0, 3, 0, 2])
        fp, fl = filter_unknowns(preds, labels)
        assert fl.tolist() == [0, 2, 4]
        assert fp.tolist() == [1, 3, 2]

    def test_output_lengths_match(self):
        labels = torch.tensor([5, 1, 5, 3])
        preds = torch.tensor([0, 1, 2, 3])
        fp, fl = filter_unknowns(preds, labels)
        assert len(fp) == len(fl)

    def test_single_valid_sample(self):
        labels = torch.tensor([5, 5, 2, 5])
        preds = torch.tensor([0, 0, 2, 0])
        fp, fl = filter_unknowns(preds, labels)
        assert fl.tolist() == [2]
        assert fp.tolist() == [2]
