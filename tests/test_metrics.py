"""
Tests for metric functions in csdp_training.utility.

kappa / acc / f1 each call filter_unknowns internally, so tests with label 5
verify that the filtering is applied before metric computation.
"""

import pickle

import pytest
import torch

from csdp_training.utility import acc, f1, get_majority_vote_predictions, kappa


def _perfect(n=100, n_classes=5):
    """Identical predictions and labels, cycling through all 5 classes."""
    labels = torch.arange(n) % n_classes
    return labels.clone(), labels


def _all_wrong(n=100, n_classes=5):
    """Every prediction is one class off (cyclically), so always wrong."""
    labels = torch.arange(n) % n_classes
    preds = (labels + 1) % n_classes
    return preds, labels


def _with_unknowns(n=50, n_classes=5):
    """Half the labels are UNKNOWN (5); the rest are perfect predictions."""
    labels = torch.zeros(n * 2, dtype=torch.long)
    labels[:n] = torch.arange(n) % n_classes
    labels[n:] = 5  # UNKNOWN
    preds = labels.clone()
    preds[n:] = 0  # wrong predictions for unknowns (should be ignored)
    return preds, labels


# ── kappa ─────────────────────────────────────────────────────────────────


class TestKappa:
    def test_perfect_predictions_give_one(self):
        preds, labels = _perfect()
        result = kappa(preds, labels)
        assert abs(result.item() - 1.0) < 1e-4

    def test_result_is_scalar(self):
        preds, labels = _perfect()
        result = kappa(preds, labels)
        assert result.ndim == 0

    def test_unknowns_are_excluded(self):
        preds, labels = _with_unknowns()
        result = kappa(preds, labels)
        # Perfect predictions on the non-unknown half → kappa should be 1
        assert abs(result.item() - 1.0) < 1e-4

    def test_all_same_class_gives_zero_or_less(self):
        # All predictions = 0, but labels vary → kappa ≤ 0
        labels = torch.arange(100) % 5
        preds = torch.zeros(100, dtype=torch.long)
        result = kappa(preds, labels)
        assert result.item() <= 0.0


# ── acc ───────────────────────────────────────────────────────────────────


class TestAcc:
    def test_perfect_predictions_give_one(self):
        preds, labels = _perfect()
        result = acc(preds, labels)
        assert abs(result.item() - 1.0) < 1e-4

    def test_all_wrong_gives_zero(self):
        preds, labels = _all_wrong()
        result = acc(preds, labels)
        assert result.item() == pytest.approx(0.0, abs=1e-4)

    def test_unknowns_are_excluded(self):
        preds, labels = _with_unknowns()
        result = acc(preds, labels)
        assert abs(result.item() - 1.0) < 1e-4

    def test_result_is_scalar(self):
        preds, labels = _perfect()
        assert acc(preds, labels).ndim == 0

    def test_half_correct_gives_half(self):
        labels = torch.zeros(100, dtype=torch.long)
        preds = torch.zeros(100, dtype=torch.long)
        preds[50:] = 1  # wrong for second half
        result = acc(preds, labels)
        assert result.item() == pytest.approx(0.5, abs=1e-4)


# ── f1 ────────────────────────────────────────────────────────────────────


class TestF1:
    def test_perfect_predictions_give_one(self):
        preds, labels = _perfect()
        result = f1(preds, labels)
        assert abs(result.item() - 1.0) < 1e-4

    def test_result_is_scalar_when_averaged(self):
        preds, labels = _perfect()
        result = f1(preds, labels, average=True)
        assert result.ndim == 0

    def test_result_is_per_class_when_not_averaged(self):
        preds, labels = _perfect()
        result = f1(preds, labels, average=False)
        assert result.shape == (5,)

    def test_per_class_scores_are_one_when_perfect(self):
        preds, labels = _perfect()
        result = f1(preds, labels, average=False)
        assert torch.all(result > 0.99)

    def test_unknowns_are_excluded(self):
        preds, labels = _with_unknowns()
        result = f1(preds, labels)
        assert abs(result.item() - 1.0) < 1e-4


# ── get_majority_vote_predictions ──────────────────────────────────────────


def _write_majority_vote_pickle(path, preds_dict, labels):
    """Write a pickle file in the format expected by get_majority_vote_predictions."""
    with open(path, "wb") as f:
        pickle.dump({"preds": preds_dict, "labels": labels}, f)


class TestGetMajorityVotePredictions:
    def test_output_length_matches_labels(self, tmp_path):
        n_epochs, n_classes = 20, 5
        labels = torch.arange(n_epochs) % n_classes
        scores = torch.zeros(n_epochs, n_classes)
        for i in range(n_epochs):
            scores[i, labels[i]] = 1.0
        path = tmp_path / "votes.pkl"
        _write_majority_vote_pickle(path, {"ch0": scores}, labels)
        votes, out_labels = get_majority_vote_predictions(path)
        assert len(votes) == n_epochs
        assert len(out_labels) == n_epochs

    def test_unanimous_predictions_are_correct(self, tmp_path):
        n_epochs, n_classes = 20, 5
        labels = torch.arange(n_epochs) % n_classes
        scores = torch.zeros(n_epochs, n_classes)
        for i in range(n_epochs):
            scores[i, labels[i]] = 1.0
        path = tmp_path / "votes.pkl"
        _write_majority_vote_pickle(path, {"ch0": scores}, labels)
        votes, _ = get_majority_vote_predictions(path)
        assert torch.equal(votes, labels)

    def test_majority_wins_with_multiple_channels(self, tmp_path):
        # 3 channels vote; 2 vote for class 0, 1 votes for class 1 → class 0 wins
        n_epochs, n_classes = 10, 5
        labels = torch.zeros(n_epochs, dtype=torch.long)
        vote_for_0 = torch.zeros(n_epochs, n_classes)
        vote_for_0[:, 0] = 1.0
        vote_for_1 = torch.zeros(n_epochs, n_classes)
        vote_for_1[:, 1] = 1.0
        preds = {"ch0": vote_for_0, "ch1": vote_for_0, "ch2": vote_for_1}
        path = tmp_path / "votes.pkl"
        _write_majority_vote_pickle(path, preds, labels)
        votes, _ = get_majority_vote_predictions(path)
        assert torch.all(votes == 0)

    def test_returned_labels_match_stored_labels(self, tmp_path):
        n_epochs, n_classes = 15, 5
        labels = torch.arange(n_epochs) % n_classes
        scores = torch.ones(n_epochs, n_classes)
        path = tmp_path / "votes.pkl"
        _write_majority_vote_pickle(path, {"ch0": scores}, labels)
        _, out_labels = get_majority_vote_predictions(path)
        assert torch.equal(out_labels, labels)
