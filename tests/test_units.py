"""Unit tests for the pure-Python pieces (no GPU, no network, no API keys)."""
import json
import os
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from kgcolpali.utils import QADataset, locate_answer  # noqa: E402
from evaluate import evaluate, normalize_answer, parse_prediction  # noqa: E402


@pytest.mark.parametrize("sentence,expected", [
    ("A", "A"),
    ("  B", "B"),
    ("C, because the lesion is benign", "C"),
    ("A or B", "A"),
    ("D and the others are wrong", "D"),
    ("Option B is correct", "B"),
    ("option c", None),            # lowercase body is not a valid option letter
    ("The answer is: C", "C"),
    ("B. The patient shows...", "B"),
    ("I am not sure about this one", None),
])
def test_locate_answer(sentence, expected):
    assert locate_answer(sentence) == expected


def test_qadataset_loads_benchmark(tmp_path):
    benchmark = {"mmlu": {"q0": {"question": "Q?", "options": {"A": "x"}, "answer": "A"},
                          "q1": {"question": "Q2?", "options": {"A": "y"}, "answer": "A"}}}
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark))
    ds = QADataset("mmlu_med", dir=str(tmp_path))
    assert len(ds) == 2
    assert ds[0]["question"] == "Q?"
    assert [d["question"] for d in ds[0:2]] == ["Q?", "Q2?"]


def test_qadataset_rejects_unknown_benchmark(tmp_path):
    (tmp_path / "benchmark.json").write_text(json.dumps({"mmlu": {}}))
    with pytest.raises(KeyError):
        QADataset("unknown_dataset", dir=str(tmp_path))


# --- evaluator: letters AND yes/no/maybe ------------------------------------

@pytest.mark.parametrize("raw,expected_code", [
    ("A", 0), ("d", 3), ("  C ", 2),
    ("yes", 100), ("No", 101), ("MAYBE", 102),
    ("", -1), (None, -1), ("garbage", -1),
])
def test_normalize_answer(raw, expected_code):
    assert normalize_answer(raw) == expected_code


def test_parse_prediction_letter_and_yesno():
    assert parse_prediction('{"answer_choice": "B"}') == normalize_answer("B")
    assert parse_prediction('{"answer_choice": "yes"}') == normalize_answer("yes")
    # free-text pubmedqa-style answer with a bare yes/no/maybe token
    assert parse_prediction("Based on the abstract, the answer is no.") == normalize_answer("no")
    assert parse_prediction("I cannot tell") == -1


def test_evaluate_pubmedqa_record(tmp_path):
    """A pubmedqa-style record (gold = yes/no/maybe) must be scored correctly,
    not collapse to -1 the way the old A-D-only mapping did."""
    records = [
        {"question": "q1", "prediction": '{"answer_choice": "yes"}', "ground_truth": "yes"},
        {"question": "q2", "prediction": "The evidence suggests no.", "ground_truth": "no"},
        {"question": "q3", "prediction": '{"answer_choice": "yes"}', "ground_truth": "maybe"},
    ]
    pred_file = os.path.join(str(tmp_path), "pubmedqa_predictions.json")
    with open(pred_file, "w") as f:
        json.dump(records, f)
    acc, std, flag = evaluate(pred_file)
    assert flag is False
    assert acc == pytest.approx(2.0 / 3.0)


def test_evaluate_letter_record(tmp_path):
    records = [
        {"question": "q1", "prediction": '{"answer_choice": "A"}', "ground_truth": "A"},
        {"question": "q2", "prediction": '{"answer_choice": "B"}', "ground_truth": "C"},
    ]
    pred_file = os.path.join(str(tmp_path), "mmlu_predictions.json")
    with open(pred_file, "w") as f:
        json.dump(records, f)
    acc, _, flag = evaluate(pred_file)
    assert flag is False
    assert acc == pytest.approx(0.5)


# --- functions.select_from_scores: index mapping ----------------------------

try:
    import torch
    from kgcolpali.functions import select_from_scores  # noqa: E402
    _HAVE_SELECT = True
except Exception:  # pragma: no cover - heavy deps (torch/colpali) unavailable
    torch = None
    _HAVE_SELECT = False


@pytest.mark.skipif(not _HAVE_SELECT, reason="functions.py heavy deps unavailable")
def test_select_from_scores_no_threshold():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5])
    top_scores, top_idx = select_from_scores(scores, threshold=0.0, k=2)
    assert top_idx.tolist() == [1, 3]
    assert top_scores.tolist() == pytest.approx([0.9, 0.7])


@pytest.mark.skipif(not _HAVE_SELECT, reason="functions.py heavy deps unavailable")
def test_select_from_scores_threshold_maps_indices():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.6])
    top_scores, top_idx = select_from_scores(scores, threshold=0.5, k=2)
    assert top_idx.tolist() == [1, 3]
    assert top_scores.tolist() == pytest.approx([0.9, 0.7])


@pytest.mark.skipif(not _HAVE_SELECT, reason="functions.py heavy deps unavailable")
def test_select_from_scores_threshold_empty():
    scores = torch.tensor([0.1, 0.2, 0.3])
    top_scores, top_idx = select_from_scores(scores, threshold=0.9, k=2)
    assert top_idx.numel() == 0
