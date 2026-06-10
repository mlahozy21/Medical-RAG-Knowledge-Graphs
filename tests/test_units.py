"""Unit tests for the pure-Python pieces (no GPU, no network, no API keys)."""
import json
import sys

import pytest

sys.path.insert(0, "src")

from kgcolpali.utils import QADataset, locate_answer  # noqa: E402


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
