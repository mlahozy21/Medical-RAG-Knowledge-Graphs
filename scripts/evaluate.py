import os
import json
import re
from kgcolpali.utils import QADataset
import numpy as np

# Multiple-choice answer space (mmlu, medqa, medmcqa, bioasq use A-D letters).
LETTER_ANSWERS = ["A", "B", "C", "D"]
LETTER2IDX = {ans: i for i, ans in enumerate(LETTER_ANSWERS)}

# pubmedqa is a yes/no/maybe benchmark rather than a multiple-choice one, so its
# gold answers ("yes"/"no"/"maybe") never map through the A-D table. Give the
# yes/no/maybe labels their own disjoint code space so they are scored correctly.
YESNO_ANSWERS = ["yes", "no", "maybe"]
YESNO2IDX = {ans: 100 + i for i, ans in enumerate(YESNO_ANSWERS)}


def normalize_answer(raw):
    """Map a raw gold/predicted answer to a canonical integer code.

    Handles both multiple-choice letters (A-D, case-insensitive) and the
    yes/no/maybe labels used by pubmedqa. Returns -1 when the answer cannot be
    interpreted (treated as wrong/unanswered).
    """
    if raw is None:
        return -1
    text = str(raw).strip()
    if not text:
        return -1
    upper = text.upper()
    if upper in LETTER2IDX:
        return LETTER2IDX[upper]
    lower = text.lower()
    if lower in YESNO2IDX:
        return YESNO2IDX[lower]
    return -1


def parse_prediction(pred_str):
    """Extract a canonical answer code from a raw model prediction string.

    Looks for an "answer_choice" JSON field first (covering both A-D and
    yes/no/maybe), then falls back to a bare yes/no/maybe token for pubmedqa-style
    free-text answers. Returns -1 when nothing recognizable is found.
    """
    if pred_str is None:
        return -1
    pred_str = str(pred_str)
    # Multiple-choice letter inside an answer_choice field.
    match = re.search(r'"answer_choice":\s*"([A-Da-d])"', pred_str)
    if match:
        return normalize_answer(match.group(1))
    # yes/no/maybe inside an answer_choice field (pubmedqa).
    match = re.search(r'"answer_choice":\s*"(yes|no|maybe)"', pred_str, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))
    # Bare yes/no/maybe token anywhere in a free-text answer (pubmedqa fallback).
    match = re.search(r'\b(yes|no|maybe)\b', pred_str, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))
    return -1


def evaluate(pred_file, split="test"):
    if not os.path.exists(pred_file):
        return 0.0, 0.0, True

    predictions = json.load(open(pred_file))
    pred = []
    truth = []

    for item in predictions:
        pred.append(parse_prediction(item["prediction"]))
        truth.append(normalize_answer(item["ground_truth"]))

    if not pred or not truth or len(pred) != len(truth):
        flag = True
    else:
        flag = False

    acc = (np.array(truth) == np.array(pred)).mean() if pred and truth else 0.0
    std = np.std((np.array(truth) == np.array(pred)).astype(int)) / np.sqrt(len(truth)) if pred and truth else 0.0
    return acc, std, flag


if __name__ == "__main__":
    pred_dir = "./prediction"

    dataset_names = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']
    datasets = {key: QADataset(key) for key in dataset_names}

    scores = []
    for dataset_name in dataset_names:
        print(f"[{dataset_name}] mean acc: ", end="")
        split = "test" if dataset_name != "medmcqa" else "dev"

        pred_file = os.path.join(pred_dir, f"{dataset_name}_predictions.json")

        if os.path.exists(pred_file):
            acc, std, flag = evaluate(pred_file, split)
            scores.append(acc)
            print(f"{acc:.4f}")
        else:
            print("NOT STARTED.")

    if scores:
        print(f"[Average] mean acc: {sum(scores) / len(scores):.4f}")
