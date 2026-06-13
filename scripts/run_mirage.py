import argparse
import json
import os
import random

from tqdm import tqdm

from kgcolpali.functions import medrag_answer
from kgcolpali.utils import QADataset

# Modes: 1. LLM only, 2. RAG only, 3. RAG and KG context, 4. RAG and KG retrieve
KG_MODE = 1
DATASETS = ["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"]
N_SAMPLES = 200
SEED = 42
PRED_DIR = "prediction"
SAVE_EVERY = 10  # checkpoint frequency (in questions)

# Retrieval hyper-parameters. These are the knobs swept per configuration in the
# README; expose them as named defaults (overridable on the CLI) so the sweeps
# are reproducible in principle rather than hardcoded at the call site.
DEFAULT_K = 3                 # number of image snippets retrieved per question
DEFAULT_THRESHOLD_RAG = 0.25  # ColPali score cutoff for visual (RAG) retrieval
DEFAULT_THRESHOLD_KG = 0.2    # ColPali score cutoff for KG-context relevance/retrieval


def save(dataset_name, records):
    os.makedirs(PRED_DIR, exist_ok=True)
    with open(os.path.join(PRED_DIR, f"{dataset_name}_predictions.json"), "w") as f:
        json.dump(records, f, indent=2)


def main(kg=KG_MODE, k=DEFAULT_K, thresholdrag=DEFAULT_THRESHOLD_RAG,
         thresholdkg=DEFAULT_THRESHOLD_KG, datasets=DATASETS, n_samples=N_SAMPLES):
    predictions = {}
    for dataset_name in datasets:
        random.seed(SEED)
        dataset = QADataset(dataset_name)
        selection = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        dataset = [dataset[i] for i in selection]
        records = []

        for i, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
            question = item["question"]
            options = item["options"]
            try:
                answer = medrag_answer(
                    question=question, options=options, kg=kg,
                    k=k, thresholdrag=thresholdrag, thresholdkg=thresholdkg,
                )
            except Exception as e:
                print(f"[{dataset_name}] question {i} failed: {e}")
                answer = ""

            records.append({
                "question": question,
                "prediction": answer,
                "ground_truth": item["answer"],
            })

            if (i + 1) % SAVE_EVERY == 0:
                save(dataset_name, records)

        save(dataset_name, records)
        predictions[dataset_name] = records
    return predictions


def parse_args():
    p = argparse.ArgumentParser(description="Run the MIRAGE evaluation sweep.")
    p.add_argument("--kg", type=int, default=KG_MODE, choices=[1, 2, 3, 4],
                   help="Mode: 1 LLM-only, 2 RAG, 3 RAG+KG context, 4 RAG+KG retrieve.")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="Number of image snippets retrieved per question.")
    p.add_argument("--thresholdrag", type=float, default=DEFAULT_THRESHOLD_RAG,
                   help="ColPali score cutoff for visual (RAG) retrieval.")
    p.add_argument("--thresholdkg", type=float, default=DEFAULT_THRESHOLD_KG,
                   help="ColPali score cutoff for KG-context relevance/retrieval.")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   help="Subset of MIRAGE datasets to run.")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES,
                   help="Number of questions sampled per dataset.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(kg=args.kg, k=args.k, thresholdrag=args.thresholdrag,
         thresholdkg=args.thresholdkg, datasets=args.datasets, n_samples=args.n_samples)
