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


def save(dataset_name, records):
    os.makedirs(PRED_DIR, exist_ok=True)
    with open(os.path.join(PRED_DIR, f"{dataset_name}_predictions.json"), "w") as f:
        json.dump(records, f, indent=2)


def main(kg=KG_MODE):
    predictions = {}
    for dataset_name in DATASETS:
        # Fix a seed so the selection is always the same
        random.seed(SEED)
        dataset = QADataset(dataset_name)
        selection = random.sample(range(len(dataset)), N_SAMPLES)
        dataset = [dataset[i] for i in selection]
        records = []

        for i, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
            question = item["question"]
            options = item["options"]
            try:
                answer = medrag_answer(
                    question=question, options=options, kg=kg,
                    k=3, thresholdrag=0.25, thresholdkg=0.2,
                )
            except Exception as e:  # noqa: BLE001 - keep the run alive on a single failure
                print(f"[{dataset_name}] question {i} failed: {e}")
                answer = ""  # empty -> the evaluator counts it as unanswered

            records.append({
                "question": question,
                "prediction": answer,
                "ground_truth": item["answer"],
            })

            # Incremental checkpoint so progress is not lost if the run crashes
            if (i + 1) % SAVE_EVERY == 0:
                save(dataset_name, records)

        save(dataset_name, records)  # final flush for this dataset
        predictions[dataset_name] = records
    return predictions


if __name__ == "__main__":
    main()
