import os
import json
import re
from kgcolpali.utils import QADataset
import numpy as np

def evaluate(pred_file, split="test"):
    if not os.path.exists(pred_file):
        return 0.0, 0.0, True
    
    predictions = json.load(open(pred_file))
    pred = []
    truth = []
    answer_list = ["A", "B", "C", "D"]
    answer2idx = {ans: i for i, ans in enumerate(answer_list)}
    
    for item in predictions:
        # Extract answer_choice using regex directly from the prediction string
        pred_str = item["prediction"]
        match = re.search(r'"answer_choice":\s*"([A-D])"', pred_str)
        if match:
            predicted_answer = match.group(1)
            if predicted_answer in answer2idx:
                pred.append(answer2idx[predicted_answer])
            else:
                pred.append(-1)
        else:
            pred.append(-1)
        
        # Use ground_truth as the correct answer
        ground_truth = item["ground_truth"]
        if ground_truth in answer2idx:
            truth.append(answer2idx[ground_truth])
        else:
            truth.append(-1)
    
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
