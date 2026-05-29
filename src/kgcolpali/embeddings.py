import os
import torch
from datasets import load_dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"

colpali_embeddings_dir ="directory where you have the embeddings"
colpali_image_embeddings_path = os.path.join(colpali_embeddings_dir, "image_embeddings.pt")
dataset = load_dataset("./syntheticDocQA_healthcare_industry_test", split="test")['image']
# Default so that `from embeddings import image_embeddings` never fails
image_embeddings = None
if os.path.exists(colpali_image_embeddings_path):
    image_embeddings= torch.load(colpali_image_embeddings_path).to(device)
    image_embeddings=i