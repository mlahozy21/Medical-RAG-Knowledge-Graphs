import os
import faiss
import json
import torch
import tqdm
import numpy as np
from datasets import load_dataset
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
model_name = "nomic-ai/nomic-embed-multimodal-3b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset_images = load_dataset("./syntheticDocQA_healthcare_industry_test", split="test")['image']
    if "cuda" in str(device):
        dtype_to_load = torch.bfloat16
    else:
        dtype_to_load = torch.float32
    print("Loading ColPali model...")    
    colpali = BiQwen2_5.from_pretrained(
        model_name,
        torch_dtype=dtype_to_load,
        device_map=device,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    
    ).eval()
    corpus_name = "Vidore"
    processor = BiQwen2_5_Processor.from_pretrained(model_name)
    db_dir = "./corpus"
    retriever_name = "ColPali/ColPali-embed-multimodal-3b"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    chunk_dir = os.path.join(db_dir, corpus_name, "chunk")
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)
    colpali_embeddings_dir = os.path.join(db_dir, corpus_name, "embeddings", retriever_name.replace("/", "_"))
    # Ensure the embeddings directory exists
    if not os.path.exists(colpali_embeddings_dir):
        os.makedirs(colpali_embeddings_dir, exist_ok=True)
    # Filename for ColPali image embeddings
    colpali_image_embeddings_path = os.path.join(colpali_embeddings_dir, "image_embeddings.pt")

    image_embeddings = None
    if os.path.exists(colpali_image_embeddings_path):
        print(f"Loading pre-calculated image embeddings from: {colpali_image_embeddings_path}")
        image_embeddings = torch.load(colpali_image_embeddings_path).to(device)
    #IF COLPALI EMBEDDINGS DO NOT EXIST, WE GENERATE THEM FROM THE DATASET
    if image_embeddings is None:
        print("Downloading the syntheticDocQA healthcare industry test dataset...")
        batch_size = 1
        all_image_embeddings = []  
        for i in range(0, len(dataset_images), batch_size):
            with torch.no_grad():
                # Select the current batch of images
                batch_images = dataset_images[i : i + batch_size]
                processed_images = processor.process_images(batch_images).to(device)
                print(f"  Processing images {i+1} to {min(i + batch_size, len(dataset_images))}...")

                # Generate embeddings for the current batch
                current_batch_embeddings = colpali(**processed_images)
                print(f"Tensor dimension (using .shape): {current_batch_embeddings.shape}")
                # Store the batch embeddings
                all_image_embeddings.append(current_batch_embeddings)

        image_embeddings = torch.cat(all_image_embeddings, dim=0)
        print(f"The dimension of image_embeddings is: {image_embeddings.shape}")
        # --- Here we save the embeddings! --- 
        torch.save(image_embeddings, colpali_image_embeddings_path)
        print(f"Image embeddings saved to: {colpali_image_embeddings_path}")
