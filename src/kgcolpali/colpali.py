import os
import torch
from dotenv import load_dotenv

from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
# Load the Hugging Face token from the environment (.env file). Never hardcode tokens.
load_dotenv()
if not os.environ.get("HF_TOKEN"):
    raise RuntimeError(
        "HF_TOKEN is not set. Create a .env file with HF_TOKEN=... (see .env.example)."
    )
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name_colpali = "nomic-ai/nomic-embed-multimodal-3b"
# Initialize the BiQwen model (a version of colpali)
colpalimodel = BiQwen2_5.from_pretrained(
    model_name_colpali,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = BiQwen2_5_Processor.from_pretrained(model_name_colpali, use_fast=True)


