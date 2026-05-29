import os
import time
import random as _random
import google.generativeai as genai
import torch
from dotenv import load_dotenv

from template import *
# Load the API key from the environment (.env file). Never hardcode keys.
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Create a .env file with GOOGLE_API_KEY=... "
        "(see .env.example)."
    )
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini 2.5 Flash model
gemini = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash-lite",
    generation_config={
        "temperature": 0.2,         # Lower temperature for more focused responses
        "top_p": 0.2,              # Restricts token selection for more concise responses
        "max_output_tokens": 1024,  # Must fit the JSON answer + step_by_step_thinking
    }
)
apitemplates = {
    "cot_system": general_cot_system,
    "cot_prompt": general_cot,
    "medrag_system": general_medrag_system,
    "medrag_prompt": general_medrag,
    "kgcontext_system": general_kg_context_system,
    "kgcontext_prompt": general_kg_context,
}


def generate_with_retry(model, content, max_retries=5, base_delay=2.0):
    """Call the LLM with exponential backoff to survive rate limits / transient errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return model.generate_content(content)
        except Exception as e:  # noqa: BLE001 - we retry on any transient API error
            last_err = e
            delay = base_delay * (2 ** attempt) + _random.uniform(0, 1)
            print(f"Gemini call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                  f"Retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")
