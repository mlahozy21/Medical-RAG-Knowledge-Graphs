# KgColPali — Multimodal Medical RAG with Knowledge Graphs

Internship project (Laboratoire MICS) on **Retrieval-Augmented Generation (RAG)** in the medical domain, combining:

- **Visual document retrieval** with ColPali / BiQwen2.5 (`nomic-ai/nomic-embed-multimodal-3b`),
- a **generative LLM** (Google Gemini) to answer multiple-choice questions,
- and the integration of a medical **Knowledge Graph** (the **Mondo** ontology) to enrich the context.

Evaluation is performed on a subset of the medical **MIRAGE** benchmark. The full report is in `INTERNSHIP REPORT.pdf` and the slides in `Presentation.pdf`. References are in `CITATIONS.md`.

> Note: the report uses three corpora; this repository includes the code for **corpus A** (`vidore/syntheticDocQA_healthcare_industry_test`).

## Operating modes

The pipeline supports four modes, selected via the `kg` variable in `mirage.py`:

| `kg` | Mode | Description |
|------|------|-------------|
| 1 | LLM only | LLM with chain-of-thought, no retrieval. |
| 2 | RAG only | Visual retrieval of relevant pages + LLM. |
| 3 | RAG + KG context | RAG + textual context extracted from the Knowledge Graph. |
| 4 | RAG + KG retrieve | The KG is also used to retrieve additional documents. |

## Repository structure

| File | Role |
|------|------|
| `mirage.py` | Generates the answers over the MIRAGE subset. **Main entry point.** |
| `evaluate.py` | Computes the accuracy of the answers produced by `mirage.py`. |
| `functions.py` | RAG logic: retrieval, image preparation for Gemini, and `medrag_answer`. |
| `kg.py` | Integration with the Mondo Knowledge Graph (entity extraction, SPARQL, context). |
| `api.py` | Initializes the Gemini client and registers the prompt templates. |
| `colpali.py` | Loads the BiQwen2.5 (ColPali) model and its processor. |
| `template.py` | Prompt templates (system + user) for each mode. |
| `embeddings.py` | Loads the dataset and the precomputed image embeddings. |
| `generate_embeddings.py` | Generates and saves the image embeddings of the corpus. |
| `utilsmirage.py` | `QADataset` and answer-parsing utilities. |

## Prerequisites

- Python 3.10
- A CUDA GPU is recommended (the BiQwen2.5 model has ~3B parameters). There is a CPU fallback in `generate_embeddings.py`, but it will be slow.
- A **Google AI Studio** (Gemini) API key and a **Hugging Face** token.

## Installation

```bash
git clone https://github.com/mlahozy21/KgColPali-LABORATOIRE-MICS-INTERNSHIP-.git
cd KgColPali-LABORATOIRE-MICS-INTERNSHIP-

python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Adjust the `torch` installation to your CUDA version following https://pytorch.org/get-started/locally/.

## Environment variables (keys)

Keys are **not** hardcoded. Create a `.env` file in the project root (it is in `.gitignore`, so it is never committed):

```dotenv
GOOGLE_API_KEY=your_google_ai_studio_key
HF_TOKEN=your_hugging_face_token
```

A template is provided in `.env.example`.

## Required data (not included in the repo)

These resources are heavy and/or external, so they are not versioned. You must obtain them before running:

1. **Document corpus** — the `vidore/syntheticDocQA_healthcare_industry_test` dataset (Hugging Face). Download it locally to `./syntheticDocQA_healthcare_industry_test` (the path expected by `embeddings.py` and `generate_embeddings.py`).
2. **Image embeddings** — `image_embeddings.pt`. If it does not exist, generate it with `generate_embeddings.py` (see below). Point `embeddings.py` (`colpali_embeddings_dir`) to its location.
3. **Mondo ontology** — a `mondo.nt` file (N-Triples) in the project root. Download Mondo from https://mondo.monarchinitiative.org / https://github.com/monarch-initiative/mondo/releases and convert it to N-Triples if needed (e.g. with `robot convert` or `rdflib`). `kg.py` loads it at startup.
4. **MIRAGE benchmark** — a `benchmark.json` file in the project root, available in the official MIRAGE repository (https://github.com/Teddy-XiongGZ/MIRAGE). It is used by `utilsmirage.QADataset`.

> Make sure the exact paths match those defined in `embeddings.py`, `kg.py`, and `utilsmirage.py`.

## Usage

**1. (Optional) Generate the corpus embeddings** if you do not have `image_embeddings.pt`:

```bash
python generate_embeddings.py
```

**2. Generate the predictions** over the MIRAGE subset (200 questions per dataset, fixed seed = 42). Choose the mode by editing the `kg` variable in `mirage.py`:

```bash
python mirage.py
```

Predictions are saved to `prediction/<dataset>_predictions.json`.

**3. Evaluate the accuracy**:

```bash
python evaluate.py
```

It prints the mean accuracy per dataset (`mmlu`, `medqa`, `medmcqa`, `pubmedqa`, `bioasq`) and the overall mean.

## Citations

See `CITATIONS.md` (ColPali, ViDoRe Benchmark V2, and MIRAGE).
