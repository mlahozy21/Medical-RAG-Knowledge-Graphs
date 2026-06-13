# KgColPali — Multimodal Medical RAG with Knowledge Graphs

Internship project (Laboratoire MICS) on **Retrieval-Augmented Generation (RAG)** in the medical domain, combining:

- **Visual document retrieval** with ColPali / BiQwen2.5 (`nomic-ai/nomic-embed-multimodal-3b`),
- a **generative LLM** (Google Gemini) to answer multiple-choice questions,
- and the integration of a medical **Knowledge Graph** (the **Mondo** ontology) to enrich the context.

Evaluation is performed on a subset of the medical **MIRAGE** benchmark. The full report is in `docs/internship_report.pdf` and the slides in `docs/presentation.pdf`. References are in `docs/CITATIONS.md`.

> Note: the report uses three corpora; this repository includes the code for **corpus A** (`vidore/syntheticDocQA_healthcare_industry_test`).

## Operating modes

The pipeline supports four modes, selected via the `KG_MODE` constant in `scripts/run_mirage.py`:

| `kg` | Mode | Description |
|------|------|-------------|
| 1 | LLM only | LLM with chain-of-thought, no retrieval. |
| 2 | RAG only | Visual retrieval of relevant pages + LLM. |
| 3 | RAG + KG context | RAG + textual context extracted from the Knowledge Graph. |
| 4 | RAG + KG retrieve | The KG is also used to retrieve additional documents. |

## Results

Evaluated on a 1,000-question subset of **MIRAGE** (200 questions per dataset, fixed seed 42),
with **Gemini 2.5 Flash-Lite** as the generator backbone. The table reports the best
configuration of each family (mean accuracy); the full per-configuration sweep — score
thresholds, `k`, and the three retrieval corpora — is in Table I of
`docs/internship_report.pdf`.

| Configuration | MMLU | MedQA | MedMCQA | PubMedQA | BioASQ | **Average** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLM only (CoT) | 0.870 | 0.730 | 0.635 | 0.500 | 0.840 | **0.715** |
| RAG + KG context *(best, thr=0.2)* | 0.860 | 0.730 | 0.645 | 0.490 | 0.850 | **0.715** |
| RAG only *(best: Corpus C, k=5, thr=0.25)* | 0.860 | 0.775 | 0.615 | 0.500 | 0.885 | **0.727** |
| **RAG + KG retrieve** *(best, k=3)* | 0.860 | 0.780 | 0.680 | 0.480 | 0.870 | **0.730** |

The full hybrid (**RAG + KG retrieve**) gives the best overall accuracy, **0.730 vs. 0.715**
for the LLM alone. The gain is modest but consistent, and concentrated on the clinical /
exam datasets (MedQA 0.730 → 0.780, MedMCQA 0.635 → 0.680). PubMedQA stays the weak point
across every configuration (~0.48–0.50), suggesting that a literature-oriented corpus
(e.g. PubMed) would be needed there — using the KG to *retrieve* extra evidence helps more
than only injecting KG text as context.

> **Reproducibility notes (honest caveats).**
> - **PubMedQA scoring fix.** PubMedQA gold labels are `yes`/`no`/`maybe`, not the `A`–`D`
>   letters used by the other four datasets. The original `scripts/evaluate.py` only mapped
>   `A`–`D`, so every PubMedQA item collapsed to "unanswered" and that column could not be
>   reproduced from the shipped code. `evaluate.py` now normalizes both letter and
>   `yes/no/maybe` answers (see `normalize_answer` / `parse_prediction`), so the PubMedQA
>   numbers above are reproducible **once predictions are regenerated** with a Gemini key.
>   The PubMedQA values in the table are from the original report; re-running the generator
>   (which needs a GPU + Gemini API key) is required to regenerate them under the fixed scorer.
> - **Corpora.** Only **corpus A** (`vidore/syntheticDocQA_healthcare_industry_test`) ships
>   with this repo. The *RAG only (Corpus C)* row and any other corpus-B/C results require the
>   other two corpora, which are not included here and must be obtained separately.
> - **Sweep knobs.** `k`, `thresholdrag`, and `thresholdkg` are no longer hardcoded; pass them
>   on the CLI (see Usage) to reproduce the per-configuration sweep in principle.


## Project structure

```
.
├── pyproject.toml              # package metadata and dependencies
├── requirements.txt            # pinned dependencies (alternative to pip install -e .)
├── README.md  LICENSE  .env.example  .gitignore
├── src/kgcolpali/              # installable library package
│   ├── api.py                  # Gemini client, prompt registry, retry helper
│   ├── colpali.py              # loads the BiQwen2.5 (ColPali) model and processor
│   ├── embeddings.py           # loads the dataset and precomputed image embeddings
│   ├── functions.py            # RAG logic: retrieval, image prep, `medrag_answer`
│   ├── kg.py                   # Mondo Knowledge Graph: entity linking, SPARQL, context
│   ├── templates.py            # prompt templates (system + user) for each mode
│   └── utils.py                # `QADataset` and answer-parsing utilities
├── scripts/                    # entry points
│   ├── generate_embeddings.py  # build and save the corpus image embeddings
│   ├── run_mirage.py           # generate answers over the MIRAGE subset
│   └── evaluate.py             # compute accuracy of the generated answers
└── docs/
    ├── internship_report.pdf
    ├── presentation.pdf
    └── CITATIONS.md
```

## Prerequisites

- Python 3.10+
- A CUDA GPU is recommended (the BiQwen2.5 model has ~3B parameters). There is a CPU fallback, but it will be slow.
- A **Google AI Studio** (Gemini) API key and a **Hugging Face** token.

## Installation

```bash
git clone https://github.com/mlahozy21/Medical-RAG-Knowledge-Graphs.git
cd Medical-RAG-Knowledge-Graphs

python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

pip install -e .                   # installs the kgcolpali package and its dependencies
```

`pip install -e .` reads the dependencies from `pyproject.toml`. Adjust the `torch` installation to your CUDA version following https://pytorch.org/get-started/locally/.

## Environment variables (keys)

Keys are **not** hardcoded. Create a `.env` file in the project root (it is in `.gitignore`, so it is never committed):

```dotenv
GOOGLE_API_KEY=your_google_ai_studio_key
HF_TOKEN=your_hugging_face_token
```

A template is provided in `.env.example`.

## Required data (not included in the repo)

These resources are heavy and/or external, so they are not versioned. Place them in the **project root** (the scripts resolve paths relative to the current working directory, so run them from the repo root):

1. **Document corpus** — the `vidore/syntheticDocQA_healthcare_industry_test` dataset (Hugging Face), downloaded to `./syntheticDocQA_healthcare_industry_test`.
2. **Image embeddings** — `image_embeddings.pt`. If missing, generate it with `scripts/generate_embeddings.py`. Point `src/kgcolpali/embeddings.py` (`colpali_embeddings_dir`) to its location.