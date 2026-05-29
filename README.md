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
3. **Mondo ontology** — `mondo.nt` (N-Triples). Download Mondo from https://mondo.monarchinitiative.org / https://github.com/monarch-initiative/mondo/releases and convert to N-Triples if needed (e.g. with `robot convert` or `rdflib`).
4. **MIRAGE benchmark** — `benchmark.json`, available in the official MIRAGE repository (https://github.com/Teddy-XiongGZ/MIRAGE), used by `kgcolpali.utils.QADataset`.

## Usage

Run all commands from the repository root.

**1. (Optional) Generate the corpus embeddings** if you do not have `image_embeddings.pt`:

```bash
python scripts/generate_embeddings.py
```

**2. Generate the predictions** over the MIRAGE subset (200 questions per dataset, fixed seed = 42). Choose the mode by editing the `KG_MODE` constant in `scripts/run_mirage.py`:

```bash
python scripts/run_mirage.py
```

Predictions are saved to `prediction/<dataset>_predictions.json`.

**3. Evaluate the accuracy**:

```bash
python scripts/evaluate.py
```

It prints the mean accuracy per dataset (`mmlu`, `medqa`, `medmcqa`, `pubmedqa`, `bioasq`) and the overall mean.

## Citations

See `docs/CITATIONS.md` (ColPali, ViDoRe Benchmark V2, and MIRAGE).

## License

Released under the MIT License — see `LICENSE`.
