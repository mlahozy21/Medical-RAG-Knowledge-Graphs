# KgColPali Diagnosis — what needs improvement

Analysis of the `KgColPali-LABORATOIRE-MICS-INTERNSHIP-` repository: a multimodal RAG pipeline (ColPali/BiQwen2.5 + Gemini) with a medical Knowledge Graph (Mondo) integration and evaluation on the MIRAGE benchmark.

> Status note: most of the items below have already been addressed in a series of fixes (imports, reproducibility, security, result correctness, architecture, robustness, and cleanup). This document records the original findings and the rationale.

## Summary

The project has an interesting, well-thought-out architecture (four modes: LLM-only, RAG, RAG+KG-context, RAG+KG-retrieve), but **as-is it did not run end to end**: a wrong module name, an import of a non-existent file, essential data files missing, and an LLM configuration that broke the very JSON the pipeline needs to parse. There were also security risks (keys in code) and a lack of reproducibility (no `requirements.txt`, no data instructions).

The issues are listed below by priority.

## 1. Blockers — the code did not start as-is

**1.1. `mirage.py` imported a non-existent module.** Line 4: `from funciones import medrag_answer`, but the file is `functions.py`. Immediate `ModuleNotFoundError`. Fixed to `from functions import medrag_answer`.

**1.2. `embeddings.py` imported from `paginasunicas`, which is not in the repo.** Line 4: `from paginasunicas import split_and_save_pdf, load_pdf_pages`. That module does not exist and, moreover, those two functions were never used in `embeddings.py`. Since `functions.py` does `from embeddings import dataset, image_embeddings`, this broken import took down the whole chain.

**1.3. `image_embeddings` could be left undefined.** In `embeddings.py` it was only assigned inside the `if os.path.exists(...)`. If the `.pt` did not exist, the name was never created and `from embeddings import ... image_embeddings` raised `ImportError`. A default `image_embeddings = None` was missing.

**1.4. Essential data files were missing, with no instructions to obtain them:**

- `mondo.nt` (the Mondo ontology in N-Triples) — `kg.py` loads it at import time and `raise`s on failure, so without it `functions.py` cannot even be imported.
- `benchmark.json` — needed by `utilsmirage.QADataset` and `mirage.py`.
- The `syntheticDocQA_*_test` dataset and the `image_embeddings.pt` embeddings.

The README did not explain where to get any of this. Without these steps nobody (not even you in 6 months) can reproduce the work.

## 2. Configuration that broke the output (high impact on results)

**2.1. `max_output_tokens = 30` in `api.py`.** The system prompts (`template.py`) ask for a JSON with `step_by_step_thinking` (a step-by-step explanation) **and** `answer_choice`. 30 tokens are not enough to even start the reasoning: the response is truncated, the JSON becomes invalid, and `evaluate.py` cannot parse it. This severely degraded the measured accuracy. It must be raised (e.g. 512–1024) or, if you want short answers, change the prompts to ask only for the letter.

**2.2. The parsing of the LLM response was fragile and contradicted the prompt itself.** In `kg.py`, `json_str = response_text.strip()[7:-3]` blindly trimmed a ```` ```json ... ``` ```` block. But the `api.py` system prompt explicitly says "without markdown code fences": if Gemini obeys, the slicing eats valid characters and breaks `json.loads`. `json_repair` was **imported but unused** — exactly the tool for this.

**2.3. The default fallback biased the metric.** `utilsmirage.locate_answer` and `evaluate.py` returned `"A"` when no answer was found. Combined with 2.1/2.2 (many unparseable answers), this artificially inflated the hit rate toward option A. Better to record the failure as "no answer" (-1, already handled) rather than "A".

## 3. Security

**3.1. API keys in source code.** `api.py` (`GOOGLE_API_KEY`) and `colpali.py` (`HF_TOKEN`) assigned the key directly in code. Even as placeholders, the pattern invites committing a real key. They must be read from environment variables / `.env` (with `python-dotenv`) and never written in code.

**3.2. SPARQL built by string interpolation.** In `kg.py`, queries were built with `% noun` using text coming from the LLM. A noun containing quotes breaks the query (and is an injection vector). Use escaping or rdflib `initBindings`.

## 4. Reproducibility and packaging

**4.1. No `requirements.txt` / `environment.yml` / `pyproject.toml`.** The project depends on many heavy, version-sensitive libraries: `torch`, `transformers`, `colpali-engine`, `faiss`, `rdflib`, `pydantic`, `json-repair`, `python-liquid`, `google-generativeai`, `datasets`, `Pillow`, `numpy`, `tqdm`. Without pinned versions it is not reproducible.

**4.2. No `.gitignore`.** Risk of committing `kg_cache.pkl`, `image_embeddings.pt`, `unmatched_nouns.txt`, `prediction/`, etc.

**4.3. Minimal README.** It did not document installation, data acquisition, environment variables, or how to run `mirage.py` / `evaluate.py`, nor what the `kg=1..4` modes and the thresholds mean.

## 5. Import-time side effects (testability)

Importing `api.py` configures Gemini and builds the model; importing `colpali.py` downloads and loads a 3B model onto the GPU; `embeddings.py` loads the dataset and embeddings; `kg.py` parses the whole ontology and the cache. All of this happens **at import**, not when a function is called. Consequences: nothing can be imported without GPU + data + keys, tests are impossible, and `evaluate.py` drags in dependencies it does not need. The right approach is to encapsulate initialization in functions / lazy loading and guard scripts with `if __name__ == "__main__"`.

## 6. Inconsistent device (GPU/CPU) handling

`api.py`, `colpali.py`, `embeddings.py`, and `functions.py` hardcoded `device = "cuda:0"`, with no fallback. Only `generate_embeddings.py` did it right (`torch.cuda.is_available()`). Device selection must be unified.

## 7. Data and naming inconsistencies

**7.1. Inconsistent dataset.** `embeddings.py` and `generate_embeddings.py` loaded `syntheticDocQA_artificial_intelligence_test`, but the README and comments mention the *healthcare* corpus (`syntheticDocQA_healthcare_industry_test`). One had to decide which and make it consistent.

**7.2. Embedding dimensions split across files.** `generate_embeddings.py` saved without `unsqueeze`, and `embeddings.py` did `unsqueeze(1)` on load. The tensor shape depended on which file was run: fragile and prone to shape errors.

**7.3. Mixed languages** (Spanish/French/English) in names and comments: `funciones`, `paginasunicas`, `seuil`. Unify to one language (English is standard for publishing).

## 8. Code cleanup

`functions.py` imported unused modules: `time`, `argparse`, `transformers`, `sys`, `genai_types`, `np`. In `kg.py`, `format_context` used a `for ... else` whose `else` always runs (there is no `break`): the relationships block worked almost by accident; it should be written as a separate loop. `infer_entity_type` always returns `"Disease"` (a stub).

## 9. Production / long-experiment robustness

`mirage.py` called Gemini 5×200 = 1000 times with no retries, backoff, or rate-limit handling; a network glitch could kill the whole run and progress would be lost (saving happened only at the end of each dataset). Use per-question try/except, retries with backoff, and incremental saving. All logging is via `print`; using the `logging` module would help.

---

## Improvement plan (highest to lowest priority)

Worked through one block at a time, validating each before moving on:

1. **Make it run** — fix imports (1.1, 1.2, 1.3), `image_embeddings=None` default, unify the dataset (7.1).
2. **Reproducibility** — `requirements.txt` with versions, `.gitignore`, and a README with installation, data (mondo.nt, benchmark.json, dataset), environment variables, and run commands.
3. **Security** — move keys to `.env` / environment variables (3.1) and parameterize the SPARQL queries (3.2).
4. **Result correctness** — raise `max_output_tokens` (2.1), robust JSON parsing with `json_repair` (2.2), and stop defaulting to "A" (2.3).
5. **Architecture** — unify device handling (6); lazy initialization to remove import-time side effects (5) is documented as a follow-up.
6. **Experiment robustness** — retries/backoff and incremental saving in `mirage.py` (9).
7. **Cleanup** — dead imports, `for ... else`, language (8, 7.3).
