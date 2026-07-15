---
title: Industrial Automation Project Suggester
emoji: 🏭
colorFrom: orange
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Industrial Automation Project Suggester (Chatbot)

A chat-style assistant that recommends scoped industrial automation / mechatronics
project ideas based on your skill level, the hardware you already have, your
interests, and your time budget.

<p>
  <img alt="Status" src="https://img.shields.io/badge/status-working%20prototype-green">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

## How it works

The app is deliberately built in two layers so it can never give a nonsense
answer, whether or not the language model is loaded:

1. **Grounded retrieval engine** (`backend/retrieval.py`, `data/projects.json`) -
   a curated set of ~28 real industrial automation / mechatronics projects,
   each tagged with skill level, required/optional hardware, category, and
   time estimate. Every request is scored deterministically against this
   dataset (hardware overlap, skill match, time fit, interest match), so the
   shortlist you get back is always a real, buildable project.
2. **Fine-tuned LLM personalization** (`backend/llm.py`, `model/lora-adapter/`) -
   a small model (`Qwen/Qwen2.5-0.5B-Instruct`) LoRA-fine-tuned on a
   domain-specific instruction dataset (`data/training_data.jsonl`, built by
   `scripts/generate_training_data.py`) writes the "why this fits you" text
   and the "what to buy next" suggestion for the top-scored projects.

   Its output is validated (`backend/engine.py::_validate_llm_output`) before
   it's ever shown - if the model returns malformed JSON, or mentions a
   component that isn't in that project's have/missing list, the response is
   rejected and a deterministic templated explanation is used instead. This
   is what makes the app "human proof": you always get a sensible, grounded
   answer, with or without the LLM.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. Without a trained adapter in `model/lora-adapter/`,
the app runs entirely on the grounded template engine - fully functional, just
without LLM-personalized phrasing. Check `/api/health` to see which mode is active.

## Training the model

This repo's LoRA adapter needs to be trained once before the fine-tuned mode
kicks in (it downloads the ~0.5B base model from the Hugging Face Hub, so it
needs a machine with real internet access):

```bash
pip install -r requirements-train.txt
python scripts/generate_training_data.py   # builds data/training_data.jsonl from data/projects.json
python scripts/train_lora.py               # ~10-15 min on a free Colab GPU, longer on CPU
```

This writes `model/lora-adapter/`. Commit that folder and the app will pick
it up automatically on next boot - no code changes needed.

No GPU handy? Open `notebooks/train_lora_colab.ipynb` in Google Colab
(Runtime → GPU), run all cells, download the resulting adapter zip, and drop
it into `model/lora-adapter/` locally before committing.

Why the adapter isn't pre-trained and included in this repo already: it was
built in a sandboxed environment with no network access to huggingface.co,
only to PyPI - so the training script could be written and unit-tested
structurally, but not run against the real base model weights.

## Deploying to Hugging Face Spaces

1. Create a new Space → SDK: **Docker**.
2. Push this repo's contents to the Space's git remote (the `---` block at
   the top of this README configures the Space automatically).
3. The Space builds the Docker image and serves the app on port 7860.
4. To ship the fine-tuned mode, train the adapter (above) and push
   `model/lora-adapter/` to the Space repo as well.

## Project layout

```
app.py                        FastAPI app: serves the frontend + /api/*
backend/
  catalog.py                  Loads + fuzzy-matches hardware names to canonical components
  retrieval.py                Deterministic scoring against data/projects.json
  llm.py                      Loads base model + LoRA adapter, generates personalized text
  engine.py                   Orchestrates retrieval + LLM + validation + fallback
  schemas.py                  Pydantic request/response models
data/
  components.json             Canonical hardware catalog + aliases
  projects.json                Curated project dataset (the "ground truth")
  training_data.jsonl          Generated instruction-tuning data (see scripts/)
static/                       Chat-wizard frontend (HTML/CSS/JS, no framework)
scripts/
  generate_training_data.py   Builds data/training_data.jsonl from data/projects.json
  train_lora.py                LoRA fine-tuning script
notebooks/train_lora_colab.ipynb   One-click Colab training notebook
tests/                        Retrieval + validation unit tests
model/lora-adapter/            Trained adapter goes here (not pre-populated)
```

## Related Projects

For a related chatbot pattern in this profile, see
[Rule-Based-Chat-Bot](https://github.com/Abdelrahmanemad0/Rule-Based-Chat-Bot)
(TF-IDF + Gradio UI).

## License

MIT — see [LICENSE](LICENSE).
