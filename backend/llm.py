"""Fine-tuned LLM generation layer.

Loads a small base model (default: Qwen2.5-0.5B-Instruct) plus a LoRA
adapter fine-tuned on the industrial-automation project domain
(see scripts/train_lora.py + scripts/generate_training_data.py).

This module is intentionally optional at import time: if torch/transformers/
peft aren't installed, or the adapter hasn't been trained yet, `is_available()`
returns False and backend/engine.py transparently falls back to the
deterministic template engine. That fallback is what makes the app
"human proof" - it never crashes and never hallucinates parts, with or
without the LLM.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "model" / "lora-adapter"

MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = Path(os.environ.get("LORA_ADAPTER_DIR", str(DEFAULT_ADAPTER_DIR)))
MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "160"))

_lock = threading.Lock()
_state = {"loaded": False, "available": False, "tokenizer": None, "model": None, "error": None}


def _adapter_present() -> bool:
    if not ADAPTER_DIR.exists():
        return False
    # a trained PEFT adapter dir contains adapter_config.json + adapter weights
    return (ADAPTER_DIR / "adapter_config.json").exists()


def _try_load():
    """Attempt to load base model + adapter. Safe to call repeatedly; only
    does real work once. Never raises - failures are recorded in _state."""
    with _lock:
        if _state["loaded"]:
            return
        _state["loaded"] = True
        try:
            if not _adapter_present():
                _state["error"] = "no trained LoRA adapter found at %s" % ADAPTER_DIR
                return
            import torch  # noqa: F401
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
            model.eval()

            _state["tokenizer"] = tokenizer
            _state["model"] = model
            _state["available"] = True
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a soft-fail path
            _state["error"] = f"{type(exc).__name__}: {exc}"
            _state["available"] = False


def is_available() -> bool:
    _try_load()
    return _state["available"]


def unavailable_reason() -> Optional[str]:
    _try_load()
    return _state["error"]


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_prompt(user_context: dict, project: dict, have: list[str], missing: list[str]) -> str:
    system = (
        "You are an assistant that explains why one specific pre-selected "
        "industrial automation / mechatronics project fits a student. "
        "Only mention hardware from the 'have' or 'missing' lists given below - "
        "never invent components that aren't listed. "
        "Reply with ONLY a JSON object: {\"rationale\": \"...\", \"buy_next\": \"...\"}. "
        "'rationale' is 2-3 sentences on why this project fits the student's skill level "
        "and interests. 'buy_next' is one short sentence naming the single most useful "
        "missing part to buy next, or \"nothing - you already have what you need\" if the "
        "missing list is empty."
    )
    user = (
        f"Student skill level: {user_context['skill_level']}\n"
        f"Student interests: {user_context['interests'] or 'not specified'}\n"
        f"Student time budget: {user_context['time_budget_hours']} hours\n"
        f"Selected project: {project['title']} ({project['category']}, "
        f"{project['skill_level']} level, ~{project['time_hours']}h)\n"
        f"Project description: {project['description']}\n"
        f"Hardware student already has for this project: {have or 'none'}\n"
        f"Hardware student is missing for this project: {missing or 'none'}\n"
    )
    return system, user


def generate_rationale(user_context: dict, project: dict, have: list[str], missing: list[str]) -> dict:
    """Returns {"rationale": str, "buy_next": str}. Raises ValueError/RuntimeError
    on any failure so the caller (engine.py) can fall back cleanly."""
    if not is_available():
        raise RuntimeError(f"LLM not available: {unavailable_reason()}")

    import torch

    tokenizer = _state["tokenizer"]
    model = _state["model"]
    system, user = _build_prompt(user_context, project, have, missing)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"LLM did not return JSON: {text[:200]!r}")
    data = json.loads(match.group(0))
    if "rationale" not in data or "buy_next" not in data:
        raise ValueError(f"LLM JSON missing required keys: {data}")
    return data
