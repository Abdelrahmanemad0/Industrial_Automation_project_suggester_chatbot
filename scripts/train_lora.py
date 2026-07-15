"""LoRA fine-tunes a small instruct model on the industrial-automation
domain dataset (data/training_data.jsonl).

Why this exists / how it fits the rest of the project:
  backend/llm.py loads a base model + this LoRA adapter at inference time.
  backend/engine.py only trusts the LLM's output after validating it against
  the grounded retrieval candidates, so even an imperfectly-trained adapter
  can't make the app hallucinate parts - it can only fail closed to the
  deterministic template fallback.

Run this on a machine with real internet access to huggingface.co (this repo
was authored in a sandbox that could NOT reach the HF Hub, so this script is
untested against real Qwen weights - see README.md's "Training the model"
section, and notebooks/train_lora_colab.ipynb, for a ready-to-run option).

Usage:
    pip install -r requirements-train.txt
    python scripts/generate_training_data.py
    python scripts/train_lora.py

Output:
    model/lora-adapter/  (adapter_config.json + adapter weights)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "training_data.jsonl"
DEFAULT_OUT = ROOT / "model" / "lora-adapter"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_dataset(tokenizer, data_path: Path, max_length: int = 512):
    from datasets import Dataset

    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]

    def to_text(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    texts = [to_text(r) for r in rows]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen2 attention/MLP projection module names.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(tokenizer, Path(args.data), max_length=args.max_length)
    split = dataset.train_test_split(test_size=min(0.1, 10 / max(len(dataset), 1)), seed=7)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(ROOT / "model" / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
    )

    trainer.train()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter to {out_dir}")


if __name__ == "__main__":
    main()
