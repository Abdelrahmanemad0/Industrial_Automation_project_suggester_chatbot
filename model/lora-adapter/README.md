Trained LoRA adapter goes here.

Run (on a machine with internet access to huggingface.co):

    pip install -r requirements-train.txt
    python scripts/generate_training_data.py
    python scripts/train_lora.py

...or use notebooks/train_lora_colab.ipynb for a one-click free-GPU option.

That produces adapter_config.json + adapter weight files in this folder.
backend/llm.py auto-detects them on next app start - no code changes needed.
