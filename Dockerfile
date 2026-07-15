FROM python:3.11-slim

WORKDIR /app

# Keep image lean: install CPU-only torch explicitly (avoids pulling multi-GB
# CUDA wheels that free HF Spaces CPU hardware can't use anyway).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces (Docker SDK) expects the app on port 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
