FROM python:3.9-slim

WORKDIR /app

# Install dependencies (forcing CPU versions for torch to save space and run on free CPU tier)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Hugging Face Spaces require the app to run on port 7860
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "7860"]
