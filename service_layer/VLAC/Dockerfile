FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evo_vlac ./evo_vlac
COPY main.py .

ENV PORT=8014 \
    VLAC_MODEL_PATH=/app/models \
    VLAC_MODEL_TYPE=internvl2

EXPOSE 8014

CMD ["python", "main.py"]
