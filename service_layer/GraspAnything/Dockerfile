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

COPY main.py .
COPY grasp_service_nofilter.py .
COPY yolo_service.py .
COPY grasp_detection ./grasp_detection
COPY models ./models
COPY utils ./utils

ENV PORT=8015 \
    DEVICE=auto

EXPOSE 8015

CMD ["python", "main.py"]
