FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgles2 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mediapipe opencv-python-headless numpy

COPY . /app

ENTRYPOINT ["python", "classify_arm.py"]
