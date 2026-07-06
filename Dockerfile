FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUSETALK_DIR=/opt/MuseTalk \
    MODEL_DIR=/models/MuseTalk \
    HF_HOME=/tmp/huggingface \
    WORK_DIR=/tmp/puream-lipsync

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/* /usr/share/doc/* /usr/share/man/* /usr/share/locale/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk

COPY fetch_models.py /app/fetch_models.py
RUN rm -rf /root/.cache/huggingface /root/.cache/pip /tmp/* /opt/MuseTalk/.git

COPY app.py /app/app.py
COPY musetalk_runner.py /app/musetalk_runner.py
COPY puream_infer.py /opt/MuseTalk/puream_infer.py

EXPOSE 9000
CMD ["gunicorn", "-b", "0.0.0.0:9000", "--timeout", "1800", "--workers", "1", "app:app"]
