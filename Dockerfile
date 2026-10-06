FROM python:3.12-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 sox libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# These wheels include CUDA 13.0 kernels, including Blackwell (RTX 50 series).
# Install matching torchaudio before faster-qwen3-tts resolves dependencies.
RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
COPY requirements-engine.txt constraints-engine.txt ./
RUN pip install --no-cache-dir -r requirements-engine.txt -c constraints-engine.txt \
    && pip check

COPY local_voice ./local_voice
COPY config ./config
EXPOSE 8765
CMD ["python", "-m", "local_voice.http_server"]
