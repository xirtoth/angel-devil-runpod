# Angel/Devil dialogue pod for RunPod.
# LLM dialogue (Ollama + Llama 3.1 8B) + TTS (Coqui XTTS v2, two built-in voices),
# served over FastAPI, auto-registers its public proxy URL with the VPS relay.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git espeak-ng curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ollama (runs the LLM locally inside the pod, talks to the GPU)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pod_app.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# 8005 = the HTTP port RunPod's proxy exposes publicly (matches POD_PORT below)
# OLLAMA_MODELS / TTS_HOME point at /workspace so that if you attach a RunPod
# Network Volume mounted at /workspace, the ~5GB LLM + ~2GB XTTS weights are
# only downloaded once and persist across future pods, instead of re-downloading
# on every cold start.
ENV OLLAMA_MODEL=llama3.1:8b \
    POD_PORT=8005 \
    VPS_UPDATE_URL=https://aihenkka.xyz/halloween-api/update-pod \
    RUNPOD_SECRET_TOKEN=henkkahallo123 \
    OLLAMA_MODELS=/workspace/ollama-models \
    TTS_HOME=/workspace/tts_cache

EXPOSE 8005

ENTRYPOINT ["./entrypoint.sh"]
