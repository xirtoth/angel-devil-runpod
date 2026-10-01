# Angel/Devil dialogue pod for RunPod.
# LLM dialogue (Ollama + Llama 3.1 8B) + TTS (Coqui XTTS v2, two built-in voices),
# served over FastAPI, auto-registers its public proxy URL with the VPS relay.
#
# Pinned to a CUDA 12.8 build on purpose: RunPod can hand out Blackwell-generation
# GPUs (RTX 50-series / RTX PRO Blackwell), whose sm_120 kernels only exist in
# PyTorch builds compiled against CUDA 12.8+. An older cu121 build loads fine but
# fails at the first actual inference with "CUDA error: no kernel image is
# available for execution on the device".
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

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
#
# RUNPOD_SECRET_TOKEN is NOT set here on purpose -- this image is public, so
# nothing secret belongs baked into it. Set it (and override VPS_UPDATE_URL if
# needed) as environment variables on the RunPod pod template instead, matching
# whatever HALLOWEEN_SECRET_TOKEN the VPS relay is actually running with.
ENV OLLAMA_MODEL=llama3.1:8b \
    POD_PORT=8005 \
    VPS_UPDATE_URL=https://aihenkka.xyz/halloween-api/update-pod \
    OLLAMA_MODELS=/workspace/ollama-models \
    TTS_HOME=/workspace/tts_cache

EXPOSE 8005

ENTRYPOINT ["./entrypoint.sh"]
