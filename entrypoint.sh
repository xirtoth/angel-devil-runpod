#!/bin/bash
set -e

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
POD_PORT="${POD_PORT:-8005}"

mkdir -p "${OLLAMA_MODELS:-/workspace/ollama-models}" "${TTS_HOME:-/workspace/tts_cache}" /workspace/halloween_pod/static

echo "[entrypoint] starting ollama server..."
ollama serve &

echo "[entrypoint] waiting for ollama to come up..."
until curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
  sleep 1
done

echo "[entrypoint] pulling model: $OLLAMA_MODEL (skips if already cached in the volume)"
# Container DNS can take a little while to become reachable right after
# startup on some RunPod hosts ("server misbehaving" from the internal
# resolver). Retry instead of letting a transient failure here kill the
# whole entrypoint (set -e would otherwise abort the script and the
# container would restart from scratch in a loop).
until ollama pull "$OLLAMA_MODEL"; do
  echo "[entrypoint] ollama pull failed (DNS/network not ready yet?), retrying in 10s..."
  sleep 10
done

echo "[entrypoint] starting pod_app (loads XTTS, then registers with VPS)..."
exec uvicorn pod_app:app --host 0.0.0.0 --port "$POD_PORT"
