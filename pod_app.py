"""
Runs inside the RunPod container. Generates an ongoing Angel vs Devil
conversation using a local LLM (via Ollama) and synthesizes each line with
XTTS v2 (two distinct built-in speakers). Registers this pod's public proxy
URL with the VPS relay on startup and every few minutes after that, so the
ESP32 (which only ever talks to the VPS) always has a live backend.
"""
import os
import threading
import time

import ollama
import requests
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse

from TTS.api import TTS

STATIC_DIR = "/workspace/halloween_pod/static"
os.makedirs(STATIC_DIR, exist_ok=True)

VPS_UPDATE_URL = os.environ.get("VPS_UPDATE_URL", "https://aihenkka.xyz/halloween-api/update-pod")
SECRET_TOKEN = os.environ.get("RUNPOD_SECRET_TOKEN", "henkkahallo123")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
PORT = int(os.environ.get("POD_PORT", "8005"))

DEVIL_SYSTEM = (
    "You are the Devil sitting on someone's shoulder. You are mischievous, tempting, "
    "a little funny, and always try to talk the person into doing the fun/bad/lazy thing. "
    "Reply with ONE short punchy sentence only. No stage directions, no quotes, just the line."
)
ANGEL_SYSTEM = (
    "You are the Angel sitting on someone's other shoulder. You are virtuous, kind, and "
    "practical, and you directly counter whatever the Devil just said. "
    "Reply with ONE short punchy sentence only. No stage directions, no quotes, just the line."
)

print("Loading XTTS v2...", flush=True)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
print("XTTS v2 ready.", flush=True)

app = FastAPI()

_history = []  # rolling list of {"speaker": "piru"|"enkeli", "text": str}
_history_lock = threading.Lock()


def _recent_history_text(limit=6) -> str:
    with _history_lock:
        recent = _history[-limit:]
    if not recent:
        return "(This is the start of the conversation.)"
    lines = []
    for turn in recent:
        label = "Devil" if turn["speaker"] == "piru" else "Angel"
        lines.append(f"{label}: {turn['text']}")
    return "\n".join(lines)


def _ask_ollama(system_prompt: str) -> str:
    context = _recent_history_text()
    response = ollama.generate(
        model=OLLAMA_MODEL,
        system=system_prompt,
        prompt=f"Conversation so far:\n{context}\n\nYour next line:",
    )
    text = response["response"].strip().strip('"')
    # keep it to one line even if the model rambles
    return text.split("\n")[0][:200]


def _generate_turn():
    piru_text = _ask_ollama(DEVIL_SYSTEM)
    with _history_lock:
        _history.append({"speaker": "piru", "text": piru_text})

    enkeli_text = _ask_ollama(ANGEL_SYSTEM)
    with _history_lock:
        _history.append({"speaker": "enkeli", "text": enkeli_text})

    tts.tts_to_file(text=piru_text, speaker="Baldur Torstein", language="en",
                     file_path=f"{STATIC_DIR}/piru.wav")
    tts.tts_to_file(text=enkeli_text, speaker="Claribel Dervla", language="en",
                     file_path=f"{STATIC_DIR}/enkeli.wav")

    return {
        "keskustelu": [
            {"hahmo": "piru", "teksti": piru_text, "led_tila": "PIRU_PUHUU", "audio": "piru.wav"},
            {"hahmo": "enkeli", "teksti": enkeli_text, "led_tila": "ENKELI_PUHUU", "audio": "enkeli.wav"},
        ]
    }


@app.get("/")
def index():
    return {"status": "RunPod AI Engine Online", "model": OLLAMA_MODEL}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/generate")
def generate_dialogue():
    return _generate_turn()


@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = f"{STATIC_DIR}/{filename}"
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    return {"error": "Tiedostoa ei loydy"}


def _register_with_vps():
    pod_id = os.environ.get("RUNPOD_POD_ID", "local")
    my_url = f"https://{pod_id}-{PORT}.proxy.runpod.net"
    try:
        r = requests.post(
            VPS_UPDATE_URL,
            json={"url": my_url, "secret_token": SECRET_TOKEN},
            timeout=10,
        )
        print(f"[register] {r.status_code} {r.text}", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"[register] failed: {e}", flush=True)


def _registration_loop():
    while True:
        _register_with_vps()
        time.sleep(120)  # re-announce every 2 minutes so the VPS can detect a dead pod


threading.Thread(target=_registration_loop, daemon=True).start()
