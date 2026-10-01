"""
Runs inside the RunPod container. Generates an ongoing Angel vs Devil
conversation using a local LLM (via Ollama) and synthesizes each line with
XTTS v2 (two distinct built-in speakers). Registers this pod's public proxy
URL with the VPS relay on startup and every few minutes after that, so the
ESP32 (which only ever talks to the VPS) always has a live backend.
"""
import os
import random
import threading
import time

import ollama
import requests
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse

# PyTorch 2.6+ flipped torch.load's default to weights_only=True, which
# breaks loading Coqui TTS's XTTS checkpoint (it pickles config classes like
# XttsConfig that aren't on PyTorch's auto-allowlist, raising
# "Weights only load failed... Unsupported global"). We trust the source
# (official Coqui-released XTTS v2 weights), so restore the old default
# for this process rather than hand-allowlisting every class TTS pickles.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from TTS.api import TTS

STATIC_DIR = "/workspace/halloween_pod/static"
os.makedirs(STATIC_DIR, exist_ok=True)

VPS_UPDATE_URL = os.environ.get("VPS_UPDATE_URL", "https://aihenkka.xyz/halloween-api/update-pod")
# No default here on purpose: this is a public image/repo, so the real shared
# secret must come from the RunPod template's environment variables, not source.
SECRET_TOKEN = os.environ.get("RUNPOD_SECRET_TOKEN")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
PORT = int(os.environ.get("POD_PORT", "8005"))

# A random topic per turn keeps the conversation from converging on the same
# few generic tropes (money/work were the repeat offenders in testing).
# Picked for the classic "small vice vs. doing the right thing" shape --
# actual moral micro-dilemmas, not just random fun activities.
TOPICS = [
    "eating the last slice without asking", "lying about why you're late",
    "taking credit for a coworker's idea", "cutting in line",
    "keeping extra change the cashier gave by mistake",
    "gossiping about a friend behind their back",
    "backing out of a promise to help a friend move",
    "snapping at someone who didn't deserve it",
    "holding a grudge instead of forgiving", "littering because the bin is far away",
    "skipping the gym", "overindulging in dessert before dinner",
    "procrastinating on a deadline", "ghosting a friend's message",
    "flirting with someone while already taken", "taking a parking spot you don't deserve",
    "honking in a fit of road rage", "pretending to be busy to avoid helping out",
    "not telling the cashier they undercharged you", "exaggerating a story to look better",
    "sneaking a peek at someone's phone", "calling in sick when you're not really sick",
    "spreading a juicy rumor", "copying someone's homework",
    "not admitting a mistake at work", "keeping the cash from a found wallet",
    "judging someone based on appearance",
    "envying a friend's success instead of being happy for them",
    # Halloween-party specific -- this started life as a Halloween decoration prop
    "eating way too much candy before the party even starts",
    "stealing candy from the trick-or-treat bowl when no one's looking",
    "scaring little trick-or-treaters way harder than necessary",
    "spiking the Halloween punch", "ding-dong-ditching a neighbor's house",
    "egging a house as a Halloween prank", "toilet-papering a neighbor's yard",
    "hogging all the good candy and leaving the rest", "double-dipping at the snack table",
    "jump-scaring a friend way too hard at the party", "telling a lie just to freak someone out",
    "sneaking into the adults-only part of the Halloween party",
    "not sharing Halloween candy with a younger sibling",
    "pretending to be sick to skip handing out candy to trick-or-treaters",
    "cutting in line for the haunted house", "copying someone else's costume idea on purpose",
    "staying out trick-or-treating well past curfew",
    "making fun of someone's Halloween costume out loud",
    "judging a stranger's weird outfit at the party",
    "laughing at someone's homemade costume behind their back",
]

DEVIL_SYSTEM_TEMPLATE = (
    "You are the Devil sitting on someone's shoulder. You are mischievous, tempting, "
    "and a little funny. Right now you're trying to talk them into: {topic}. "
    "Reply with ONE short punchy sentence only. No stage directions, no quotes, just the line."
)
ANGEL_SYSTEM_TEMPLATE = (
    "You are the Angel sitting on someone's other shoulder. You are virtuous, kind, and "
    "practical, and you directly counter whatever the Devil just said about: {topic}. "
    "Reply with ONE short punchy sentence only. No stage directions, no quotes, just the line."
)

print("Loading XTTS v2...", flush=True)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")

# Pick two distinct built-in speakers programmatically instead of hardcoding
# names (XTTS v2 ships ~58 preset speakers, but exact names/spelling have
# shifted between versions -- a hardcoded guess here is what broke this
# earlier: "Baldur Torstein" simply doesn't exist in this build's list).
# The public `tts.speakers` property doesn't exist on this installed version,
# so reach into the same place the model's own synthesize() method does:
# self.speaker_manager.speakers[speaker_id] inside tts_model.
try:
    _available_speakers = list(tts.speakers or [])
except AttributeError:
    _available_speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())

if len(_available_speakers) < 2:
    raise RuntimeError(f"Expected at least 2 built-in speakers, got: {_available_speakers}")

# Optional explicit override via RunPod template env vars, so picking a
# better-sounding voice is just a pod restart, not a new Docker build.
# Falls back to the first two available speakers if unset/invalid.
_devil_override = os.environ.get("DEVIL_SPEAKER")
_angel_override = os.environ.get("ANGEL_SPEAKER")
DEVIL_SPEAKER = _devil_override if _devil_override in _available_speakers else _available_speakers[0]
ANGEL_SPEAKER = _angel_override if _angel_override in _available_speakers else _available_speakers[1]
print(f"XTTS v2 ready. Using speakers -> devil: {DEVIL_SPEAKER!r}, angel: {ANGEL_SPEAKER!r}", flush=True)

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
    topic = random.choice(TOPICS)
    piru_text = _ask_ollama(DEVIL_SYSTEM_TEMPLATE.format(topic=topic))
    with _history_lock:
        _history.append({"speaker": "piru", "text": piru_text})

    enkeli_text = _ask_ollama(ANGEL_SYSTEM_TEMPLATE.format(topic=topic))
    with _history_lock:
        _history.append({"speaker": "enkeli", "text": enkeli_text})

    tts.tts_to_file(text=piru_text, speaker=DEVIL_SPEAKER, language="en",
                     file_path=f"{STATIC_DIR}/piru.wav")
    tts.tts_to_file(text=enkeli_text, speaker=ANGEL_SPEAKER, language="en",
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


@app.get("/speakers")
def speakers():
    return {"all_speakers": _available_speakers, "devil": DEVIL_SPEAKER, "angel": ANGEL_SPEAKER}


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
    if not SECRET_TOKEN:
        print("[register] RUNPOD_SECRET_TOKEN is not set - skipping registration. "
              "Set it in the RunPod template's environment variables.", flush=True)
        return
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
