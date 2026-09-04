"""STT -> RAG -> LLM (Ollama) -> TTS pipeline for live calls.

This is the phase-2, CPU-friendly counterpart to the Colab notebook's pipeline:
- STT: faster-whisper (same as the notebook)
- RAG: Chroma + sentence-transformers (same as the notebook)
- LLM: Ollama instead of transformers+bitsandbytes -- bitsandbytes 4-bit needs a
  CUDA GPU, which an always-on, free, self-hosted phone bridge won't have. Ollama
  runs quantized models efficiently on CPU, which is what "free and always on"
  requires. Swap OLLAMA_MODEL for a bigger model once this moves to a GPU host.
- TTS: Piper (same as the notebook, already CPU-only)
"""

import io
import time
import wave

import chromadb
import numpy as np
import requests
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
from sentence_transformers import SentenceTransformer

from programme_config import (
    CONSENT_DISCLOSURE,
    KNOWLEDGE_BASE,
    OPT_OUT_REPLY,
    SYSTEM_PROMPT_TEMPLATE,
    is_opt_out_request,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct"  # small enough to run on CPU for testing

# "medium" trades speed for accuracy vs "small" -- noticeably slower on CPU,
# but mishears fewer words, which matters more than shaving a few seconds off
# an already multi-second reply time.
WHISPER_MODEL_SIZE = "medium"

# Telugu voice, confirmed directly against the rhasspy/piper-voices file
# browser (te/te_IN has three speakers: maya, padmavathi, venkatesh, all at
# "medium" quality). venkatesh is male, pairing with the "Srinivas" persona --
# swap to "padmavathi" or "maya" (same path pattern) for a female voice.
# Single-speaker voice, so no speaker_id needed (PIPER_SPEAKER_ID is ignored
# for single-speaker models).
PIPER_MODEL_PATH = "piper_voices/te_IN-venkatesh-medium.onnx"
PIPER_CONFIG_PATH = "piper_voices/te_IN-venkatesh-medium.onnx.json"
PIPER_SPEAKER_ID = 0

# faster-whisper transcribes much more reliably when told the expected
# language up front instead of auto-detecting it turn by turn.
WHISPER_LANGUAGE = "te"

# >1.0 = slower speech, <1.0 = faster. Piper's default (omitted) reads quite
# fast for a phone call; 1.2 slows it down ~20%.
PIPER_LENGTH_SCALE = 1.2

print("Loading faster-whisper...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

print("Loading embedder + knowledge base...")
# Multilingual, not "all-MiniLM-L6-v2" (English-only) -- with Telugu callers,
# an English-only embedder would badly mismatch a Telugu question against the
# (English) knowledge base text, breaking retrieval and the grounding
# requirement. This model covers Telugu among 50+ languages.
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.EphemeralClient()
try:
    chroma_client.delete_collection("programme_kb")
except Exception:
    pass
kb_collection = chroma_client.create_collection("programme_kb")
kb_collection.add(
    ids=[d["id"] for d in KNOWLEDGE_BASE],
    documents=[d["text"] for d in KNOWLEDGE_BASE],
    embeddings=embedder.encode([d["text"] for d in KNOWLEDGE_BASE]).tolist(),
)

print("Loading Piper voice...")
piper_voice = PiperVoice.load(PIPER_MODEL_PATH, config_path=PIPER_CONFIG_PATH, use_cuda=False)

SUPPRESSION_LIST = []


def retrieve_context(query: str, k: int = 2) -> str:
    q_emb = embedder.encode([query]).tolist()
    results = kb_collection.query(query_embeddings=q_emb, n_results=k)
    return "\n".join(results["documents"][0])


def transcribe_pcm(pcm_16khz_f32: np.ndarray) -> str:
    """pcm_16khz_f32: mono float32 samples in [-1, 1] at 16kHz."""
    segments, _ = whisper_model.transcribe(pcm_16khz_f32, beam_size=5, language=WHISPER_LANGUAGE)
    return " ".join(seg.text.strip() for seg in segments).strip()


def generate_response(user_text: str, chat_history=None) -> dict:
    chat_history = chat_history or []
    t0 = time.time()

    if is_opt_out_request(user_text):
        SUPPRESSION_LIST.append(user_text)
        reply = OPT_OUT_REPLY
        return {"reply": reply, "suppressed": True, "elapsed": time.time() - t0}

    context_text = retrieve_context(user_text, k=2)
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE}]
    messages += chat_history
    messages.append({
        "role": "user",
        "content": f"RETRIEVED CONTEXT:\n{context_text}\n\nCALLER SAID: {user_text}",
    })

    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
              "options": {"num_predict": 60, "temperature": 0.4}},
        timeout=60,
    )
    resp.raise_for_status()
    reply = resp.json()["message"]["content"].strip()
    return {"reply": reply, "suppressed": False, "elapsed": time.time() - t0}


_SYN_CONFIG = SynthesisConfig(speaker_id=PIPER_SPEAKER_ID, length_scale=PIPER_LENGTH_SCALE)


def synthesize_pcm(text: str) -> np.ndarray:
    """Returns mono int16 PCM samples at the voice's native sample rate (see
    piper_voice.config.sample_rate, typically 22050 Hz)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file, syn_config=_SYN_CONFIG)
    buf.seek(0)
    with wave.open(buf, "rb") as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def piper_sample_rate() -> int:
    return piper_voice.config.sample_rate
