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
from piper import PiperVoice
from sentence_transformers import SentenceTransformer

from programme_config import (
    CONSENT_DISCLOSURE,
    KNOWLEDGE_BASE,
    SYSTEM_PROMPT_TEMPLATE,
    is_opt_out_request,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct"  # small enough to run on CPU for testing

WHISPER_MODEL_SIZE = "small"
PIPER_MODEL_PATH = "piper_voices/en_US-lessac-medium.onnx"
PIPER_CONFIG_PATH = "piper_voices/en_US-lessac-medium.onnx.json"

print("Loading faster-whisper...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

print("Loading embedder + knowledge base...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
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
    segments, _ = whisper_model.transcribe(pcm_16khz_f32, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


def generate_response(user_text: str, chat_history=None) -> dict:
    chat_history = chat_history or []
    t0 = time.time()

    if is_opt_out_request(user_text):
        SUPPRESSION_LIST.append(user_text)
        reply = (
            "Understood -- I've flagged this number to be removed from our calling list "
            "immediately, and you will not receive further calls about this programme. "
            "Sorry for the interruption, and thank you for your time."
        )
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
              "options": {"num_predict": 120, "temperature": 0.4}},
        timeout=60,
    )
    resp.raise_for_status()
    reply = resp.json()["message"]["content"].strip()
    return {"reply": reply, "suppressed": False, "elapsed": time.time() - t0}


def synthesize_pcm(text: str) -> np.ndarray:
    """Returns mono int16 PCM samples at the voice's native sample rate (see
    piper_voice.config.sample_rate, typically 22050 Hz)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)
    buf.seek(0)
    with wave.open(buf, "rb") as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def piper_sample_rate() -> int:
    return piper_voice.config.sample_rate
