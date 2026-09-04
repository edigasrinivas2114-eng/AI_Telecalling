"""STT -> RAG -> LLM (Ollama) -> TTS pipeline for live calls.

This is the phase-2, CPU-friendly counterpart to the Colab notebook's pipeline:
- STT: faster-whisper (same as the notebook)
- RAG: Chroma + sentence-transformers (same as the notebook)
- LLM: Ollama instead of transformers+bitsandbytes -- bitsandbytes 4-bit needs a
  CUDA GPU, which an always-on, free, self-hosted phone bridge won't have. Ollama
  runs quantized models efficiently on CPU, which is what "free and always on"
  requires. Swap OLLAMA_MODEL for a bigger model once this moves to a GPU host.
- TTS: edge-tts (free access to Microsoft's neural voices -- no API key, no
  Azure account, no cost -- instead of Piper). Piper's Telugu voices are
  robotic-sounding; these are the same production-quality neural voices Azure
  sells, reached through Microsoft Edge's "Read aloud" service. This is an
  unofficial (if long-stable and widely used) way of reaching that service,
  and each call needs live internet access, unlike Piper's fully offline
  synthesis.
"""

import asyncio
import io
import time

import chromadb
import edge_tts
import numpy as np
import requests
from faster_whisper import WhisperModel
from pydub import AudioSegment
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

# faster-whisper transcribes much more reliably when told the expected
# language up front instead of auto-detecting it turn by turn.
WHISPER_LANGUAGE = "te"

# te-IN-MohanNeural (male) pairs with the "Srinivas" persona; te-IN-ShrutiNeural
# is the female alternative -- easy one-line swap if you'd rather try a woman's
# voice. Both confirmed real Azure/edge-tts voice names.
EDGE_TTS_VOICE = "te-IN-MohanNeural"
EDGE_TTS_RATE = "+15%"  # positive = faster; tune further if still too slow/fast

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


async def _edge_tts_mp3_bytes(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE, rate=EDGE_TTS_RATE)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def synthesize_pcm(text: str):
    """Returns (mono int16 PCM samples, sample_rate) -- edge-tts's native
    output rate, so callers must resample to whatever they actually need."""
    mp3_bytes = asyncio.run(_edge_tts_mp3_bytes(text))
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3").set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    return samples, audio.frame_rate
