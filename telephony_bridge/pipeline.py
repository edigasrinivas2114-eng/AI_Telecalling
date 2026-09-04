"""STT -> RAG -> LLM (Claude) -> TTS pipeline for live calls.

This is the phase-2 counterpart to the Colab notebook's pipeline:
- STT: faster-whisper (same as the notebook)
- RAG: Chroma + sentence-transformers (same as the notebook)
- LLM: Claude API (claude-haiku-4-5) instead of a local model. The earlier
  local-Ollama approach (qwen2.5:3b-instruct on CPU) was both too slow
  (15-25s+ per reply) and not reliably fluent in Telugu -- a small
  general-purpose open model at that size isn't a strong bet for a
  lower-resource language. Claude's API runs on real GPU infrastructure (fast)
  and Haiku 4.5 is capable and cheap. Tradeoff: no longer free/fully
  self-hosted for this piece -- needs an ANTHROPIC_API_KEY and has a real,
  if small, per-call cost. Requires the `anthropic` package and that env var
  set (see README).
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

import anthropic
import chromadb
import edge_tts
import numpy as np
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

# Fastest, cheapest current Claude model -- a good fit for short conversational
# turns like this (short system prompt + short history + short reply), not a
# "downgrade": the alternative it's replacing is a local 3B model, not a
# bigger Claude model. Reads the API key from the ANTHROPIC_API_KEY env var --
# never hardcode it here.
ANTHROPIC_MODEL = "claude-haiku-4-5"
anthropic_client = anthropic.Anthropic()

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
    messages = list(chat_history)
    messages.append({
        "role": "user",
        "content": f"RETRIEVED CONTEXT:\n{context_text}\n\nCALLER SAID: {user_text}",
    })

    # Telugu: "One moment, let me try that again." -- unverified wording (see
    # the translation note at the top of programme_config.py), used only as a
    # spoken fallback on API failure, never as text sent anywhere.
    fallback_reply = "ఒక్క నిమిషం, మళ్ళీ ప్రయత్నిస్తాను."

    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            temperature=0.4,
            system=SYSTEM_PROMPT_TEMPLATE,
            messages=messages,
        )
        reply = next((b.text for b in response.content if b.type == "text"), "").strip()
    except anthropic.RateLimitError as e:
        print(f"  [LLM ERROR] Claude API rate limited: {e.message}")
        reply = fallback_reply
    except anthropic.APIStatusError as e:
        print(f"  [LLM ERROR] Claude API error {e.status_code}: {e.message}")
        reply = fallback_reply
    except anthropic.APIConnectionError:
        print("  [LLM ERROR] Could not reach the Claude API (network issue)")
        reply = fallback_reply

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
