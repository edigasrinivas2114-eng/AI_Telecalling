"""STT -> RAG -> LLM (Claude via OpenRouter) -> TTS pipeline for live calls.

This is the phase-2 counterpart to the Colab notebook's pipeline:
- STT: Whisper Large V3 Turbo, hosted via OpenRouter (`openai/whisper-large-v3-turbo`)
  rather than faster-whisper running locally. Local Whisper on this machine's CPU
  kept hitting slow/stuck transcriptions (occasionally 50-70+ seconds) that no
  amount of tuning (smaller model, timeouts, VAD adjustments) fully resolved --
  moving the compute to OpenRouter's hosted infrastructure removes that CPU
  contention entirely. Same Whisper model family as before, so Telugu support
  carries over. Billed against the OpenRouter credit balance, same account as
  the LLM call below. POSTs directly to OpenRouter's transcription endpoint
  (JSON body with base64-encoded WAV audio) since this shape is specific to
  OpenRouter, not the same as the openai package's audio.transcriptions
  helper (which expects multipart file upload).
- RAG: Chroma + sentence-transformers (same as the notebook)
- LLM: Claude Haiku 4.5, reached through OpenRouter (`anthropic/claude-haiku-4.5`)
  rather than the direct Anthropic API. OpenRouter exposes an OpenAI-compatible
  API for every model it hosts, Claude included, so this uses the `openai`
  package pointed at OpenRouter's endpoint, billed against the OpenRouter
  account's own credit balance -- a free OpenRouter *model* was tried first
  (google/gemma-4-26b-a4b-it:free, then z-ai/glm-5.2:free) but both hit real
  reliability problems (shared-pool rate limits, then a broken upstream
  provider). This paid model avoids the shared free-tier pool entirely.
  Needs an OPENROUTER_API_KEY env var with a funded OpenRouter credit balance
  (see README).
- TTS: edge-tts (free access to Microsoft's neural voices -- no API key, no
  Azure account, no cost -- instead of Piper). Piper's Telugu voices are
  robotic-sounding; these are the same production-quality neural voices Azure
  sells, reached through Microsoft Edge's "Read aloud" service. This is an
  unofficial (if long-stable and widely used) way of reaching that service,
  and each call needs live internet access, unlike Piper's fully offline
  synthesis. Switched back to Telugu per explicit request, after an earlier
  English-only period -- Telugu neural voices (even Microsoft's) still sound
  more synthetic than English ones, a real limitation across every TTS
  provider, but real Telugu-speaking leads matter more than that gap.
"""

import asyncio
import base64
import io
import json
import os
import re
import time
import wave

import chromadb
import edge_tts
import numpy as np
import openai
import requests
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer

from programme_config import (
    CONSENT_DISCLOSURE,
    KNOWLEDGE_BASE,
    OPT_OUT_REPLY,
    SYSTEM_PROMPT_TEMPLATE,
    is_opt_out_request,
)

# Claude Haiku 4.5 via OpenRouter (paid, billed against the OpenRouter account's
# credit balance) -- not the free tier, and not the direct Anthropic API. Reads
# the key from OPENROUTER_API_KEY -- never hardcode it here.
OPENROUTER_MODEL = "anthropic/claude-haiku-4.5"
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Hosted Whisper via OpenRouter -- see module docstring for why this replaced
# local faster-whisper. Uses the same OPENROUTER_API_KEY as the LLM call.
OPENROUTER_STT_MODEL = "openai/whisper-large-v3-turbo"
OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
STT_TIMEOUT_S = 15  # generous for a network call; hosted Whisper itself is very fast

# Back to Telugu per explicit request -- te-IN-MohanNeural (male) pairs with
# the "Srinivas" persona; te-IN-ShrutiNeural is the female alternative. Both
# confirmed real Azure/edge-tts voice names. Known tradeoff (see module
# docstring): still more synthetic-sounding than English neural voices, a
# real limitation of Telugu TTS across every provider today.
EDGE_TTS_VOICE = "te-IN-MohanNeural"
EDGE_TTS_RATE = "+15%"  # positive = faster; tune further if still too slow/fast

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


def _strip_markdown(text: str) -> str:
    """Backstop for TTS: the system prompt tells the model not to use markdown,
    but that's not guaranteed, and asterisks/headers read aloud sound wrong."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text.replace("*", "").replace("_", "")


def retrieve_context(query: str, k: int = 2) -> str:
    q_emb = embedder.encode([query]).tolist()
    results = kb_collection.query(query_embeddings=q_emb, n_results=k)
    return "\n".join(results["documents"][0])


# Cyrillic (U+0400-04FF) and Devanagari (U+0900-097F) -- a Telugu- or
# English-speaking caller can never genuinely produce either script. Real
# test calls showed the hosted STT hallucinating full sentences in these
# scripts on noisy/ambiguous clips (the `language` hint below doesn't fully
# prevent this), so any transcription containing them is almost certainly
# garbage, not real speech -- treat it as "heard nothing" rather than
# passing hallucinated text to the LLM.
_INVALID_SCRIPT_RE = re.compile(r"[Ѐ-ӿऀ-ॿ]")


def _wav_bytes_from_pcm(pcm_16khz_f32: np.ndarray) -> bytes:
    """Hosted transcription needs an actual audio file, not a raw sample
    array -- wraps the same 16kHz mono float32 samples the old local-Whisper
    path used into an in-memory WAV container."""
    pcm_int16 = np.clip(pcm_16khz_f32 * 32768.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def transcribe_pcm(pcm_16khz_f32: np.ndarray) -> str:
    """pcm_16khz_f32: mono float32 samples in [-1, 1] at 16kHz."""
    wav_bytes = _wav_bytes_from_pcm(pcm_16khz_f32)
    b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

    try:
        response = requests.post(
            url=OPENROUTER_STT_URL,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OPENROUTER_STT_MODEL,
                "input_audio": {"data": b64_audio, "format": "wav"},
                # "te" (ISO-639-1) -- the standard Whisper language-hint param name;
                # not confirmed against OpenRouter's own docs for this endpoint (the
                # sample we had didn't show it), but without it short/ambiguous clips
                # were being auto-detected as random languages/scripts (a Telugu
                # greeting transcribed once in Cyrillic, once in English letters) --
                # test this actually narrows it to Telugu; revert if it's ignored or
                # causes errors.
                "language": "te",
            }),
            timeout=STT_TIMEOUT_S,
        )
        response.raise_for_status()
        text = response.json().get("text", "").strip()
        if _INVALID_SCRIPT_RE.search(text):
            print(f"  [STT] discarding transcription in an impossible script: \"{text}\"")
            return ""
        return text
    except requests.exceptions.Timeout:
        print(f"  [STT] timed out after {STT_TIMEOUT_S}s -- abandoning this transcription")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"  [STT ERROR] OpenRouter transcription failed: {e}")
        return ""


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

    # Telugu: "One moment, let me try that again." -- spoken fallback on API
    # failure only, never sent anywhere as text.
    fallback_reply = "ఒక్క నిమిషం, మళ్ళీ ప్రయత్నిస్తాను."

    try:
        # OpenAI-style chat payload: system prompt goes inside `messages` as
        # its own entry (OpenRouter/OpenAI has no separate top-level `system`
        # field the way the Anthropic API does).
        response = openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            max_tokens=300,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE}] + messages,
        )
        reply = _strip_markdown((response.choices[0].message.content or "").strip())
    except openai.RateLimitError as e:
        print(f"  [LLM ERROR] OpenRouter rate limited: {e}")
        reply = fallback_reply
    except openai.APIStatusError as e:
        print(f"  [LLM ERROR] OpenRouter API error {e.status_code}: {e.message}")
        reply = fallback_reply
    except openai.APIConnectionError:
        print("  [LLM ERROR] Could not reach OpenRouter (network issue)")
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
