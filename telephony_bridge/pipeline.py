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
- TTS: Sarvam AI's Bulbul v3 (`bulbul:v3`), a text-to-speech model built and
  trained specifically for Indian languages (11 of them, Telugu included),
  reached through Sarvam's own API/SDK -- a separate paid account from
  OpenRouter, since OpenRouter doesn't host any Telugu-specialized TTS model.
  This replaced Google's Gemini 3.1 Flash TTS Preview (via OpenRouter), which
  was a general-purpose multilingual model with unconfirmed Telugu quality --
  Sarvam's whole product focus is Indian-language speech, a stronger bet for
  this project's actual audience than a general model's broad language list.
  Uses the official `sarvamai` Python SDK rather than raw HTTP calls, both
  because Sarvam's own request/response field names have changed recently
  (e.g. `target_language_code` -> `language_code`) and because the SDK
  absorbs changes like that instead of this code needing to track them.
  Requests audio at 8kHz directly (`speech_sample_rate=8000`) -- the same
  rate AudioSocket needs -- since Sarvam is designed for telephony use and
  should sound better tuned for 8kHz than a naive downsample from a higher
  native rate would. The response is a genuine WAV container (unlike the
  OpenRouter/Gemini endpoint this replaced, which claimed mp3 but actually
  returned raw headerless PCM), so it's parsed with the stdlib `wave` module.
  Needs a SARVAM_API_KEY env var (see README).
"""

import base64
import io
import json
import os
import re
import time
import wave

import chromadb
import numpy as np
import openai
import requests
from sarvamai import SarvamAI
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

# Sarvam AI's Bulbul TTS -- see module docstring. A separate paid account
# from OpenRouter; reads its own key from SARVAM_API_KEY (see README).
# "anand" is one of bulbul:v3's ~39 speaker names, confirmed valid for this
# model version (speaker names aren't interchangeable across bulbul
# versions) -- picked as a male voice to fit the "Srinivas" persona, swap to
# another name from Sarvam's voice list if it doesn't suit.
SARVAM_TTS_MODEL = "bulbul:v3"
SARVAM_TTS_SPEAKER = "anand"
SARVAM_TTS_LANGUAGE_CODE = "te-IN"
sarvam_client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

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


def synthesize_pcm(text: str):
    """Returns (mono int16 PCM samples, sample_rate) -- callers must resample
    to whatever they actually need."""
    response = sarvam_client.text_to_speech.convert(
        text=text,
        language_code=SARVAM_TTS_LANGUAGE_CODE,
        model=SARVAM_TTS_MODEL,
        speaker=SARVAM_TTS_SPEAKER,
        speech_sample_rate=8000,  # matches AudioSocket's native rate -- no resampling needed
    )
    audio_bytes = base64.b64decode(response.audios[0])
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        pcm_bytes = wf.readframes(wf.getnframes())

    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, sample_rate
