"""One-off diagnostic: generates the same sample line in a batch of Aura-2
voices via OpenRouter, so you can listen to each side by side and pick one
for OPENROUTER_TTS_VOICE in pipeline.py, without needing a live call for
every candidate.

Run this on the same machine as bridge_service.py (needs OPENROUTER_API_KEY
set, same as the bridge itself):

    python3 test_deepgram_voices.py

Then play the saved .wav files, e.g.:
    afplay voice_samples/aura-2-thalia-en.wav          # macOS
    ffplay -nodisp -autoexit voice_samples/aura-2-thalia-en.wav  # most platforms
    aplay voice_samples/aura-2-thalia-en.wav           # Linux (ALSA)
    start voice_samples\\aura-2-thalia-en.wav           # Windows

Applies the same TTS_GAIN as pipeline.py's synthesize_pcm() so what you hear
here matches production loudness, not the raw (quieter) API output -- keep
this in sync if you tune TTS_GAIN there.

The voice names below are Aura-2's documented catalog as of when this was
written -- Deepgram's exact lineup can change. A name that fails just gets
skipped (printed, not fatal) so one bad/renamed entry doesn't stop the rest
from generating.
"""

import os
import re
import wave

import numpy as np
import requests

OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"
OPENROUTER_TTS_MODEL = "deepgram/aura-2"
TTS_GAIN = 2.0  # keep in sync with pipeline.py's TTS_GAIN
OUT_DIR = "voice_samples"

SAMPLE_TEXT = (
    "Hello! I'm Srinivas, calling from Raga Tech Source. "
    "This call is being recorded. Could you tell me your name?"
)

# A wider spread across genders/accents from Aura-2's catalog (91 voices
# total per Deepgram, spanning American/British/Irish/Australian/Filipino
# English) -- still not exhaustive, and names past the first ten are less
# certain (Deepgram's docs weren't directly reachable to cross-check every
# one while writing this). Add or remove names freely -- a wrong/renamed
# one just gets skipped, it won't stop the rest from generating.
CANDIDATE_VOICES = [
    "aura-2-draco-en",      # British male -- current production voice, for comparison
    "aura-2-thalia-en",     # American female -- Deepgram's commonly-used default
    "aura-2-orpheus-en",    # American male
    "aura-2-arcas-en",      # American male -- used earlier in this project
    "aura-2-athena-en",     # British female
    "aura-2-electra-en",    # British female
    "aura-2-angus-en",      # Irish male
    "aura-2-helios-en",     # American male
    "aura-2-luna-en",       # American female
    "aura-2-stella-en",     # American female
    "aura-2-hera-en",       # American female
    "aura-2-zeus-en",       # American male
    "aura-2-perseus-en",    # American male
    "aura-2-cora-en",       # American female
    "aura-2-cordelia-en",   # American female
    "aura-2-apollo-en",     # American male
    "aura-2-hermes-en",     # American male
    "aura-2-atlas-en",      # American male
    "aura-2-juno-en",       # American female
    "aura-2-selene-en",     # American female
]


def synthesize_and_save(voice: str) -> None:
    try:
        response = requests.post(
            url=OPENROUTER_TTS_URL,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_TTS_MODEL,
                "input": SAMPLE_TEXT,
                "voice": voice,
                "response_format": "pcm",
            },
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        print(f"  [{voice}] FAILED: {e}")
        return

    if not response.ok:
        print(f"  [{voice}] FAILED: HTTP {response.status_code} -- {response.text[:200]}")
        return

    content_type = response.headers.get("content-type", "")
    rate_match = re.search(r"rate=(\d+)", content_type)
    sample_rate = int(rate_match.group(1)) if rate_match else 24000

    samples = np.frombuffer(response.content, dtype="<i2")
    samples = np.clip(samples.astype(np.float32) * TTS_GAIN, -32768, 32767).astype(np.int16)

    out_path = os.path.join(OUT_DIR, f"{voice}.wav")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    print(f"  [{voice}] saved -> {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Synthesizing {len(CANDIDATE_VOICES)} candidate voices...")
    for voice_name in CANDIDATE_VOICES:
        synthesize_and_save(voice_name)
    print(f"\nDone. Play the files in {OUT_DIR}/ and listen for clarity/tone -- e.g.:")
    print(f"  ffplay -nodisp -autoexit {OUT_DIR}/aura-2-draco-en.wav")
    print("Once you pick one, set OPENROUTER_TTS_VOICE in pipeline.py to that voice name.")
