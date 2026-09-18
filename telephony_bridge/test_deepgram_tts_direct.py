"""One-off diagnostic: calls Deepgram's Aura-2 TTS via OpenRouter directly
(bypassing the phone bridge entirely) and saves the response as an mp3 file.
Run this on the same machine as bridge_service.py:

    python3 test_deepgram_tts_direct.py

Then play test_deepgram_output.mp3 and listen for quality/latency.

Note: OpenRouter's /audio/speech endpoint defaults to raw PCM, not mp3
(the opposite of OpenAI's own API default) -- this request explicitly sets
response_format="mp3" to get a real, predictable mp3 back instead of
guessing, after an earlier integration (Gemini TTS) got bitten by exactly
this kind of undocumented default.

Aura-2 is English-only -- there is no Telugu (or any non-English) voice for
it at all, so this is only useful if the whole script goes back to English.

Note: this saves the RAW mp3 straight from the API, without the TTS_GAIN
digital volume boost pipeline.py's synthesize_pcm() now applies (real calls
came through quieter than expected) -- so this file will sound quieter than
an actual call. For comparing multiple voices at production loudness, use
test_deepgram_voices.py instead, which applies the same gain.
"""

import os

import requests

OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"
OPENROUTER_TTS_MODEL = "deepgram/aura-2"
OPENROUTER_TTS_VOICE = "aura-2-draco-en"  # British male voice; edit this to try others, e.g.
                                           # "aura-2-arcas-en" (American male), "aura-2-electra-en"
                                           # (British female) -- see Deepgram's Aura-2 voice list

text = "Hello! This is Srinivas, calling from Raga Tech Source. This call is being recorded. Could you tell me your name?"

response = requests.post(
    url=OPENROUTER_TTS_URL,
    headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    },
    json={
        "model": OPENROUTER_TTS_MODEL,
        "input": text,
        "voice": OPENROUTER_TTS_VOICE,
        "response_format": "mp3",
    },
    timeout=30,
)

print("HTTP status:", response.status_code)
print("Content-Type:", response.headers.get("content-type"))
print("Bytes received:", len(response.content))
print("First 16 bytes:", response.content[:16])

response.raise_for_status()

with open("test_deepgram_output.mp3", "wb") as f:
    f.write(response.content)
print("Saved test_deepgram_output.mp3 -- play it and listen for quality + how fast this ran.")
