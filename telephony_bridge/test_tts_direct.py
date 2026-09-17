"""One-off diagnostic: calls OpenRouter's TTS endpoint directly (bypassing the
phone bridge entirely) and saves the raw response to a file you can play
locally. Run this on the same machine as bridge_service.py:

    python3 test_tts_direct.py

Then play test_tts_output.mp3 in any media player. If it's silent or garbled
there too, the problem is the TTS model/API response itself, not anything in
bridge_service.py's resampling or AudioSocket framing. If it sounds correct
here, the bug is downstream in the bridge's audio path instead.
"""

import os

import requests

OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"
OPENROUTER_TTS_MODEL = "google/gemini-3.1-flash-tts-preview"
OPENROUTER_TTS_VOICE = "Zephyr"

text = "నమస్కారం! నేను Srinivas, Raga Tech Source నుండి మాట్లాడుతున్నాను."

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
    },
    timeout=30,
)

print("HTTP status:", response.status_code)
print("Content-Type:", response.headers.get("content-type"))
print("Bytes received:", len(response.content))
print("First 16 bytes:", response.content[:16])

response.raise_for_status()

with open("test_tts_output.mp3", "wb") as f:
    f.write(response.content)
print("Saved test_tts_output.mp3 -- play it and listen.")
