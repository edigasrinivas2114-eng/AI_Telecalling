"""One-off diagnostic: calls OpenRouter's TTS endpoint directly (bypassing the
phone bridge entirely) and saves the response as a proper, playable WAV file.
Run this on the same machine as bridge_service.py:

    python3 test_tts_direct.py

Then play test_tts_output.wav in any media player. If it's silent or garbled
there too, the problem is the TTS model/API response itself, not anything in
bridge_service.py's resampling or AudioSocket framing. If it sounds correct
here, the bug is downstream in the bridge's audio path instead.

Note: this endpoint returns raw headerless PCM (Content-Type:
audio/pcm;rate=<N>;channels=<N>), not an mp3 file, despite OpenRouter's own
sample code writing the response straight to "output.mp3" -- that sample's
filename is misleading, confirmed by inspecting the actual Content-Type
header. Playing those raw bytes directly (or decoding them as mp3, as an
earlier version of pipeline.py did) produces garbage/silence, not a
"can't play this file" error, which is why that bug wasn't obvious from
logs alone.
"""

import os
import re
import wave

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

content_type = response.headers.get("content-type", "")
print("HTTP status:", response.status_code)
print("Content-Type:", content_type)
print("Bytes received:", len(response.content))

response.raise_for_status()

rate_match = re.search(r"rate=(\d+)", content_type)
channels_match = re.search(r"channels=(\d+)", content_type)
sample_rate = int(rate_match.group(1)) if rate_match else 24000
channels = int(channels_match.group(1)) if channels_match else 1
print(f"Parsed: {sample_rate}Hz, {channels} channel(s)")

nonzero_bytes = sum(1 for b in response.content if b != 0)
print(f"Non-zero bytes: {nonzero_bytes} / {len(response.content)} "
      f"({100 * nonzero_bytes / max(len(response.content), 1):.1f}%) "
      "-- near 0% means the API returned near-silence, not a decode bug")

with wave.open("test_tts_output.wav", "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(sample_rate)
    wf.writeframes(response.content)
print("Saved test_tts_output.wav -- play it and listen.")
