"""One-off diagnostic: calls Sarvam AI's TTS API directly (bypassing the
phone bridge entirely) and saves the response as a playable WAV file. Run
this on the same machine as bridge_service.py, after `pip install -r
requirements.txt` and setting SARVAM_API_KEY:

    python3 test_sarvam_tts_direct.py

Then play test_sarvam_output.wav in any media player and listen for whether
it's genuinely intelligible, natural-sounding Telugu -- that's the whole
point of switching to a Telugu-specialized model, so it's worth checking
directly rather than assuming it worked just because the request succeeded.
"""

import base64
import io
import os
import wave

from sarvamai import SarvamAI

SARVAM_TTS_MODEL = "bulbul:v3"
SARVAM_TTS_SPEAKER = "anand"
SARVAM_TTS_LANGUAGE_CODE = "te-IN"

text = "నమస్కారం! నేను Srinivas, Raga Tech Source నుండి మాట్లాడుతున్నాను."

client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
response = client.text_to_speech.convert(
    text=text,
    language_code=SARVAM_TTS_LANGUAGE_CODE,
    model=SARVAM_TTS_MODEL,
    speaker=SARVAM_TTS_SPEAKER,
    speech_sample_rate=8000,
)

print("Number of audio chunks returned:", len(response.audios))

audio_bytes = base64.b64decode(response.audios[0])
print("Decoded bytes:", len(audio_bytes))
print("First 16 bytes:", audio_bytes[:16], "(should start with b'RIFF' for a real WAV file)")

with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
    sample_rate = wf.getframerate()
    channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    nframes = wf.getnframes()
    pcm_bytes = wf.readframes(nframes)

print(f"Parsed WAV: {sample_rate}Hz, {channels} channel(s), {sampwidth * 8}-bit, "
      f"{nframes} frames ({nframes / sample_rate:.2f}s)")

nonzero_bytes = sum(1 for b in pcm_bytes if b != 0)
print(f"Non-zero bytes: {nonzero_bytes} / {len(pcm_bytes)} "
      f"({100 * nonzero_bytes / max(len(pcm_bytes), 1):.1f}%) "
      "-- near 0% means near-silence, not real speech")

with open("test_sarvam_output.wav", "wb") as f:
    f.write(audio_bytes)
print("Saved test_sarvam_output.wav -- play it and listen for real, natural Telugu.")
