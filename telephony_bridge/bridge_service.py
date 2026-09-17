"""Exotel AgentStream (Voicebot Applet) <-> STT/RAG/LLM/TTS bridge.

Replaces the earlier Asterisk AudioSocket bridge -- that setup kept hitting
"one-way audio" (caller's mic reached the bridge fine, proven by working
STT, but the bot's replies never reached the caller's ear, confirmed on two
different audio devices including AirPods) that pointed to Asterisk's own
SIP/RTP/NAT handling, not this code. Exotel is a cloud telephony provider:
it terminates the actual phone call on its own infrastructure and connects
to this bridge over a plain WebSocket, so there's no local SIP/RTP/NAT setup
to get wrong. It's also India-focused (this business's actual market) with
built-in DLT/TRAI compliance handling, unlike Asterisk (nothing) or Twilio
(self-managed, and pricier for India-domestic calls).

Run this service, expose it publicly (a real deployment, or `ngrok http
8090` for local testing -- Exotel's cloud needs to reach this over the
internet, unlike Asterisk which could run on the same LAN as the softphones),
and point an Exotel Voicebot Applet at wss://<public-host>/?sample-rate=8000.
It will:
  1. Play the consent disclosure as soon as the call connects.
  2. Buffer caller audio, use VAD to detect when they've finished a turn.
  3. Transcribe -> retrieve KB context -> generate a reply -> speak it back.
  4. Repeat until the caller hangs up.

Barge-in: a background task reads every incoming audio frame for the whole
life of the call (not just while "listening"), so it can tell the caller
started talking even while the bot is mid-reply. When that happens, the
bot's current reply is cut off immediately (and an explicit "clear" event
tells Exotel to drop whatever audio it has already buffered from us, rather
than playing out a queued backlog after we've stopped sending) instead of
blindly finishing the whole scripted line.

Protocol reference (Exotel's AgentStream WebSocket protocol -- confirmed via
Exotel's own published protocol doc and the pipecat-ai project's
ExotelFrameSerializer, both independent of this codebase; the parsing below
is deliberately defensive about exact field casing since neither source was
directly testable before a real call, and logs any event it doesn't
recognize so the parsing can be corrected against real traffic if needed):
  Exotel -> bot, one JSON object per WebSocket text message:
    {"event": "connected"}                                   -- handshake
    {"event": "start", "start": {"streamSid": ..., "callSid": ...}, ...}
    {"event": "media", "media": {"payload": "<base64 PCM>"}}
    {"event": "dtmf", "dtmf": {"digit": "..."}}
    {"event": "stop"}                                        -- call ended
  bot -> Exotel:
    {"event": "media", "stream_sid": ..., "media": {"payload": "<base64 PCM>"}}
    {"event": "clear", "stream_sid": ...}                    -- barge-in
  Audio is raw 16-bit linear PCM mono (not mu-law), sample rate configured
  via the `?sample-rate=` query param on the WebSocket URL Exotel connects
  to (this bridge expects 8000, matching AudioSocket's old fixed rate, so
  none of the VAD/resampling logic below needed to change).

One asyncio task pair per call (a reader task feeding VAD/turn-detection,
and the main conversation task) -- fine for testing a handful of concurrent
calls; pipeline.py's blocking network calls (STT/LLM/TTS) run via
asyncio.to_thread() so they don't stall the event loop for other calls.
"""

import asyncio
import base64
import json
import math
import time

import numpy as np
import webrtcvad
import websockets
from scipy.signal import resample_poly

import pipeline
from programme_config import CONSENT_DISCLOSURE

HOST = "0.0.0.0"
PORT = 8090

SAMPLE_RATE = 8000          # requested via ?sample-rate=8000 in the Exotel Voicebot Applet config
FRAME_BYTES = 320           # 20ms of 8kHz 16-bit mono PCM
FRAME_MS = 20
SILENCE_MS_TO_END_TURN = 800
# Lowered again from 8s -- real test calls kept hitting this full cap even on
# short replies like "thank you"/"bye", pointing to background noise keeping
# the VAD's silence timer from ever resetting. Hosted Whisper doesn't have a
# vad_filter equivalent to fall back on (unlike the old local setup), so a
# long noisy capture doesn't just come back empty -- it hallucinates a full,
# wrong-language sentence. A shorter cap bounds how much noise it can be
# asked to make sense of.
MAX_UTTERANCE_MS = 5_000

# Minimum peak amplitude (int16 scale, max 32767) for a captured turn to even
# be sent to Whisper. Real test calls showed a clean, consistent split:
# genuine speech always peaked at 17000+, while near-silent/background-noise
# captures (peak under ~600) still got transcribed as confident-sounding text
# like "Thank you." -- a well-known Whisper failure mode (hallucinating stock
# phrases on silence, likely from YouTube-heavy training data). The script-
# based hallucination filter in pipeline.py can't catch this since it's valid
# English, not gibberish script. Skipping the STT call outright below this
# threshold is safe given the size of the gap observed, and saves an API call.
MIN_SPEECH_PEAK = 3000

# How many consecutive speech frames are needed before treating it as a
# barge-in (interrupting the bot's current reply). Higher than the 1-frame
# threshold used for normal turn-taking, since a false trip here cuts the bot
# off mid-sentence for a stray noise blip rather than just starting to listen
# a little early.
BARGE_IN_SPEECH_FRAMES = 12  # 240ms

# Ignore barge-in entirely for the first stretch of each reply. A genuine
# caller interruption doesn't happen in the very first fraction of a second
# of the bot starting to talk.
BARGE_IN_GRACE_MS = 500

vad = webrtcvad.Vad(2)  # aggressiveness 0-3 -- 2 is the middle ground; 3 was too strict
                         # and missed real speech entirely in real testing.


def resample(int16_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return int16_array.astype(np.float32)
    g = math.gcd(orig_sr, target_sr)
    resampled = resample_poly(int16_array.astype(np.float32), target_sr // g, orig_sr // g)
    return np.clip(resampled, -32768, 32767)


class CallState:
    """Shared state between the per-call reader task (owns writes to the
    turn-buffering fields, driven by incoming "media" events) and the main
    conversation task (only reads them, after turn_ready fires). A single
    asyncio task reads the websocket at a time, so no lock is needed the way
    the old thread-based version required."""

    def __init__(self):
        self.stream_sid = None
        self.call_id = "unknown"
        self.speech_frames = []
        self.silence_run_ms = 0
        self.started_speaking = False
        self.utterance_ms = 0
        self.consecutive_speech_frames = 0
        self.started = asyncio.Event()    # the "start" event has been parsed (stream_sid/call_id ready)
        self.turn_ready = asyncio.Event()  # a full turn (utterance) is ready for STT
        self.barge_in = asyncio.Event()    # caller started talking -- cut off the current reply
        self.hangup = asyncio.Event()


def handle_audio_frame(frame: bytes, state: CallState):
    """Runs for every incoming audio frame for the whole life of the call
    (not just while "listening") -- this is what makes barge-in possible."""
    if len(frame) != FRAME_BYTES:
        return
    is_speech = vad.is_speech(frame, SAMPLE_RATE)

    # A completed turn is already waiting for the main task to drain it
    # (e.g. it's still busy running STT on the previous turn) -- stop
    # growing this buffer further, or a slow main task lets it balloon well
    # past MAX_UTTERANCE_MS.
    if state.turn_ready.is_set():
        if is_speech:
            state.consecutive_speech_frames += 1
            if state.consecutive_speech_frames >= BARGE_IN_SPEECH_FRAMES:
                state.barge_in.set()
        else:
            state.consecutive_speech_frames = 0
        return

    if is_speech:
        state.consecutive_speech_frames += 1
        state.speech_frames.append(frame)
        state.silence_run_ms = 0
        state.started_speaking = True
        state.utterance_ms += FRAME_MS
        if state.consecutive_speech_frames >= BARGE_IN_SPEECH_FRAMES:
            state.barge_in.set()
    else:
        state.consecutive_speech_frames = 0
        if state.started_speaking:
            # keep a little trailing silence, sounds more natural
            state.speech_frames.append(frame)
            state.silence_run_ms += FRAME_MS
            state.utterance_ms += FRAME_MS

    turn_done = state.started_speaking and (
        state.silence_run_ms >= SILENCE_MS_TO_END_TURN
        or state.utterance_ms >= MAX_UTTERANCE_MS
    )
    if turn_done:
        state.turn_ready.set()


async def reader_task(ws, state: CallState):
    """Reads every incoming WebSocket message for the whole life of the
    call, parses Exotel's protocol events, and feeds audio into VAD/turn
    detection. Runs concurrently with the main conversation task below."""
    try:
        async for raw_msg in ws:
            try:
                msg = json.loads(raw_msg)
            except (TypeError, ValueError):
                continue
            event = msg.get("event")

            if event == "media":
                payload = msg.get("media", {}).get("payload")
                if payload:
                    handle_audio_frame(base64.b64decode(payload), state)
                continue

            if event == "start":
                start = msg.get("start", {})
                state.stream_sid = (
                    start.get("streamSid") or start.get("stream_sid")
                    or msg.get("streamSid") or msg.get("stream_sid")
                )
                state.call_id = start.get("callSid") or start.get("call_sid") or "unknown"
                state.started.set()
                continue

            if event == "stop":
                state.hangup.set()
                state.turn_ready.set()  # wake the main task so it can exit
                break

            if event in ("connected", "dtmf", "mark"):
                continue

            # An event this parser doesn't recognize -- log the raw message
            # rather than silently ignoring it, since Exotel's exact field
            # names/casing weren't directly testable before a real call.
            print(f"[unhandled event {event!r}] {raw_msg[:300]}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        state.hangup.set()
        state.turn_ready.set()


async def send_audio_frame(ws, stream_sid, frame: bytes):
    if len(frame) < FRAME_BYTES:
        frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
    await ws.send(json.dumps({
        "event": "media",
        "stream_sid": stream_sid,
        "media": {"payload": base64.b64encode(frame).decode("ascii")},
    }))


async def send_clear(ws, stream_sid):
    await ws.send(json.dumps({"event": "clear", "stream_sid": stream_sid}))


async def speak(ws, text: str, state: CallState) -> bool:
    """Synthesizes and plays `text`, stopping early if the caller starts
    talking (barge-in). Returns True if the reply was cut short."""
    t0 = time.time()
    pcm_native, native_rate = await asyncio.to_thread(pipeline.synthesize_pcm, text)
    pcm_8k = resample(pcm_native, native_rate, SAMPLE_RATE).astype(np.int16)
    synth_elapsed = time.time() - t0
    print(f"  [TTS {synth_elapsed:.2f}s] \"{text}\"")

    state.barge_in.clear()
    raw = pcm_8k.astype("<i2").tobytes()
    grace_frames = BARGE_IN_GRACE_MS // FRAME_MS
    for frame_idx, i in enumerate(range(0, len(raw), FRAME_BYTES)):
        if frame_idx >= grace_frames and state.barge_in.is_set():
            print("  [TTS] interrupted -- caller started talking")
            await send_clear(ws, state.stream_sid)
            return True
        await send_audio_frame(ws, state.stream_sid, raw[i:i + FRAME_BYTES])
        await asyncio.sleep(FRAME_MS / 1000.0)
    return False


async def run_call(ws):
    state = CallState()
    reader = asyncio.create_task(reader_task(ws, state))

    await state.started.wait()
    call_id = state.call_id
    print(f"[call {call_id}] connected")

    chat_history = []
    result = None
    try:
        await speak(ws, CONSENT_DISCLOSURE, state)
        while not state.hangup.is_set():
            await state.turn_ready.wait()
            if state.hangup.is_set():
                print(f"[call {call_id}] hangup")
                break

            pcm_8k = np.frombuffer(b"".join(state.speech_frames), dtype="<i2")
            captured_ms = state.utterance_ms
            state.speech_frames = []
            state.silence_run_ms = 0
            state.started_speaking = False
            state.utterance_ms = 0
            state.consecutive_speech_frames = 0
            state.turn_ready.clear()

            # Peak/RMS of the raw captured audio -- printed alongside every
            # STT result so a run of empty transcriptions can be told apart
            # from a genuine mic/gain problem versus audio that's actually
            # there but Whisper still can't use.
            peak = int(np.abs(pcm_8k).max()) if len(pcm_8k) else 0
            rms = float(np.sqrt(np.mean(pcm_8k.astype(np.float64) ** 2))) if len(pcm_8k) else 0.0

            if peak < MIN_SPEECH_PEAK:
                print(f"[call {call_id}] [STT skipped, captured {captured_ms}ms, "
                      f"peak={peak} rms={rms:.0f}] too quiet to be real speech -- not calling Whisper")
                continue

            t0 = time.time()
            pcm_16k_f32 = resample(pcm_8k, SAMPLE_RATE, 16000) / 32768.0
            caller_text = await asyncio.to_thread(pipeline.transcribe_pcm, pcm_16k_f32)
            stt_elapsed = time.time() - t0
            if not caller_text:
                print(f"[call {call_id}] [STT {stt_elapsed:.2f}s, captured {captured_ms}ms, "
                      f"peak={peak} rms={rms:.0f}] heard nothing usable -- check mic input / VAD sensitivity")
                continue
            print(f"[call {call_id}] [STT {stt_elapsed:.2f}s, captured {captured_ms}ms, "
                  f"peak={peak} rms={rms:.0f}] \"{caller_text}\"")

            result = await asyncio.to_thread(pipeline.generate_response, caller_text, chat_history)
            print(f"[call {call_id}] [LLM {result['elapsed']:.2f}s] "
                  f"({'SUPPRESSED' if result['suppressed'] else 'reply'}) \"{result['reply']}\"")
            chat_history.append({"role": "user", "content": caller_text})
            chat_history.append({"role": "assistant", "content": result["reply"]})

            await speak(ws, result["reply"], state)
            if result["suppressed"]:
                await asyncio.sleep(0.5)
                break
    except websockets.exceptions.ConnectionClosed:
        print(f"[call {call_id}] connection dropped")
    finally:
        state.hangup.set()
        reader.cancel()
        print(f"[call {call_id}] closed")


if __name__ == "__main__":
    async def main():
        print(f"Exotel AgentStream bridge listening on {HOST}:{PORT}")
        print("Point an Exotel Voicebot Applet at wss://<public-host>/?sample-rate=8000")
        print("(use `ngrok http 8090` or similar to expose this locally for testing).")
        async with websockets.serve(run_call, HOST, PORT):
            await asyncio.Future()  # run forever

    asyncio.run(main())
