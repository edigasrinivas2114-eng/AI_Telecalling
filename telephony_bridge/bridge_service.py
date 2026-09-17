"""Asterisk AudioSocket <-> STT/RAG/LLM/TTS bridge.

Run this alongside Asterisk. Point an AudioSocket() dialplan app at this
service's host:port (see asterisk_config/extensions_snippet.conf) and it will:
  1. Play the consent disclosure as soon as the call connects.
  2. Buffer caller audio, use VAD to detect when they've finished a turn.
  3. Transcribe -> retrieve KB context -> generate a reply -> speak it back.
  4. Repeat until the caller hangs up.

Barge-in: a single background thread reads every incoming audio frame for the
whole life of the call (not just while "listening"), so it can tell the
caller started talking even while the bot is mid-reply. When that happens,
the bot's current reply is cut off immediately instead of blindly finishing
the whole scripted line -- the caller's speech that triggered the interrupt
is already being captured as the start of their next turn.

Protocol reference: Asterisk AudioSocket sends/expects messages of
[1-byte type][2-byte big-endian length][payload]:
  0x01 = UUID (16 bytes, sent once at call start)
  0x10 = audio (320 bytes = 20ms of 8kHz 16-bit mono PCM)
  0x00 = hangup/terminate (0-length payload)

One thread per call, plus one audio-reader thread per call -- fine for
testing a handful of concurrent calls.
"""

import math
import socket
import socketserver
import struct
import threading
import time
import uuid

import numpy as np
import webrtcvad
from scipy.signal import resample_poly

import pipeline
from programme_config import CONSENT_DISCLOSURE

HOST = "0.0.0.0"
PORT = 8090

SAMPLE_RATE = 8000          # AudioSocket's fixed rate
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
# based hallucination filter below can't catch this since it's valid English,
# not gibberish script. Skipping the STT call outright below this threshold
# is safe given the size of the gap observed, and saves an API call too.
MIN_SPEECH_PEAK = 3000

# How many consecutive speech frames are needed before treating it as a
# barge-in (interrupting the bot's current reply). Higher than the 1-frame
# threshold used for normal turn-taking, since a false trip here cuts the bot
# off mid-sentence for a stray noise blip rather than just starting to listen
# a little early. Raised from 4 (80ms) -- real testing showed every single
# reply getting cut off almost instantly, which pointed to acoustic echo
# (the test device's mic hearing its own speaker) rather than genuine
# interruptions; a real fix for that is testing with a headset, but this adds
# some margin regardless, since even genuine interruptions don't need an
# 80ms trigger.
BARGE_IN_SPEECH_FRAMES = 12  # 240ms

# Ignore barge-in entirely for the first stretch of each reply. A genuine
# caller interruption doesn't happen in the very first fraction of a second
# of the bot starting to talk -- real test calls showed every single reply
# getting cut short almost immediately, which BARGE_IN_SPEECH_FRAMES alone
# didn't fix (acoustic echo/ambient noise can still rack up 240ms of
# VAD-flagged "speech" within a second or two of playback starting). This
# grace period is a second line of defense on top of that, not a substitute
# for testing with a headset if echo turns out to be the actual cause.
BARGE_IN_GRACE_MS = 500

# Asterisk's AudioSocket app kills the call after ~2s of the bridge sending
# nothing back, regardless of whether the call is otherwise still alive. STT+LLM
# on CPU routinely take longer than that, so a keepalive thread sends silent
# frames at this interval (well under the 2s cutoff) while a reply is being
# generated, and stops as soon as the real reply audio is ready to send.
KEEPALIVE_INTERVAL_S = 0.5

MSG_TERMINATE = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10

vad = webrtcvad.Vad(2)  # aggressiveness 0-3 -- reverted from 3: real testing showed max
                         # strictness stopped picking up real speech at all ("its not getting my
                         # voice"), which is a worse failure mode for a phone bot than an
                         # occasional false trigger from noise. 2 is the middle ground; the buffer
                         # overrun bug (captured audio exceeding MAX_UTTERANCE_MS) fixed alongside
                         # this was likely a bigger contributor to the STT slowness than VAD level
                         # was anyway.


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def recv_message(sock: socket.socket):
    header = recv_exact(sock, 3)
    if len(header) < 3:
        return None, None
    msg_type = header[0]
    (length,) = struct.unpack(">H", header[1:3])
    payload = recv_exact(sock, length) if length else b""
    return msg_type, payload


def send_audio_frame(sock: socket.socket, frame: bytes, lock: threading.Lock):
    if len(frame) < FRAME_BYTES:
        frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
    with lock:
        sock.sendall(bytes([MSG_AUDIO]) + struct.pack(">H", len(frame)) + frame)


def run_keepalive(sock: socket.socket, lock: threading.Lock, stop_event: threading.Event):
    """Sends silent audio frames until stop_event is set, to keep Asterisk's
    AudioSocket inactivity timeout from killing the call while we're busy
    running STT/LLM/TTS."""
    silence = b"\x00" * FRAME_BYTES
    while not stop_event.wait(KEEPALIVE_INTERVAL_S):
        try:
            send_audio_frame(sock, silence, lock)
        except (ConnectionResetError, BrokenPipeError, OSError):
            return


def resample(int16_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return int16_array.astype(np.float32)
    g = math.gcd(orig_sr, target_sr)
    resampled = resample_poly(int16_array.astype(np.float32), target_sr // g, orig_sr // g)
    return np.clip(resampled, -32768, 32767)


class CallState:
    """Shared, lock-protected state between the per-call audio-reader thread
    and the main call-handling thread. The reader thread owns writes to the
    turn-buffering fields; the main thread only reads them (after acquiring
    the lock) when turn_ready fires."""

    def __init__(self):
        self.lock = threading.Lock()
        self.speech_frames = []
        self.silence_run_ms = 0
        self.started_speaking = False
        self.utterance_ms = 0
        self.consecutive_speech_frames = 0
        self.turn_ready = threading.Event()   # a full turn (utterance) is ready for STT
        self.barge_in = threading.Event()     # caller started talking -- cut off the current reply
        self.hangup = threading.Event()


def audio_reader(sock: socket.socket, state: CallState, call_id: str):
    """Runs for the whole life of the call, not just while "listening" --
    this is what makes barge-in possible: the caller's speech is captured and
    turn-detected continuously, whether the bot is talking or silent."""
    try:
        while not state.hangup.is_set():
            msg_type, payload = recv_message(sock)
            if msg_type is None or msg_type == MSG_TERMINATE:
                state.hangup.set()
                state.turn_ready.set()  # wake the main thread so it can exit
                break
            if msg_type != MSG_AUDIO or len(payload) != FRAME_BYTES:
                continue

            is_speech = vad.is_speech(payload, SAMPLE_RATE)
            with state.lock:
                # A completed turn is already waiting for the main thread to
                # drain it (e.g. it's still busy running STT on the previous
                # turn) -- stop growing this buffer further, or a slow main
                # thread lets it balloon well past MAX_UTTERANCE_MS.
                if state.turn_ready.is_set():
                    if is_speech:
                        state.consecutive_speech_frames += 1
                        if state.consecutive_speech_frames >= BARGE_IN_SPEECH_FRAMES:
                            state.barge_in.set()
                    else:
                        state.consecutive_speech_frames = 0
                    continue

                if is_speech:
                    state.consecutive_speech_frames += 1
                    state.speech_frames.append(payload)
                    state.silence_run_ms = 0
                    state.started_speaking = True
                    state.utterance_ms += FRAME_MS
                    if state.consecutive_speech_frames >= BARGE_IN_SPEECH_FRAMES:
                        state.barge_in.set()
                else:
                    state.consecutive_speech_frames = 0
                    if state.started_speaking:
                        # keep a little trailing silence, sounds more natural
                        state.speech_frames.append(payload)
                        state.silence_run_ms += FRAME_MS
                        state.utterance_ms += FRAME_MS

                turn_done = state.started_speaking and (
                    state.silence_run_ms >= SILENCE_MS_TO_END_TURN
                    or state.utterance_ms >= MAX_UTTERANCE_MS
                )
                if turn_done:
                    state.turn_ready.set()
    except (ConnectionResetError, BrokenPipeError, OSError):
        state.hangup.set()
        state.turn_ready.set()


def speak(sock: socket.socket, text: str, lock: threading.Lock, state: CallState) -> bool:
    """Synthesizes and plays `text`, stopping early if the caller starts
    talking (barge-in). Returns True if the reply was cut short."""
    t0 = time.time()
    # synthesize_pcm() is a slow network round-trip (Gemini TTS has taken
    # 6-12+ seconds in real testing) -- Asterisk's AudioSocket app kills the
    # call after ~2s of the bridge sending nothing back, so without a
    # keepalive running here too, a slow synthesis call gets the connection
    # closed out from under it before the first real audio frame is even
    # sent (this crashed the very first greeting with a BrokenPipeError on a
    # real test call). The main loop's own keepalive only covers STT/LLM,
    # not this call, and the initial greeting's speak() call has no
    # surrounding keepalive at all -- so this needs to be self-contained.
    keepalive_stop = threading.Event()
    keepalive_thread = threading.Thread(
        target=run_keepalive, args=(sock, lock, keepalive_stop), daemon=True
    )
    keepalive_thread.start()
    try:
        pcm_native, native_rate = pipeline.synthesize_pcm(text)
    finally:
        keepalive_stop.set()
        keepalive_thread.join()
    pcm_8k = resample(pcm_native, native_rate, SAMPLE_RATE).astype(np.int16)
    synth_elapsed = time.time() - t0
    print(f"  [TTS {synth_elapsed:.2f}s] \"{text}\"")

    state.barge_in.clear()
    raw = pcm_8k.astype("<i2").tobytes()
    grace_frames = BARGE_IN_GRACE_MS // FRAME_MS
    for frame_idx, i in enumerate(range(0, len(raw), FRAME_BYTES)):
        if frame_idx >= grace_frames and state.barge_in.is_set():
            print("  [TTS] interrupted -- caller started talking")
            return True
        send_audio_frame(sock, raw[i:i + FRAME_BYTES], lock)
        time.sleep(FRAME_MS / 1000.0)
    return False


class CallHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        msg_type, payload = recv_message(sock)
        call_id = str(uuid.UUID(bytes=payload)) if msg_type == MSG_UUID and len(payload) == 16 else "unknown"
        print(f"[call {call_id}] connected")

        write_lock = threading.Lock()
        chat_history = []
        state = CallState()

        reader_thread = threading.Thread(target=audio_reader, args=(sock, state, call_id), daemon=True)
        reader_thread.start()

        try:
            speak(sock, CONSENT_DISCLOSURE, write_lock, state)
            while not state.hangup.is_set():
                state.turn_ready.wait()
                if state.hangup.is_set():
                    print(f"[call {call_id}] hangup")
                    break

                with state.lock:
                    pcm_8k = np.frombuffer(b"".join(state.speech_frames), dtype="<i2")
                    captured_ms = state.utterance_ms
                    state.speech_frames = []
                    state.silence_run_ms = 0
                    state.started_speaking = False
                    state.utterance_ms = 0
                    state.consecutive_speech_frames = 0
                    state.turn_ready.clear()

                keepalive_stop = threading.Event()
                keepalive_thread = threading.Thread(
                    target=run_keepalive, args=(sock, write_lock, keepalive_stop), daemon=True
                )
                keepalive_thread.start()
                try:
                    # Peak/RMS of the raw captured audio -- printed alongside every
                    # STT result so a run of empty transcriptions can be told apart
                    # from a genuine mic/gain problem (peak near 0 on int16's
                    # -32768..32767 range) versus audio that's actually there but
                    # Whisper still can't use (peak/RMS in a normal range).
                    peak = int(np.abs(pcm_8k).max()) if len(pcm_8k) else 0
                    rms = float(np.sqrt(np.mean(pcm_8k.astype(np.float64) ** 2))) if len(pcm_8k) else 0.0

                    if peak < MIN_SPEECH_PEAK:
                        print(f"[call {call_id}] [STT skipped, captured {captured_ms}ms, "
                              f"peak={peak} rms={rms:.0f}] too quiet to be real speech -- not calling Whisper")
                        continue

                    t0 = time.time()
                    pcm_16k_f32 = resample(pcm_8k, SAMPLE_RATE, 16000) / 32768.0
                    caller_text = pipeline.transcribe_pcm(pcm_16k_f32)
                    stt_elapsed = time.time() - t0
                    if not caller_text:
                        print(f"[call {call_id}] [STT {stt_elapsed:.2f}s, captured {captured_ms}ms, "
                              f"peak={peak} rms={rms:.0f}] heard nothing usable -- check mic input / VAD sensitivity")
                        continue
                    print(f"[call {call_id}] [STT {stt_elapsed:.2f}s, captured {captured_ms}ms, "
                          f"peak={peak} rms={rms:.0f}] \"{caller_text}\"")

                    result = pipeline.generate_response(caller_text, chat_history=chat_history)
                    print(f"[call {call_id}] [LLM {result['elapsed']:.2f}s] "
                          f"({'SUPPRESSED' if result['suppressed'] else 'reply'}) \"{result['reply']}\"")
                    chat_history.append({"role": "user", "content": caller_text})
                    chat_history.append({"role": "assistant", "content": result["reply"]})
                finally:
                    keepalive_stop.set()
                    keepalive_thread.join()

                speak(sock, result["reply"], write_lock, state)
                if result["suppressed"]:
                    time.sleep(0.5)
                    break
        except (ConnectionResetError, BrokenPipeError):
            print(f"[call {call_id}] connection dropped")
        finally:
            state.hangup.set()
            print(f"[call {call_id}] closed")


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print(f"AudioSocket bridge listening on {HOST}:{PORT}")
    print("Point your Asterisk dialplan's AudioSocket() app at this host:port.")
    server = ThreadingTCPServer((HOST, PORT), CallHandler)
    server.serve_forever()
