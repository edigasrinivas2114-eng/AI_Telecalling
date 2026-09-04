"""Asterisk AudioSocket <-> STT/RAG/LLM/TTS bridge.

Run this alongside Asterisk. Point an AudioSocket() dialplan app at this
service's host:port (see asterisk_config/extensions_snippet.conf) and it will:
  1. Play the consent disclosure as soon as the call connects.
  2. Buffer caller audio, use VAD to detect when they've finished a turn.
  3. Transcribe -> retrieve KB context -> generate a reply -> speak it back.
  4. Repeat until the caller hangs up.

Protocol reference: Asterisk AudioSocket sends/expects messages of
[1-byte type][2-byte big-endian length][payload]:
  0x01 = UUID (16 bytes, sent once at call start)
  0x10 = audio (320 bytes = 20ms of 8kHz 16-bit mono PCM)
  0x00 = hangup/terminate (0-length payload)

One thread per call -- fine for testing a handful of concurrent calls.
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
MAX_UTTERANCE_MS = 15_000

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

vad = webrtcvad.Vad(2)  # aggressiveness 0-3; 2 is a reasonable default for phone audio


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


def send_audio(sock: socket.socket, pcm_8k_int16: np.ndarray, lock: threading.Lock):
    """Send int16 8kHz mono PCM as a sequence of 320-byte AudioSocket frames,
    paced roughly in real time so Asterisk plays it back naturally."""
    raw = pcm_8k_int16.astype("<i2").tobytes()
    for i in range(0, len(raw), FRAME_BYTES):
        send_audio_frame(sock, raw[i:i + FRAME_BYTES], lock)
        time.sleep(FRAME_MS / 1000.0)


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


def speak(sock: socket.socket, text: str, lock: threading.Lock):
    print(f"  [TTS] \"{text}\"")
    pcm_native, native_rate = pipeline.synthesize_pcm(text)
    pcm_8k = resample(pcm_native, native_rate, SAMPLE_RATE).astype(np.int16)
    send_audio(sock, pcm_8k, lock)


class CallHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        msg_type, payload = recv_message(sock)
        call_id = str(uuid.UUID(bytes=payload)) if msg_type == MSG_UUID and len(payload) == 16 else "unknown"
        print(f"[call {call_id}] connected")

        write_lock = threading.Lock()
        chat_history = []
        speak(sock, CONSENT_DISCLOSURE, write_lock)

        speech_frames = []
        silence_run_ms = 0
        started_speaking = False
        utterance_ms = 0

        try:
            while True:
                msg_type, payload = recv_message(sock)
                if msg_type is None or msg_type == MSG_TERMINATE:
                    print(f"[call {call_id}] hangup")
                    break
                if msg_type != MSG_AUDIO or len(payload) != FRAME_BYTES:
                    continue

                is_speech = vad.is_speech(payload, SAMPLE_RATE)
                if is_speech:
                    speech_frames.append(payload)
                    silence_run_ms = 0
                    started_speaking = True
                    utterance_ms += FRAME_MS
                elif started_speaking:
                    speech_frames.append(payload)  # keep a little trailing silence, sounds more natural
                    silence_run_ms += FRAME_MS
                    utterance_ms += FRAME_MS

                turn_done = started_speaking and (
                    silence_run_ms >= SILENCE_MS_TO_END_TURN or utterance_ms >= MAX_UTTERANCE_MS
                )
                if not turn_done:
                    continue

                pcm_8k = np.frombuffer(b"".join(speech_frames), dtype="<i2")
                speech_frames, silence_run_ms, started_speaking, utterance_ms = [], 0, False, 0

                keepalive_stop = threading.Event()
                keepalive_thread = threading.Thread(
                    target=run_keepalive, args=(sock, write_lock, keepalive_stop), daemon=True
                )
                keepalive_thread.start()
                try:
                    t0 = time.time()
                    pcm_16k_f32 = resample(pcm_8k, SAMPLE_RATE, 16000) / 32768.0
                    caller_text = pipeline.transcribe_pcm(pcm_16k_f32)
                    stt_elapsed = time.time() - t0
                    if not caller_text:
                        continue
                    print(f"[call {call_id}] [STT {stt_elapsed:.2f}s] \"{caller_text}\"")

                    result = pipeline.generate_response(caller_text, chat_history=chat_history)
                    print(f"[call {call_id}] [LLM {result['elapsed']:.2f}s] "
                          f"({'SUPPRESSED' if result['suppressed'] else 'reply'}) \"{result['reply']}\"")
                    chat_history.append({"role": "user", "content": caller_text})
                    chat_history.append({"role": "assistant", "content": result["reply"]})
                finally:
                    keepalive_stop.set()
                    keepalive_thread.join()

                speak(sock, result["reply"], write_lock)
                if result["suppressed"]:
                    time.sleep(0.5)
                    break
        except (ConnectionResetError, BrokenPipeError):
            print(f"[call {call_id}] connection dropped")
        finally:
            print(f"[call {call_id}] closed")


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print(f"AudioSocket bridge listening on {HOST}:{PORT}")
    print("Point your Asterisk dialplan's AudioSocket() app at this host:port.")
    server = ThreadingTCPServer((HOST, PORT), CallHandler)
    server.serve_forever()
