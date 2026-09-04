# Phase 2 (testing): AI telecaller over a real phone call, $0 cost

This connects the same STT -> RAG -> LLM -> TTS idea from `ai_telecaller_poc.ipynb`
to an actual phone call, using only free, self-hosted software. It does **not**
need a real phone number: you register two free softphone apps to your own
Asterisk server and call one from the other -- one call reaches the AI.

Runs on your own machine or a free-tier VM (e.g. Oracle Cloud Always Free) --
**not** in Colab (Colab isn't reachable from the internet and isn't always-on,
which a live call bridge needs).

Why the LLM is different here than in the notebook: `bitsandbytes` 4-bit
quantization needs a CUDA GPU. An always-on, free, self-hosted bridge won't
have a GPU sitting idle for it, so this uses **Ollama** instead, which runs
quantized models efficiently on CPU. Swap `OLLAMA_MODEL` in `pipeline.py` for
a bigger model once you move this to a GPU host.

Why TTS is different here than in the notebook: Piper's Telugu voices sound
noticeably synthetic. This uses **edge-tts** instead -- free access to
Microsoft's production neural voices (the same ones Azure sells), reached via
Microsoft Edge's "Read aloud" service, no API key or Azure account needed.
Trade-off: each reply needs live internet access (Piper runs fully offline).

## What's in this folder

- `bridge_service.py` -- the AudioSocket server: bridges live call audio to the
  STT/RAG/LLM/TTS pipeline.
- `pipeline.py` -- STT (faster-whisper), RAG (Chroma + sentence-transformers),
  LLM (Ollama), TTS (edge-tts / Microsoft neural voices).
- `programme_config.py` -- the same editable programme pitch variables as the
  notebook. Edit the values here too.
- `asterisk_config/pjsip_snippet.conf` -- two test SIP extensions (1000, 1001).
- `asterisk_config/extensions_snippet.conf` -- dialplan: 1000 <-> 1001 can call
  each other normally; either can dial 8000 to reach the AI.
- `requirements.txt` -- Python dependencies for this service.

## Setup

### 1. Install Asterisk

On Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y asterisk
```
(Or use the official Docker image if you'd rather not install it directly on
the host -- either works, this component doesn't have the "no Docker"
constraint the Colab POC had.)

### 2. Configure Asterisk

Append the contents of `asterisk_config/pjsip_snippet.conf` to
`/etc/asterisk/pjsip.conf`, and the contents of
`asterisk_config/extensions_snippet.conf` to `/etc/asterisk/extensions.conf`.

**Change the two passwords** (`changeme1000`, `changeme1001`) in
`pjsip_snippet.conf` before using this anywhere network-reachable.

Then reload:
```bash
sudo asterisk -rx "pjsip reload"
sudo asterisk -rx "dialplan reload"
```

### 3. Install Ollama and pull a small model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b-instruct
```
Leave `ollama serve` running (the install script sets it up as a systemd
service that starts automatically -- check with `systemctl status ollama`).

### 4. Install ffmpeg (needed to decode edge-tts's audio)

```bash
sudo apt-get install -y ffmpeg
```

### 5. Install Python dependencies for the bridge

```bash
cd telephony_bridge
python3 -m venv venv
source venv/bin/activate
# CPU-only torch build first, to avoid pulling multi-GB CUDA packages you won't use:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

No voice file to download this time -- edge-tts fetches the voice live over the network on
each call, using the voice name set in `pipeline.py` (`EDGE_TTS_VOICE`, currently
`te-IN-MohanNeural`; swap to `te-IN-ShrutiNeural` for a female voice, or any other
[Edge TTS voice name](https://github.com/rany2/edge-tts) for a different language).

### 6. Edit the programme details

Open `programme_config.py` and fill in the real `PROGRAMME_*` / `CERTIFICATION_NAME`
/ `COMPANY_NAME` values (same as you did in the notebook).

### 7. Start the bridge service

```bash
python3 bridge_service.py
```
You should see: `AudioSocket bridge listening on 0.0.0.0:8090`

### 8. Install two softphones and test

Install [Zoiper](https://www.zoiper.com/) or [Linphone](https://www.linphone.org/)
(free) on your phone or laptop -- twice, or on two different devices.

Register:
- Softphone A: account `1000`, password `changeme1000`, server = your Asterisk box's IP
- Softphone B: account `1001`, password `changeme1001`, server = your Asterisk box's IP

From softphone A, **dial `8000`**. You should hear the AI's consent disclosure,
then be able to talk to it -- ask about the fee, the dates, say "remove my
number," etc., same test cases as the notebook's harness. Watch the terminal
running `bridge_service.py` for STT/LLM timing and transcripts.

## Known limitations of this first version

- TTS needs live internet access (edge-tts calls out to Microsoft's service per
  reply) -- unlike Piper, it won't work fully offline. It's also an unofficial
  (if long-stable, widely used) way of reaching that service, not a supported
  public API -- worth knowing if this ever needs a guaranteed SLA.
- No barge-in: the AI finishes speaking before it listens again (talking over
  it won't interrupt it).
- One thread per call -- fine for testing a handful of calls, not for scale.
- VAD-based turn detection is a fixed silence timeout (`SILENCE_MS_TO_END_TURN`
  in `bridge_service.py`), not adaptive -- tune it if it cuts callers off too
  eagerly or waits too long.
- No outbound dialing, no answering-machine detection, no post-call
  classification pass yet -- those are the next steps once this wiring is
  confirmed working.
