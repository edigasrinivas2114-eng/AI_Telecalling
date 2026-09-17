# Phase 2 (testing): AI telecaller over a real phone call

This connects the same STT -> RAG -> LLM -> TTS idea from `ai_telecaller_poc.ipynb`
to an actual phone call, using **Exotel** as the telephony provider. Unlike the
earlier Asterisk-based version of this bridge, this needs a real (if trial/free)
Exotel account and phone number -- but you get a real phone number to call, and
Exotel handles all the actual SIP/RTP telephony plumbing, which is what a
self-hosted Asterisk setup kept getting wrong (see "Why the telephony layer is
different" below).

Runs on your own machine or a free-tier VM (e.g. Oracle Cloud Always Free) --
**not** in Colab (Colab isn't reachable from the internet and isn't always-on,
which a live call bridge needs). Whichever machine it runs on needs to be
reachable over the public internet, since Exotel's cloud platform connects to
it directly -- a real deployment, or a tunnel like `ngrok` for local testing
(see setup step 5).

Why the telephony layer is different here than earlier versions of this file
described: this originally ran on a self-hosted Asterisk server, with two free
softphone apps registered to it as test extensions (no real phone number
needed). That repeatedly hit "one-way audio" -- the caller's mic reached the
bridge fine (proven by working speech-to-text), but the bot's replies never
reached the caller's ear, confirmed on two different audio setups including
AirPods (which rules out acoustic echo/laptop-speaker issues). That combination
pointed at Asterisk's own SIP/RTP/NAT handling, not this code -- a well-known,
hard-to-debug class of problem with self-hosted PBX setups. **Exotel** replaces
Asterisk entirely: it's a cloud telephony provider that terminates the actual
phone call on its own infrastructure and streams call audio to this bridge over
a plain WebSocket (its **AgentStream** / Voicebot Applet feature), so there's no
local SIP/RTP/NAT configuration to get wrong. It's also India-focused (this
business's actual market) with built-in DLT/TRAI compliance handling, which
this project will need eventually anyway to call real leads -- unlike Asterisk
(no compliance tooling at all) or Twilio (mature and well-documented, but
pricier for India-domestic calls and DLT compliance is entirely self-managed).

Why the LLM is different here than in the notebook: this originally used Ollama
running a small local model (qwen2.5:3b-instruct) on CPU, avoiding any cost or API
key. In practice that was both too slow (15-25s+ per reply) and not reliably fluent
in Telugu -- a small general-purpose open model at that size isn't a strong bet for
a lower-resource language. This now uses **Claude Haiku 4.5 via OpenRouter**
(`anthropic/claude-haiku-4.5`, reached through the OpenAI-compatible client, since
OpenRouter exposes that API shape for every model it hosts): fast (runs on real GPU
infrastructure, not your CPU) and considerably more capable, at the cost of a real,
if small, per-call fee billed against your OpenRouter credit balance. (Free
OpenRouter models were tried first but hit real reliability problems -- shared-pool
rate limits, then a broken upstream provider -- so this uses a paid model instead,
still through the same OpenRouter account rather than a separate Anthropic one.)
See setup step 2 below.

Why TTS is different here than in the notebook: Piper's voices sound
noticeably synthetic. Several Telugu-capable options were tried after that --
**edge-tts** (Microsoft's neural voices, still sounded synthetic for Telugu
specifically) and **Sarvam AI's Bulbul v3** (trained specifically on Indian
languages, but needs a separate paid account outside OpenRouter, ruled out).
There was also a detour to **Deepgram Aura-2 via OpenRouter** for its speed
and clarity, but it's English-only, and Telugu is this project's actual
target audience, so that meant running the whole script in English -- not
the real goal. TTS is back on **Google's Gemini 3.1 Flash TTS Preview via
OpenRouter** (`google/gemini-3.1-flash-tts-preview`) for Telugu support --
this was never confirmed against Google's own docs, and it's noticeably
slower (6-12+ seconds per reply) than Deepgram was, both real trade-offs
accepted deliberately to get Telugu output at all through OpenRouter alone.

## What's in this folder

- `bridge_service.py` -- the WebSocket server: bridges live call audio
  (Exotel AgentStream) to the STT/RAG/LLM/TTS pipeline.
- `pipeline.py` -- STT (Whisper Large V3 Turbo, hosted via OpenRouter), RAG
  (Chroma + sentence-transformers), LLM (Claude Haiku 4.5 via OpenRouter),
  TTS (Gemini 3.1 Flash TTS Preview, hosted via OpenRouter).
- `programme_config.py` -- the same editable programme pitch variables as the
  notebook. Edit the values here too.
- `test_deepgram_tts_direct.py` -- standalone script for previewing Deepgram
  Aura-2 voices (kept from the English detour -- useful again if TTS ever
  moves back to Deepgram/English).
- `requirements.txt` -- Python dependencies for this service.

## Setup

### 1. Sign up for Exotel and get a trial number

1. Go to https://exotel.com/ and sign up (7-day free trial, ₹500 in free
   call/SMS credit -- no upfront cost).
2. From the Exotel dashboard, note your **Account SID** and **API
   Key/Token**, and provision a trial **ExoPhone** (virtual number).
3. Set up a call flow with a **Voicebot Applet** (bidirectional streaming) on
   that ExoPhone, pointed at this bridge's WebSocket URL (see step 5 for the
   URL once the bridge is running and exposed).
4. **Never paste your Account SID/API Token into a chat with me or commit
   them to git.**

### 2. Get an OpenRouter API key and add credit

1. Go to https://openrouter.ai/ and sign in (or create an account).
2. Create an API key under **API Keys -> Create Key**. The full key value is
   shown only once, right when you create it -- copy it immediately.
3. Go to **Credits** and add a small balance (a few dollars covers a lot of
   testing at Haiku 4.5's pricing).
4. **Never paste this key into a chat with me or commit it to git** -- set it as an
   environment variable instead:
```bash
echo 'export OPENROUTER_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```
The `openai` Python package (installed in the next step, used here as an
OpenAI-compatible client pointed at OpenRouter) reads this via `pipeline.py` --
no code change needed if you rotate the key later, just update the env var and
restart the bridge. This is billed against your OpenRouter credit balance --
`pipeline.py`'s `OPENROUTER_MODEL` (`anthropic/claude-haiku-4.5`) is a paid
model, so calls cost something per use, same as calling Claude directly would.
TTS (`google/gemini-3.1-flash-tts-preview`) is billed against this same
OpenRouter account and key -- no separate signup needed for it.

### 3. Install Python dependencies for the bridge

```bash
cd telephony_bridge
python3 -m venv venv
source venv/bin/activate
# CPU-only torch build first (needed by the sentence-transformers embedder), to avoid
# pulling multi-GB CUDA packages you won't use -- speech-to-text no longer needs this,
# it runs hosted via OpenRouter now, not locally:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

No voice file to download this time -- the TTS request goes out live over the network on
each call, using the voice name set in `pipeline.py` (`OPENROUTER_TTS_VOICE`, currently
`"Zephyr"`, one of Gemini's ~30 language-agnostic character voices -- there's no
locale-specific name like edge-tts's `te-IN-*` voices; Gemini is expected to speak
whatever language the input text is in). There's no local CLI to preview a Gemini voice
before committing to it -- the only way to check one is a real test call.

### 4. Edit the programme details

Open `programme_config.py` and fill in the real `PROGRAMME_*` / `CERTIFICATION_NAME`
/ `COMPANY_NAME` values (same as you did in the notebook).

### 5. Start the bridge service and expose it publicly

```bash
python3 bridge_service.py
```
You should see: `Exotel AgentStream bridge listening on 0.0.0.0:8090`

Exotel's cloud platform needs to reach this over the public internet -- unlike
the old Asterisk setup, softphones on the same machine/LAN won't work here.
For local testing, expose the port with a tunnel:
```bash
ngrok http 8090
```
This gives a public `https://<random>.ngrok-free.app` URL; use the
`wss://<random>.ngrok-free.app/?sample-rate=8000` version of it (`wss://`, not
`https://`) as the WebSocket URL in the Exotel Voicebot Applet config from
step 1. The `?sample-rate=8000` query param tells Exotel to stream audio at
8kHz, matching what `bridge_service.py` expects -- don't drop it. For a real
deployment (not just local testing), point the Voicebot Applet at your
server's actual public address instead of an ngrok tunnel.

### 6. Test with a real call

Call your Exotel trial number from your own phone. You should hear the AI's
consent disclosure, then be able to talk to it -- ask about the fee, the
tracks, say "remove my number," etc., same test cases as the notebook's
harness. Watch the terminal running `bridge_service.py` for STT/LLM timing
and transcripts. If a message doesn't parse as expected, it'll print as
`[unhandled event ...]` with the raw JSON -- Exotel's exact protocol field
names weren't directly testable before a real call, so paste that line back
if you see it, so the parsing can be corrected.

## Known limitations of this first version

- LLM and speech-to-text calls both cost real money now (Claude Haiku 4.5 and
  Whisper Large V3 Turbo, both via OpenRouter) and need internet access + a
  funded `OPENROUTER_API_KEY` -- STT moved off this machine's CPU specifically
  because local transcription kept hitting unpredictable multi-second-to-a-minute
  stalls; hosted Whisper trades a small per-call cost for reliability and speed.
- TTS also costs real money now and needs live internet access (Gemini 3.1
  Flash TTS Preview via OpenRouter, same funded `OPENROUTER_API_KEY` as the
  LLM/STT) -- unlike Piper, it won't work offline. Telugu output
  quality/correctness for this specific model was not confirmed before
  switching to it, and it's noticeably slower (6-12+ seconds per reply) than
  the Deepgram/English setup tried in between -- listen critically on a real
  test call rather than assuming it's right.
- Exotel's trial account can usually only call/receive from phone numbers
  you've manually verified in their console -- fine for testing, not for
  calling arbitrary leads until the account is upgraded off the trial tier.
- Barge-in exists (the caller talking over the bot cuts its reply short) but
  is tuned with a fixed threshold (`BARGE_IN_SPEECH_FRAMES` in
  `bridge_service.py`) -- a brief cough or background noise could still
  occasionally interrupt a reply; loosen/tighten that constant if it's too
  trigger-happy or too slow to react.
- One asyncio task pair per call -- fine for testing a handful of calls, not for scale.
- VAD-based turn detection is a fixed silence timeout (`SILENCE_MS_TO_END_TURN`
  in `bridge_service.py`), not adaptive -- tune it if it cuts callers off too
  eagerly or waits too long.
- No outbound dialing, no answering-machine detection, no post-call
  classification pass yet -- those are the next steps once this wiring is
  confirmed working.
