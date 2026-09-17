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
noticeably synthetic. Several other options were tried after that --
**edge-tts** (Microsoft's neural voices, synthetic-sounding for Telugu),
**Google's Gemini 3.1 Flash TTS Preview via OpenRouter** and **Sarvam AI's
Bulbul v3** (Telugu-capable, but Sarvam needs a separate paid account and
Gemini's Telugu quality wasn't good enough on a real test call), and
**Deepgram Aura-2 via OpenRouter** (clearer/faster, but English-only, which
meant running the whole script in English). TTS now runs on **ElevenLabs**
(`eleven_flash_v2_5`, their lowest-latency model) with a specific custom
voice picked and previewed outside this codebase -- another separate paid
account outside OpenRouter (ElevenLabs isn't in OpenRouter's TTS catalog at
all), chosen for a further step up in voice quality for a real
business-facing bot. English-only in practice (the voice/model weren't
chosen or tested for Telugu), so this continues on the English script from
the Deepgram phase.

## What's in this folder

- `bridge_service.py` -- the WebSocket server: bridges live call audio
  (Exotel AgentStream) to the STT/RAG/LLM/TTS pipeline.
- `pipeline.py` -- STT (Whisper Large V3 Turbo, hosted via OpenRouter), RAG
  (Chroma + sentence-transformers), LLM (Claude Haiku 4.5 via OpenRouter),
  TTS (ElevenLabs, `eleven_flash_v2_5`).
- `programme_config.py` -- the same editable programme pitch variables as the
  notebook. Edit the values here too.
- `test_deepgram_tts_direct.py` -- standalone script for previewing Deepgram
  Aura-2 voices (kept from an earlier phase -- useful again if TTS ever
  moves back to Deepgram).
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

### 3. Get an ElevenLabs API key and voice

TTS runs on ElevenLabs, a separate account/API key from OpenRouter (ElevenLabs
isn't in OpenRouter's TTS catalog).

1. Go to https://elevenlabs.io/ and sign in (or create an account).
2. Create/select the voice you want the bot to use, and copy its **Voice ID**
   from the Voice Library (not a secret, safe to note down normally).
3. Create an API key from your ElevenLabs account settings.
4. **Never paste the API key into a chat with me or commit it to git** -- set
   it and the voice ID as environment variables instead:
```bash
echo 'export ELEVENLABS_API_KEY="your-key-here"' >> ~/.bashrc
echo 'export ELEVENLABS_VOICE_ID="your-voice-id-here"' >> ~/.bashrc
source ~/.bashrc
```
The `elevenlabs` Python package (installed in the next step) reads both via
`pipeline.py` -- no code change needed to rotate the key or swap voices, just
update the env vars and restart the bridge. This is billed against your
ElevenLabs account, separately from OpenRouter.

### 4. Install Python dependencies for the bridge

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
each call, using the voice set in `pipeline.py` (`ELEVENLABS_VOICE_ID`, read from the
`ELEVENLABS_VOICE_ID` env var set in step 3). Preview voices directly in your ElevenLabs
account's Voice Library before picking one -- no need to run any code to hear a sample.

### 5. Edit the programme details

Open `programme_config.py` and fill in the real `PROGRAMME_*` / `CERTIFICATION_NAME`
/ `COMPANY_NAME` values (same as you did in the notebook).

### 6. Start the bridge service and expose it publicly

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

### 7. Test with a real call

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
- TTS also costs real money now and needs live internet access (ElevenLabs,
  a separate funded `ELEVENLABS_API_KEY` from OpenRouter's LLM/STT) -- unlike
  Piper, it won't work offline. The voice/model in use weren't chosen or
  tested for Telugu, which is why this whole script currently runs in
  English -- see the TTS note near the top of this file.
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
