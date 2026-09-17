# Phase 2 (testing): AI telecaller over a real phone call

This connects the same STT -> RAG -> LLM -> TTS idea from `ai_telecaller_poc.ipynb`
to an actual phone call, using mostly free, self-hosted software (one exception below).
It does **not** need a real phone number: you register two free softphone apps to your
own Asterisk server and call one from the other -- one call reaches the AI.

Runs on your own machine or a free-tier VM (e.g. Oracle Cloud Always Free) --
**not** in Colab (Colab isn't reachable from the internet and isn't always-on,
which a live call bridge needs).

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
See setup step 3 below.

Why TTS is different here than in the notebook: Piper's voices sound
noticeably synthetic. This first moved to **edge-tts** (free access to
Microsoft's neural voices via Microsoft Edge's "Read aloud" service, no API
key needed), but Telugu specifically still sounded noticeably synthetic even
with Microsoft's voices -- a real limitation, not a config issue, that briefly
led to running the whole pipeline in English instead (better TTS quality,
worse fit for actual Telugu-speaking leads). TTS then moved to Google's
Gemini 3.1 Flash TTS Preview via OpenRouter on the theory that its broad
claimed language coverage would help, but that support was never confirmed
for Telugu specifically. TTS has since moved again, to **Sarvam AI's Bulbul
v3** (`bulbul:v3`) -- a model trained specifically on Indian languages,
Telugu included, rather than a general-purpose multilingual model. This is a
separate paid account/API key from OpenRouter (OpenRouter doesn't host a
Telugu-specialized TTS model), reached through Sarvam's own `sarvamai`
Python SDK. Still worth listening to critically rather than assuming it's
fixed, but it's a stronger bet than a general model's broad language list.
See setup step 3 below for getting a Sarvam API key.

## What's in this folder

- `bridge_service.py` -- the AudioSocket server: bridges live call audio to the
  STT/RAG/LLM/TTS pipeline.
- `pipeline.py` -- STT (Whisper Large V3 Turbo, hosted via OpenRouter), RAG
  (Chroma + sentence-transformers), LLM (Claude Haiku 4.5 via OpenRouter),
  TTS (Sarvam AI's Bulbul v3).
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

### 3. Get an OpenRouter API key and add credit

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

### 4. Get a Sarvam AI API key and add credit

TTS runs on Sarvam AI, a separate account/API key from OpenRouter (OpenRouter
doesn't host a Telugu-specialized TTS model).

1. Go to https://www.sarvam.ai/ and sign in (or create an account).
2. Create an API key/subscription key from your Sarvam dashboard. Same rule
   as the OpenRouter key: copy it immediately, it's typically shown once.
3. Add a small balance to cover testing.
4. **Never paste this key into a chat with me or commit it to git**:
```bash
echo 'export SARVAM_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```
The `sarvamai` Python package (installed in the next step) reads this via
`pipeline.py` -- no code change needed if you rotate the key later, just
update the env var and restart the bridge.

### 5. Install Python dependencies for the bridge

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
each call, using the speaker name set in `pipeline.py` (`SARVAM_TTS_SPEAKER`, currently
`"anand"`, one of bulbul:v3's ~39 speaker names -- speaker names aren't interchangeable
across Sarvam's bulbul model versions, so a name valid for v2 won't work for v3 and vice
versa). There's no local CLI to preview a voice before committing to it -- the only way to
check one is a real test call.

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

- LLM and speech-to-text calls both cost real money now (Claude Haiku 4.5 and
  Whisper Large V3 Turbo, both via OpenRouter) and need internet access + a
  funded `OPENROUTER_API_KEY` -- STT moved off this machine's CPU specifically
  because local transcription kept hitting unpredictable multi-second-to-a-minute
  stalls; hosted Whisper trades a small per-call cost for reliability and speed.
- TTS also costs real money now and needs live internet access (Sarvam AI's
  Bulbul v3, a separate funded `SARVAM_API_KEY` from OpenRouter's LLM/STT) --
  unlike Piper, it won't work offline. It's trained specifically on Indian
  languages, a stronger bet than the general-purpose models tried before it,
  but still listen critically on a real test call rather than assuming it's
  right.
- Barge-in exists (the caller talking over the bot cuts its reply short) but
  is tuned with a fixed threshold (`BARGE_IN_SPEECH_FRAMES` in
  `bridge_service.py`) -- a brief cough or background noise could still
  occasionally interrupt a reply; loosen/tighten that constant if it's too
  trigger-happy or too slow to react.
- One thread per call -- fine for testing a handful of calls, not for scale.
- VAD-based turn detection is a fixed silence timeout (`SILENCE_MS_TO_END_TURN`
  in `bridge_service.py`), not adaptive -- tune it if it cuts callers off too
  eagerly or waits too long.
- No outbound dialing, no answering-machine detection, no post-call
  classification pass yet -- those are the next steps once this wiring is
  confirmed working.
