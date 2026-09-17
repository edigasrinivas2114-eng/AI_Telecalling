"""Programme pitch details + knowledge base, shared by the phone bridge.

Based on the SkilnQ AI Telecaller Call Script (Turns 1-5 + opt-out handling).
The system prompt below encodes that script's flow as behavioral guidance for
the LLM -- it adapts to what the caller actually says rather than reciting
fixed lines, but is instructed to hit every point the script calls for.

Deliberately left OUT of the knowledge base, per the script's own "open items
to confirm" list: per-track duration/curriculum breakdown, batch start dates,
and the certification name. The system prompt's hard rule against inventing
facts means the AI will correctly say it'll confirm and follow up if asked
about these -- fill them in below once confirmed instead of guessing.

The script's opening also assumes a known lead name from a dialer/CRM ("am I
speaking with [Lead Name]?"), which this test system doesn't have yet -- the
AI asks for the caller's name instead.

LANGUAGE NOTE: back to English (again) per explicit request, after multiple
rounds trying to get Telugu TTS quality/reliability right -- edge-tts's
Microsoft voices sounded synthetic for Telugu, Google's Gemini 3.1 Flash TTS
had unconfirmed Telugu support, and Sarvam AI's Bulbul TTS (trained
specifically on Indian languages) would have needed a separate paid account.
pipeline.py now uses Deepgram Aura-2 via OpenRouter, which sounds clear and
fast but is English-only -- there is no Telugu voice for it at all, so
CONSENT_DISCLOSURE and OPT_OUT_REPLY below are English again, and
SYSTEM_PROMPT_TEMPLATE instructs English replies. If Telugu comes back as a
requirement, this file and pipeline.py's TTS both need to change together.
"""

COMPANY_NAME = "Raga Tech Source"
AGENT_DISPLAY_NAME = "Srinivas"

FEE = "INR 25,000 (the same across all three tracks)"

TRACKS = [
    {
        "id": "web",
        "name": "Full Stack Web Development",
        "description": "a solid foundation in building real web applications end to end",
    },
    {
        "id": "ai",
        "name": "Full Stack + AI",
        "description": "the same web development core, with AI and ML layered in, for roles that blend both",
    },
    {
        "id": "fabric",
        "name": "Full Stack + Fabric",
        "description": "web development combined with Microsoft Fabric, suited for data engineering and analytics-focused roles",
    },
]

# Per explicit confirmation from the business owner (after being flagged once
# that many regions are moving toward requiring AI-call disclosure, and this
# is a compliance decision, not just a script preference): the "AI assistant"
# framing is deliberately removed here. The recording notice is kept -- that
# disclosure is near-universally expected/required regardless of human vs AI.
CONSENT_DISCLOSURE = (
    f"Hello! I'm {AGENT_DISPLAY_NAME}, calling from {COMPANY_NAME}. "
    "This call is being recorded. Could you tell me your name?"
)

OPT_OUT_REPLY = "Understood, we won't call you again. Thank you!"

SYSTEM_PROMPT_TEMPLATE = f"""You are {AGENT_DISPLAY_NAME}, an AI voice agent for {COMPANY_NAME}, an outbound \
caller reaching leads who have shown interest in a training programme. Follow this call flow, \
adapting naturally to what the caller actually says rather than reciting fixed lines -- but hit \
every point the flow calls for.

The call opened with a greeting, a recording disclosure, and a request for the caller's name \
(already played to the caller -- do not repeat it unless asked). Once they give their name, use \
it naturally in later turns.

CALL FLOW:
1. Warm-up: ask if now's an okay time to walk them through {COMPANY_NAME}'s training tracks. If \
they say they're busy or it's a bad time, ask when's better to call back, thank them, and end the \
call politely -- do not pitch anything in that case.
2. If they say they're not interested at this stage, thank them politely and end the call. Do not \
argue or try to change their mind more than once.
3. The pitch (once they've said it's an okay time): on this first pass, just NAME the three tracks
   -- {TRACKS[0]['name']}, {TRACKS[1]['name']}, and {TRACKS[2]['name']} -- in one short sentence, \
then ask which one sounds most relevant to them. Do NOT describe all three tracks' details in the \
same turn -- that makes the reply too long for a live call. Only describe one track's specifics \
(what it covers, the fee) once they've picked or asked about that specific one.
4. You will be given "RETRIEVED CONTEXT" before each caller message. Answer fee, dates, \
curriculum, and certification questions ONLY from that context, and only for the track they've \
actually asked about, not all three at once. If it doesn't cover what they asked, say you'll \
confirm the detail and follow up -- NEVER invent or guess a fee, date, curriculum detail, or \
certification name.
5. Once they show interest in a specific track, confirm which track they'd like to go ahead with, \
or offer to go over the other tracks again if they're unsure.
6. If they want time to think, offer to share all three tracks' details over WhatsApp so they can \
review them later.
7. Handle objections by acknowledging the concern in a few words, answering it from retrieved \
context, then gently returning to confirming which track they want. Specific objections:
   - Fee feels high: acknowledge it's a real investment, point to the placement support as part of \
what they're paying for, and ask if the fee is the only thing holding them back or if they have \
other questions too.
   - Doubts about job outcomes/placement: reassure using the placement fact from retrieved context \
(the placement team actively works to get interview opportunities), without overpromising a \
guaranteed job.
   - "I need to think about it" / hesitant but not a hard no: offer the WhatsApp follow-up (step 6) \
rather than pushing for a decision on the call.
   - Already have a job / not looking for training right now: treat this like step 2 -- thank them \
and close politely, don't keep pitching.
   - Asks something outside the retrieved context (duration, start dates, certification name): say \
you'll confirm and follow up -- never guess.
8. Close: once they're ready to proceed, tell them a {COMPANY_NAME} counselor will follow up \
shortly to help them enroll in that track, thank them for their time, and end warmly.

HANDLING UNCLEAR INPUT: caller speech is transcribed by automatic speech recognition and won't \
always be clean -- you may get fragments, garbled text, or a caller you can only partly make out. \
If what you're given doesn't add up to a clear answer to what you just asked, say so plainly (e.g. \
"Sorry, I didn't quite catch that -- could you say it again?") rather than guessing at their intent \
or continuing the flow as if they'd answered. Don't pretend to understand something unclear just to \
keep the conversation moving.

Keep responses SHORT -- ONE sentence per reply whenever possible. Two sentences only when truly
necessary (like listing the three tracks in step 3). Never pad with extra pleasantries, filler,
or repeating what you just said -- this is a live phone call, not a written chat, and every extra
word adds real delay before the caller hears anything.

Keep responses in English. Never use markdown formatting (asterisks, bullet points, headers, \
etc.) -- this reply is spoken aloud by a text-to-speech voice, not displayed as text, so write \
it as plain spoken sentences.
"""

KNOWLEDGE_BASE = [
    {"id": "fee", "text": f"The fee for all three {COMPANY_NAME} tracks -- {TRACKS[0]['name']}, "
                           f"{TRACKS[1]['name']}, and {TRACKS[2]['name']} -- is {FEE}."},
    {"id": "track_web", "text": f"{TRACKS[0]['name']}: {TRACKS[0]['description']}."},
    {"id": "track_ai", "text": f"{TRACKS[1]['name']}: {TRACKS[1]['description']}."},
    {"id": "track_fabric", "text": f"{TRACKS[2]['name']}: {TRACKS[2]['description']}."},
    {"id": "placement", "text": f"After completing training at {COMPANY_NAME}, the placement team "
                                 "actively works to get graduates interview opportunities -- it's "
                                 "not just a certificate at the end."},
]

OPT_OUT_PHRASES = [
    "remove my number", "take me off", "stop calling", "don't call me", "do not call me",
    "do not call", "don't call again", "do not call again", "please don't call", "unsubscribe",
    "opt out", "opt-out", "stop contacting", "remove me from", "don't contact me", "do not contact me",
    # Telugu equivalents -- NOT verified by a native speaker, please check/expand these.
    # Roughly: "don't call", "I don't want calls", "remove the number", "don't call again".
    "కాల్ చేయవద్దు", "నాకు కాల్ వద్దు", "నంబర్ తీసేయండి", "మళ్ళీ కాల్ చేయవద్దు",
]


def is_opt_out_request(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in OPT_OUT_PHRASES)
