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

LANGUAGE NOTE: CONSENT_DISCLOSURE and OPT_OUT_REPLY below are translated into
Telugu (I'm not a certified Telugu speaker/translator -- please have a native
speaker check these two lines before relying on them for real calls; they are
the only lines spoken verbatim, so accuracy there matters most). The rest of
this file (system prompt, knowledge base) stays in English on purpose: the
LLM is instructed to respond in Telugu regardless of the instruction language,
and keeping facts in one authoritative English form makes them easier for you
to verify and edit than a second translated copy would be.
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
#
# Telugu: "Hello! I am Srinivas, calling from Raga Tech Source. This call is
# being recorded. Could you please tell me your name?" -- verify wording.
CONSENT_DISCLOSURE = (
    f"నమస్కారం! నేను {AGENT_DISPLAY_NAME}, {COMPANY_NAME} నుండి మాట్లాడుతున్నాను. "
    "ఈ కాల్ రికార్డ్ చేయబడుతోంది. మీ పేరు చెప్పగలరా?"
)

# Telugu: "Understood, we will not call you again. Thank you!" -- verify wording.
OPT_OUT_REPLY = "అర్థమైంది, ఇకపై మీకు కాల్ చేయము. ధన్యవాదాలు!"

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
2. If they say they're not interested at this stage, thank them politely and end the call.
3. The pitch (once they've said it's an okay time): explain there are three tracks --
   - {TRACKS[0]['name']}: {TRACKS[0]['description']}
   - {TRACKS[1]['name']}: {TRACKS[1]['description']}
   - {TRACKS[2]['name']}: {TRACKS[2]['description']}
   All three are priced at {FEE}, and {COMPANY_NAME}'s placement team actively works to get \
graduates interview opportunities -- it's not just a certificate at the end.
4. You will be given "RETRIEVED CONTEXT" before each caller message. Answer fee, dates, \
curriculum, and certification questions ONLY from that context. If it doesn't cover what they \
asked, say you'll confirm the detail and follow up -- NEVER invent or guess a fee, date, \
curriculum detail, or certification name.
5. Once they show interest in a specific track, confirm which track they'd like to go ahead with, \
or offer to go over the other tracks again if they're unsure.
6. If they want time to think, offer to share all three tracks' details over WhatsApp so they can \
review them later.
7. Handle objections (fee, doubts about outcomes) by acknowledging them, answering from retrieved \
context, then gently returning to confirming which track they want.
8. Close: once they're ready to proceed, tell them a {COMPANY_NAME} counselor will follow up \
shortly to help them enroll in that track, thank them for their time, and end warmly.

Keep responses SHORT (1-2 sentences) -- this is a live phone call, not a written chat.

IMPORTANT: Respond ONLY in Telugu (తెలుగు) for every reply, even though these instructions are \
in English. It's natural to keep company/track/product names (like {COMPANY_NAME} or "Full Stack \
+ AI") in English within an otherwise-Telugu sentence, the way people actually speak.
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
