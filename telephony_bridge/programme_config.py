"""Programme pitch details + knowledge base, shared by the phone bridge.

Keep these in sync with the PROGRAMME_* variables in ai_telecaller_poc.ipynb --
this is the phase-2 (live call) equivalent of that notebook's Section 2/3.
"""

# ============================================================
# EDIT ME: real programme details go here (same values as the notebook)
# ============================================================
PROGRAMME_NAME = "Full-Stack AI Engineering Bootcamp"
PROGRAMME_FEE = "INR 45,000 (plus applicable taxes)"
PROGRAMME_DATES = "Batch starts 6 October 2026, runs for 8 weeks, weekday evenings 7-9 PM IST"
PROGRAMME_DURATION = "8 weeks, 3 live sessions per week"
PROGRAMME_CURRICULUM = [
    "Python & ML fundamentals",
    "LLM application development (RAG, agents, fine-tuning basics)",
    "Deploying AI systems to production",
    "Capstone project with mentor review",
]
CERTIFICATION_NAME = "Certificate of Completion, co-signed by RagaTech Source"
COMPANY_NAME = "RagaTech Source"
AGENT_DISPLAY_NAME = "Riya"

CONSENT_DISCLOSURE = (
    f"Hi, this is {AGENT_DISPLAY_NAME}, an AI assistant calling on behalf of {COMPANY_NAME}. "
    "This call is being conducted by an AI system and is recorded for quality and training "
    "purposes. Is it okay if I take a couple of minutes to tell you about a training "
    "programme we're running?"
)

SYSTEM_PROMPT_TEMPLATE = f"""You are {AGENT_DISPLAY_NAME}, an AI voice agent for {COMPANY_NAME}, calling to \
pitch the "{PROGRAMME_NAME}" training programme and gauge the caller's interest.

Rules you must always follow:
1. The call opened with a clear disclosure that this is an AI-conducted, recorded call \
(that disclosure has already been played to the caller -- do not repeat it unless asked).
2. You will be given "RETRIEVED CONTEXT" before each caller message. When the caller asks \
about the fee, dates/schedule, duration, curriculum, or certification, you MUST base your \
answer ONLY on that retrieved context. If the retrieved context does not contain the answer, \
say you'll confirm the detail and follow up -- NEVER invent or guess a fee, date, or \
certification detail.
3. For general questions unrelated to the programme, answer briefly and helpfully, then \
steer the conversation back to the programme.
4. Keep responses SHORT (1-3 sentences) -- this is a live phone call, not a written chat.
5. Naturally probe the caller's level of interest as the conversation progresses.
6. Be respectful of objections. If the caller is hesitant, answer their concern once; do not \
pressure them repeatedly.
"""

KNOWLEDGE_BASE = [
    {"id": "fee", "text": f"The fee for the {PROGRAMME_NAME} is {PROGRAMME_FEE}."},
    {"id": "dates", "text": f"The {PROGRAMME_NAME} schedule: {PROGRAMME_DATES}."},
    {"id": "duration", "text": f"The {PROGRAMME_NAME} duration and format: {PROGRAMME_DURATION}."},
    {"id": "curriculum", "text": f"The {PROGRAMME_NAME} curriculum covers: " + "; ".join(PROGRAMME_CURRICULUM) + "."},
    {"id": "certification", "text": f"On completing the {PROGRAMME_NAME}, participants receive: {CERTIFICATION_NAME}."},
]

OPT_OUT_PHRASES = [
    "remove my number", "take me off", "stop calling", "don't call me", "do not call me",
    "do not call", "unsubscribe", "opt out", "opt-out", "stop contacting", "remove me from",
    "don't contact me", "do not contact me",
]


def is_opt_out_request(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in OPT_OUT_PHRASES)
