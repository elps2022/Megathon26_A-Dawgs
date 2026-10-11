"""Safety and Domain Moderation Guardrails for Lloyds Green Home AI Advisor.

Enforces:
1. Strict Safety & Content Moderation: Blocks hate speech, harassment, sexism, racism,
   violence, and attacks on protected characteristics under FCA and Lloyds ethical AI policy.
2. Domain Relevance & Gibberish Filter: Intercepts off-topic queries (e.g. general trivia,
   random words) and gibberish, politely redirecting users to home energy decarbonisation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ActionType = Literal["ALLOW", "BLOCK_SAFETY", "REFOCUS_OFF_TOPIC", "REFOCUS_GIBBERISH"]


@dataclass
class GuardrailResult:
    action: ActionType
    reason: str
    message: str | None = None

    @property
    def passed(self) -> bool:
        return self.action == "ALLOW"


# Regex patterns for safety violations (hate speech, slurs, harassment, protected characteristics attacks)
SAFETY_PATTERNS = [
    # Gender / Sexism
    r"\b(women|men|females|males|girls|boys)\s+are\s+(lesser|inferior|worse|stupid|worthless|trash|property|slaves)\b",
    r"\b(hate|kill|abuse|subjugate)\s+(all\s+)?(women|men|females|males)\b",
    # Race / Ethnicity / Discrimination
    r"\b(black|white|asian|jewish|muslim|arab|roma|minority)\s+lives\s+(don'?t|do\s+not)\s+matter\b",
    r"\b(white|black|aryan)\s+supremac(y|ist)\b",
    r"\b(inferior|subhuman|dirty|scum)\s+(race|people|immigrants|ethnic)\b",
    r"\b(death\s+to|kill\s+all|wipe\s+out)\s+\w+\b",
    # Severe slurs & derogatory terms
    r"\b(nigg[a-z]*|fag[a-z]*|kike|chink|spic|paki|retard[s]?)\b",
    # Violence / self-harm / weapons
    r"\b(how\s+to\s+(build|make)\s+(a\s+)?bomb|mass\s+shooting|kill\s+myself|commit\s+suicide)\b",
    # Prompt injection / jailbreaks
    r"\b(ignore\s+(all\s+)?previous\s+instructions|system\s+prompt|jailbreak|dan\s+mode|unfiltered\s+mode)\b",
]

COMPILED_SAFETY = [re.compile(p, re.IGNORECASE) for p in SAFETY_PATTERNS]

# Explicit off-topic keywords (trivia, animals, pop culture, cooking, sports)
OFF_TOPIC_TRIGGERS = [
    r"\b(color|colour)\s+of\s+the\s+sky\b",
    r"\b(recipe|cook|bake|pasta|pizza|cake)\b",
    r"\b(football|soccer|cricket|nba|world\s+cup|match|score\s+a\s+goal)\b",
    r"\b(who\s+is\s+the\s+president|capital\s+of|tell\s+me\s+a\s+joke|poem)\b",
    r"\b(dog[s]?|cat[s]?|pet[s]?|puppy|kitten|lion|tiger|elephant)\b",
    r"\b(crypto|bitcoin|ethereum|stock\s+market|forex)\b",
    r"\b(movie[s]?|song[s]?|celebrity|hollywood|actor|actress)\b",
]

COMPILED_OFF_TOPIC = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_TRIGGERS]

# Energy & housing domain vocabulary (if user prompt contains at least one, it's considered domain-relevant)
DOMAIN_TERMS = {
    "home", "house", "property", "flat", "terrace", "detached", "bungalow", "building", "dwelling",
    "score", "green", "rating", "epc", "sap", "band",
    "energy", "electricity", "gas", "fuel", "bill", "bills", "kwh", "saving", "savings", "cost", "costs",
    "carbon", "co2", "emission", "emissions", "climate", "net zero", "kg", "tonnes",
    "retrofit", "upgrade", "upgrades", "measure", "measures", "improve", "improvement", "plan",
    "insulation", "insulate", "loft", "roof", "wall", "walls", "cavity", "solid", "glazing", "window", "windows", "draught",
    "heating", "heater", "heat pump", "heat pumps", "boiler", "radiator", "radiators", "thermostat", "trv", "controls",
    "solar", "pv", "panel", "panels", "inverter", "generation",
    "grant", "grants", "bus", "eco4", "gbis", "subsidy", "subsidies", "cashback", "finance", "lender", "mortgage", "lloyds",
    "value", "valuation", "hpi", "uplift", "equity", "ltv",
    "infeasible", "barrier", "suitable", "unsuitable", "qualify", "eligib",
    "deteriorat", "protect", "prevent", "avoid", "drop", "worse", "decrease", "risk",
    "hello", "hi", "hey", "help", "advisor", "recommend", "advice", "who are you", "what can you do",
    "follow-up", "continue", "more", "tell me more", "details", "why", "explain"
}


def is_gibberish(text: str) -> bool:
    """Detect non-word keyboard mashing or unintelligible strings."""
    stripped = text.strip()
    if len(stripped) < 4:
        return False
    # If it's a single word with no spaces and long length
    words = stripped.split()
    if len(words) == 1 and len(words[0]) >= 7:
        word = words[0].lower()
        # Count vowels
        vowels = sum(1 for c in word if c in "aeiouy")
        vowel_ratio = vowels / len(word)
        # Check consonant clusters (4 or more consecutive consonants)
        consonant_cluster = re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", word)
        if vowel_ratio < 0.15 or vowel_ratio > 0.85 or consonant_cluster:
            return True
    return False


def evaluate_guardrails(query: str, has_history: bool = False) -> GuardrailResult:
    """Run safety and domain boundaries on incoming customer prompt."""
    q_norm = " ".join(query.strip().split())

    # 1. Safety & Content Moderation Gate (Always enforced, even with history)
    for pat in COMPILED_SAFETY:
        if pat.search(q_norm):
            return GuardrailResult(
                action="BLOCK_SAFETY",
                reason="policy_violation_hate_speech_or_harm",
                message=(
                    "This request violates our acceptable use and safety policy. "
                    "As the Lloyds Bank Green Home Advisor, I am committed to respectful, inclusive, "
                    "and professional guidance focused strictly on property energy efficiency and sustainability."
                ),
            )

    # 2. Gibberish & Keyboard Mashing Detection
    if is_gibberish(q_norm):
        return GuardrailResult(
            action="REFOCUS_GIBBERISH",
            reason="unintelligible_input",
            message=(
                "I couldn't understand that message. "
                "As your Lloyds Green Home Advisor, I can answer questions about your property's Green Home Score, "
                "recommended energy retrofits (such as heat pumps and insulation), fuel bill savings, and available grants. "
                "Please ask a question about your home's decarbonisation plan."
            ),
        )

    # 3. Explicit Off-Topic Queries
    for pat in COMPILED_OFF_TOPIC:
        if pat.search(q_norm):
            return GuardrailResult(
                action="REFOCUS_OFF_TOPIC",
                reason="explicit_off_topic_query",
                message=(
                    "That question is outside the scope of home energy and sustainability. "
                    "As the Lloyds Bank Green Home Advisor, I specialise in helping you understand your property's "
                    "Green Home Score, energy efficiency upgrades, fuel bill savings, and green mortgage incentives. "
                    "Feel free to ask how you can improve your home's energy performance."
                ),
            )

    # 4. If part of an ongoing multi-turn dialogue, allow natural conversational follow-ups
    if has_history:
        return GuardrailResult(action="ALLOW", reason="passed_conversation_history")

    # 5. Domain Relevance Gate: Check if standalone query has any connection to housing, energy, or greetings
    q_words = re.findall(r"[a-z0-9\-]+", q_norm.lower())
    has_domain_term = any(
        w in DOMAIN_TERMS or any(dt in w for dt in ["insulat", "heat", "retro", "energ", "epc", "electr", "grant", "mortg"])
        for w in q_words
    )

    # Single-word or short queries that have zero domain connection (e.g. "dogs", "blue car")
    if len(q_words) <= 3 and not has_domain_term:
        return GuardrailResult(
            action="REFOCUS_OFF_TOPIC",
            reason="unrelated_topic",
            message=(
                "That topic is outside the scope of property decarbonisation. "
                "I am here to assist you with your home's Green Home Score, recommended retrofit measures, "
                "fuel bill and carbon savings, and eligible grants. How can I assist with your home today?"
            ),
        )

    # 6. Passed all guardrails
    return GuardrailResult(action="ALLOW", reason="passed")

