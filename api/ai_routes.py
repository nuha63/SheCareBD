"""
SheCare BD — AI API Routes
===========================
POST /api/ai/insight  — Step 3 (COMPLETE)
POST /api/ai/chat     — Step 4 (COMPLETE)

Both routes use the shared safety pipeline:
    input_risk_check() → llm_service.generate() → output_post_filter()
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from services.safety_filter import (
    input_risk_check,
    output_post_filter,
    SAFE_FALLBACK_MESSAGE,
)
from services.llm_service import (
    generate,
    SHARED_WELLNESS_SYSTEM_PROMPT,
    REQUIRED_DISCLAIMER,
    LLM_FALLBACK_MESSAGE,
)
from services.daily_limit_service import (
    is_limit_reached,
    increment_usage,
    DAILY_CHAT_LIMIT,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ==================================================
# SHARED SAFETY RESPONSES
# ==================================================

EMERGENCY_SAFE_RESPONSE = (
    "এটি জরুরি পরিস্থিতি হতে পারে। অতিরিক্ত রক্তপাত, অজ্ঞান হয়ে যাওয়া, "
    "শ্বাসকষ্ট বা অসহ্য ব্যথার মতো গুরুতর উপসর্গ থাকলে দেরি না করে "
    "নিকটস্থ জরুরি চিকিৎসা সেবা নিন। "
    + REQUIRED_DISCLAIMER
)

MEDICAL_RISK_SAFE_RESPONSE = (
    "আমি নির্দিষ্ট ওষুধ, ডোজ বা প্রেসক্রিপশন পরামর্শ দিতে পারি না। "
    "তবে আপনার উপসর্গ সম্পর্কে সাধারণ wellness ও health education দিতে পারি। "
    "প্রয়োজন হলে একজন যোগ্য চিকিৎসকের সঙ্গে কথা বলুন। "
    + REQUIRED_DISCLAIMER
)

DAILY_LIMIT_RESPONSE = (
    f"আজকের AI chat limit ({DAILY_CHAT_LIMIT} টি) শেষ হয়েছে। "
    "আগামীকাল আবার চেষ্টা করতে পারবেন। "
    + REQUIRED_DISCLAIMER
)

CHAT_FALLBACK_RESPONSE = (
    "এই মুহূর্তে AI সহায়তা প্রদান করা সম্ভব হচ্ছে না। "
    "পর্যাপ্ত ঘুম, পানি পান ও হালকা ব্যায়াম সুস্বাস্থ্যের জন্য সহায়ক। "
    + REQUIRED_DISCLAIMER
)


# ==================================================
# CHAT-SPECIFIC SYSTEM PROMPT
# ==================================================

CHAT_SYSTEM_PROMPT_SUFFIX = """

CHAT-SPECIFIC INSTRUCTIONS:
You are a conversational Bangla-first women's wellness assistant for SheCare BD.
The user is asking a general wellness question in real-time chat.

YOUR ROLE:
- Respond warmly and briefly (2–4 sentences maximum).
- You MUST ALWAYS respond in pure Bengali (Bangla) language and script. Even if the user uses English terms or phonetically spelled English words, you must reply in Bengali. NEVER reply in English.
- Focus on: sleep, hydration, nutrition education, exercise, stress management,
  emotional wellbeing, menstrual cycle education, lifestyle habits.

WHAT YOU MUST NEVER DO:
1. Name any specific medicine, drug, or pharmaceutical product.
2. Suggest any dosage, quantity, or frequency of any medication.
3. Provide a prescription or tell the user what to take.
4. Diagnose any medical condition, disease, or disorder.
5. Claim that one symptom definitely causes another medical outcome.
6. Provide treatment instructions or medical protocols.

LANGUAGE RULES:
- Use only association/correlation language:
    ✓ "আপনার লগে ... একই সময়ে দেখা যাচ্ছে"
    ✓ "may be associated with"
    ✓ "sometimes seen together"
    ✗ "... এর কারণে ... হচ্ছে" (NEVER use this)
    ✗ "causes" (NEVER use this for medical claims)

IF USER ASKS FOR MEDICINE / PRESCRIPTION / DIAGNOSIS:
- Politely explain you cannot provide that.
- Redirect to general wellness guidance.
- Suggest consulting a registered healthcare professional.

MANDATORY DISCLAIMER:
Every single response MUST end with exactly:
এটি চিকিৎসকের বিকল্প নয়। গুরুতর বা দীর্ঘস্থায়ী উপসর্গ থাকলে নিবন্ধিত চিকিৎসকের পরামর্শ নিন।
"""

CHAT_FULL_SYSTEM_PROMPT = SHARED_WELLNESS_SYSTEM_PROMPT + CHAT_SYSTEM_PROMPT_SUFFIX


# ==================================================
# HELPER — resolve anonymous IP key from request
# ==================================================

def _get_ip_key(request: Optional[Request]) -> str:
    """
    Derive an anonymous daily-limit key from the client's IP address.
    Falls back to '__test__' when no real request object is available
    (e.g. during unit tests), which bypasses the daily limit entirely.
    """
    if request is None:
        return "__test__"
    client = getattr(request, "client", None)
    if client and client.host:
        return client.host
    # X-Forwarded-For header (behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return "__test__"


# ==================================================
# ─────────────────────────────────────────────────
#   STEP 3 — INSIGHT ENDPOINT
# ─────────────────────────────────────────────────
# ==================================================

class InsightPayload(BaseModel):
    """
    Minimized analytics payload from Flutter's AnalyticsService.
    Only aggregated, anonymized values — no PII accepted.
    """
    mood_average: Optional[float] = Field(
        default=None, ge=0.0, le=5.0,
        description="Average mood score (0–5 scale)",
    )
    average_sleep: Optional[float] = Field(
        default=None, ge=0.0, le=24.0,
        description="Average sleep duration in hours (0–24)",
    )
    cycle_average: Optional[int] = Field(
        default=None, ge=1, le=90,
        description="Average menstrual cycle length in days",
    )
    frequent_symptoms: List[str] = Field(
        default_factory=list, max_length=10,
        description="Up to 10 most-logged symptom category names",
    )

    @field_validator("frequent_symptoms")
    @classmethod
    def symptoms_are_short_strings(cls, v: List[str]) -> List[str]:
        for item in v:
            if len(item) > 80:
                raise ValueError(
                    f"Symptom label too long ({len(item)} chars). "
                    "Only category names are accepted."
                )
        return v


class InsightResponse(BaseModel):
    success: bool
    insight: str
    source: str  # "gemini" | "fallback" | "safety_fallback"


def _build_analytics_summary(payload: InsightPayload) -> str:
    parts = ["Wellness tracking summary (last 30 days):"]
    if payload.mood_average is not None:
        parts.append(f"- Average mood: {payload.mood_average:.1f} / 5.0")
    if payload.average_sleep is not None:
        parts.append(f"- Average sleep: {payload.average_sleep:.1f} hours per night")
    if payload.cycle_average is not None:
        parts.append(f"- Average cycle length: {payload.cycle_average} days")
    if payload.frequent_symptoms:
        parts.append(f"- Most frequently logged symptoms: {', '.join(payload.frequent_symptoms)}")
    else:
        parts.append("- No specific symptoms logged frequently")
    return "\n".join(parts)


def _build_deterministic_fallback(payload: InsightPayload) -> str:
    lines = []
    if payload.frequent_symptoms and payload.average_sleep is not None:
        symptoms = " এবং ".join(payload.frequent_symptoms[:2])
        lines.append(
            f"আপনার সাম্প্রতিক লগে গড় ঘুম ছিল {payload.average_sleep:.1f} ঘণ্টা "
            f"এবং {symptoms} বেশি দেখা গেছে।"
        )
    elif payload.frequent_symptoms:
        symptoms = " এবং ".join(payload.frequent_symptoms[:2])
        lines.append(f"আপনার সাম্প্রতিক লগে {symptoms} বেশি দেখা গেছে।")
    elif payload.average_sleep is not None:
        lines.append(f"আপনার গড় ঘুম ছিল {payload.average_sleep:.1f} ঘণ্টা।")

    if payload.average_sleep is not None and payload.average_sleep < 7.0:
        lines.append("পর্যাপ্ত বিশ্রাম ও নিয়মিত ঘুমের রুটিন সহায়ক হতে পারে।")
    elif payload.mood_average is not None and payload.mood_average < 3.0:
        lines.append(
            "মানসিক স্বাস্থ্যের যত্নে হালকা ব্যায়াম ও সামাজিক সংযোগ সহায়ক হতে পারে।"
        )
    else:
        lines.append("পর্যাপ্ত পানি পান ও নিয়মিত বিশ্রামের দিকে নজর রাখা উপকারী।")

    lines.append(REQUIRED_DISCLAIMER)
    return " ".join(lines)


INSIGHT_SYSTEM_PROMPT_SUFFIX = """

INSIGHT-SPECIFIC INSTRUCTIONS:
You are analyzing a 30-day self-reported wellness tracking summary.
The data contains only aggregated numbers — no names, no personal identifiers.

Your task:
1. Provide exactly 1 short wellness insight (2–3 sentences maximum).
2. Observe any pattern in the aggregated data using association language only.
3. Give one gentle, actionable lifestyle suggestion.
4. NEVER state that one metric causes another.
5. NEVER diagnose, prescribe, or recommend medicines.
6. Keep it warm, brief, and supportive.
7. Write in Bengali (Bangla).
8. End with the required medical disclaimer on a new line.
"""

INSIGHT_FULL_SYSTEM_PROMPT = SHARED_WELLNESS_SYSTEM_PROMPT + INSIGHT_SYSTEM_PROMPT_SUFFIX


@router.post("/insight", response_model=InsightResponse)
def generate_insight(payload: InsightPayload) -> InsightResponse:
    """POST /api/ai/insight — Step 3."""
    summary = _build_analytics_summary(payload)
    logger.info("Insight request. Summary length: %d chars", len(summary))

    risk = input_risk_check(summary)
    if risk == "emergency":
        return InsightResponse(success=True, insight=EMERGENCY_SAFE_RESPONSE, source="safety_fallback")
    if risk == "medical_risk":
        return InsightResponse(success=True, insight=MEDICAL_RISK_SAFE_RESPONSE, source="safety_fallback")

    raw_text = generate(system_prompt=INSIGHT_FULL_SYSTEM_PROMPT, user_content=summary)
    filtered_text = output_post_filter(raw_text)

    if raw_text == LLM_FALLBACK_MESSAGE or filtered_text == SAFE_FALLBACK_MESSAGE:
        return InsightResponse(success=True, insight=_build_deterministic_fallback(payload), source="fallback")

    return InsightResponse(success=True, insight=filtered_text, source="gemini")


# ==================================================
# ─────────────────────────────────────────────────
#   STEP 4 — CHAT ENDPOINT
# ─────────────────────────────────────────────────
# ==================================================

class ChatRequest(BaseModel):
    """
    Minimal chat request — single message only.
    No PII, no journal history, no phone number.
    """
    message: str = Field(..., min_length=1, max_length=1000,
                         description="The user's wellness question (1–1000 chars)")


class ChatResponse(BaseModel):
    success: bool
    response: str
    source: str  # "gemini" | "emergency" | "medical_risk" | "daily_limit" | "fallback"


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """
    POST /api/ai/chat — Step 4

    Pipeline:
        ChatRequest (validated, 1–1000 chars, no PII)
            ↓
        Resolve anonymous IP key
            ↓
        Check daily limit (5/day per IP)
            ├── limit reached → DAILY_LIMIT_RESPONSE, source="daily_limit"
            └── under limit
                    ↓
                input_risk_check(message)
                    ├── "emergency"    → EMERGENCY_SAFE_RESPONSE,    source="emergency"
                    ├── "medical_risk" → MEDICAL_RISK_SAFE_RESPONSE, source="medical_risk"
                    └── "safe"
                            ↓
                        generate(CHAT_FULL_SYSTEM_PROMPT, message)
                            ↓
                        output_post_filter(raw_text)
                            ├── safe     → source="gemini"  (increment usage)
                            └── blocked  → CHAT_FALLBACK_RESPONSE, source="fallback"

    Errors: Never exposes API keys, exceptions, or Gemini internals.
    """
    ip_key = _get_ip_key(request)
    message = payload.message.strip()

    # ── 1. Daily limit check ────────────────────────────────────────────
    if is_limit_reached(ip_key):
        logger.info("Daily limit reached for %s", ip_key)
        return ChatResponse(
            success=True,
            response=DAILY_LIMIT_RESPONSE,
            source="daily_limit",
        )

    # ── 2. Input safety check ───────────────────────────────────────────
    risk = input_risk_check(message)
    logger.info("Chat input_risk_check: %s", risk)

    if risk == "emergency":
        return ChatResponse(
            success=True,
            response=EMERGENCY_SAFE_RESPONSE,
            source="emergency",
        )

    if risk == "medical_risk":
        return ChatResponse(
            success=True,
            response=MEDICAL_RISK_SAFE_RESPONSE,
            source="medical_risk",
        )

    # ── 3. Generate (only for safe input) ──────────────────────────────
    raw_text = generate(system_prompt=CHAT_FULL_SYSTEM_PROMPT, user_content=message)

    # ── 4. Output post-filter ───────────────────────────────────────────
    filtered_text = output_post_filter(raw_text)

    is_llm_failed = raw_text == LLM_FALLBACK_MESSAGE
    is_filter_blocked = filtered_text == SAFE_FALLBACK_MESSAGE

    if is_llm_failed or is_filter_blocked:
        # Do NOT increment usage on failed/blocked responses
        return ChatResponse(
            success=True,
            response=CHAT_FALLBACK_RESPONSE,
            source="fallback",
        )

    # ── 5. Increment usage on successful Gemini response ───────────────
    increment_usage(ip_key)

    return ChatResponse(
        success=True,
        response=filtered_text,
        source="gemini",
    )
