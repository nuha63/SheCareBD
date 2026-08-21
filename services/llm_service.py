"""
SheCare BD — Gemini LLM Service  (google-genai SDK)
=====================================================
Centralised wrapper for all Gemini API communication.
Uses the modern `google-genai` SDK (google.genai.Client).

Usage (in route files — Step 3+):
    from services.llm_service import generate, SHARED_WELLNESS_SYSTEM_PROMPT

    raw_text = generate(
        system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT + insight_specific_instructions,
        user_content=user_message,
    )
    safe_text = output_post_filter(raw_text)   # always applied by the caller

This module does NOT:
  - call output_post_filter() — the caller's responsibility
  - contain route/endpoint logic
  - contain database logic
  - duplicate safety-filter keyword lists (see safety_filter.py)
  - hard-code any API key
"""
import os
import logging
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load .env if not already loaded (safe to call multiple times)
load_dotenv()

logger = logging.getLogger(__name__)


# ==================================================
# CONSTANTS
# ==================================================

REQUIRED_DISCLAIMER = (
    "এটি চিকিৎসকের বিকল্প নয়। "
    "গুরুতর বা দীর্ঘস্থায়ী উপসর্গ থাকলে নিবন্ধিত চিকিৎসকের পরামর্শ নিন।"
)

LLM_FALLBACK_MESSAGE = (
    "এই মুহূর্তে আপনার জন্য একটি AI wellness insight তৈরি করা সম্ভব হচ্ছে না। "
    "কিছুক্ষণ পরে আবার চেষ্টা করুন। " + REQUIRED_DISCLAIMER
)

# Model name — centralised so it is easy to upgrade later
GEMINI_MODEL = "gemini-3.6-flash"


# ==================================================
# SHARED WELLNESS SYSTEM PROMPT
# Callers MAY append feature-specific instructions.
# ==================================================

SHARED_WELLNESS_SYSTEM_PROMPT = """
You are a wellness education assistant for SheCare BD, a Bangla-first women's wellness app in Bangladesh.

IDENTITY AND SCOPE:
- You are NOT a doctor, nurse, or medical professional.
- You provide general wellness, lifestyle, and educational information only.
- Your purpose is to help users understand patterns in their own health logs.

ABSOLUTE PROHIBITIONS — you must NEVER:
1. Diagnose any disease, disorder, or medical condition.
2. Recommend any specific medicine by name or category.
3. Suggest any dosage, quantity, or frequency of medication.
4. Provide administration or prescription instructions.
5. Tell the user to start, stop, or change any medication.
6. Provide definitive treatment instructions or plans.
7. Claim that one symptom or data point definitively causes another medical outcome.

DATA GROUNDING:
- Use ONLY the information explicitly provided in the user's logged data or prompt.
- Do NOT invent, assume, or infer unprovided data such as hydration, nutrition, exercise, stress, or other lifestyle factors.
- If a possible contributing factor is not present in the provided data, do not state that it may be related.
- Clearly distinguish observed patterns from general wellness education.

LANGUAGE AND TONE:
- Respond in Bengali (Bangla) unless the user writes in English, in which case you may respond in English.
- Be empathetic, warm, and supportive.
- Use cautious, association-based language such as:
    * "may be associated with"
    * "can sometimes be linked with"
    * "your logged data shows a pattern between"
    * "আপনার লগে একটি সম্পর্ক দেখা যাচ্ছে"
    * "একই সময়ে বেশি দেখা যাচ্ছে"
- Do NOT use definitive causal language such as:
    * "X causes Y"
    * "কম ঘুমের কারণে আপনার cramps হচ্ছে"

GUIDANCE SCOPE (what you CAN discuss):
- Sleep hygiene and rest habits
- Hydration and nutrition education
- Exercise and movement
- Stress management and mindfulness
- Mood and emotional wellbeing patterns
- General menstrual cycle education
- Lifestyle habits that may support general wellbeing
- Encouraging users to notice patterns in their logs

IF ASKED FOR DIAGNOSIS / MEDICATION / PRESCRIPTION:
- Politely explain that you are not able to provide that information.
- Redirect to general wellness guidance.
- Always recommend consulting a registered healthcare professional
  for persistent, severe, or concerning symptoms.

MANDATORY DISCLAIMER:
Every single response MUST end with exactly this line:
এটি চিকিৎসকের বিকল্প নয়। গুরুতর বা দীর্ঘস্থায়ী উপসর্গ থাকলে নিবন্ধিত চিকিৎসকের পরামর্শ নিন।
""".strip()


# ==================================================
# PRIVATE HELPERS
# ==================================================

def _get_api_key() -> Optional[str]:
    """Read GEMINI_API_KEY from environment. Returns None if missing."""
    return os.getenv("GEMINI_API_KEY")


def _ensure_disclaimer(text: str) -> str:
    """
    Guarantee the required disclaimer appears in the response.
    If already present, do not duplicate it.
    If missing, append it.
    """
    if REQUIRED_DISCLAIMER in text:
        return text
    return text.rstrip() + "\n\n" + REQUIRED_DISCLAIMER


def _extract_text(response) -> Optional[str]:
    """
    Safely extract plain text from a Gemini SDK response object.
    Works with both the new google-genai SDK and MagicMock in tests.
    Returns None if the response contains no usable text.
    """
    try:
        text = response.text
        if text and str(text).strip():
            return str(text).strip()
    except (AttributeError, ValueError):
        pass
    return None


# ==================================================
# PUBLIC API
# ==================================================

def generate(system_prompt: str, user_content: str) -> str:
    """
    Send a prompt to Gemini and return the generated text.


    Args:
        system_prompt: Instructions that configure Gemini's behaviour.
        user_content: The actual user message or analytics payload string.


    Returns:
        Generated text string (always str, never a raw SDK object).
        Falls back to LLM_FALLBACK_MESSAGE on any failure.


    NOTE:
        This function does NOT call output_post_filter().
        The CALLER is responsible for that step.
    """


    # --------------------------------------------------
    # 1. Validate API key
    # --------------------------------------------------
    api_key = _get_api_key()


    if not api_key:
        logger.warning(
            "GEMINI_API_KEY is not set. Returning fallback."
        )
        return LLM_FALLBACK_MESSAGE


    # --------------------------------------------------
    # 2. Create Gemini client
    # --------------------------------------------------
    try:
        client = genai.Client(api_key=api_key)


    except Exception as client_err:
        logger.error(
            "Gemini client configuration failed: %s",
            type(client_err).__name__,
        )
        return LLM_FALLBACK_MESSAGE


    # --------------------------------------------------
    # 3. Build combined prompt
    # --------------------------------------------------
    full_prompt = (
        f"{system_prompt}\n\n"
        f"---\n\n"
        f"{user_content}"
    )


    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ]
            )
        )


    except Exception as api_err:
        logger.exception(
            "Gemini API call failed: %s",
            type(api_err).__name__,
        )
        return LLM_FALLBACK_MESSAGE


    # --------------------------------------------------
    # 5. Extract response text
    # --------------------------------------------------
    text = _extract_text(response)


    if not text:
        logger.warning(
            "Gemini returned an empty or unusable response."
        )
        return LLM_FALLBACK_MESSAGE


    # --------------------------------------------------
    # 6. Ensure disclaimer
    # --------------------------------------------------
    text = _ensure_disclaimer(text)


    return text
