"""
SheCare BD — Shared AI Safety Filter
=====================================
Single source of truth for all AI safety checks.
Used by BOTH /api/ai/insight and /api/ai/chat routes.

Priority:  emergency > medical_risk > safe

DO NOT duplicate these patterns in route files, main.py,
or any other service. Import this module instead.
"""
import re

# ==================================================
# SAFE FALLBACK — used by output_post_filter
# ==================================================

SAFE_FALLBACK_MESSAGE = (
    "আমি নির্দিষ্ট ওষুধ, ডোজ, চিকিৎসা বা রোগ নির্ণয়ের পরামর্শ দিতে পারি না। "
    "আপনার লগ করা তথ্য নিয়ে সাধারণ wellness ও lifestyle guidance দিতে পারি। "
    "গুরুতর বা দীর্ঘস্থায়ী উপসর্গ থাকলে নিবন্ধিত চিকিৎসকের পরামর্শ নিন।"
)


# ==================================================
# EMERGENCY PATTERNS  (highest priority)
# ==================================================

EMERGENCY_PATTERNS: list[str] = [
    # English
    "heavy bleeding",
    "severe bleeding",
    "excessive bleeding",
    "soaking pads repeatedly",
    "soaking through pads",
    "fainting",
    "fainted",
    "unconscious",
    "passed out",
    "severe dizziness",
    "difficulty breathing",
    "trouble breathing",
    "shortness of breath",
    "can't breathe",
    "cannot breathe",
    "chest pain",
    "severe abdominal pain",
    "unbearable pain",
    "severe pain",
    "suicidal thoughts",
    "suicide",
    "self harm",
    "harming myself",
    "kill myself",
    "want to die",
    "seizure",
    # Bangla
    "খুব বেশি রক্তপাত",
    "অতিরিক্ত রক্তপাত",
    "প্রচুর রক্তপাত",
    "অস্বাভাবিক রক্তপাত",
    "অনেক বেশি রক্ত",
    "রক্তপাত হচ্ছে",       # NEW: "আমার অনেক রক্তপাত হচ্ছে"
    "রক্তপাত",              # NEW: catches any sentence containing রক্তপাত
    "অজ্ঞান",
    "অজ্ঞান হয়ে যাওয়া",
    "অজ্ঞান হয়ে যাচ্ছি",  # NEW: "আমি অজ্ঞান হয়ে যাচ্ছি"
    "মাথা খুব ঘোরা",
    "শ্বাসকষ্ট",
    "শ্বাস নিতে পারছি না",
    "শ্বাস নিতে কষ্ট হচ্ছে",  # NEW: test case in audit
    "বুক ব্যথা",
    "প্রচণ্ড পেট ব্যথা",
    "অসহ্য ব্যথা",
    "প্রচণ্ড ব্যথা",
    "তীব্র ব্যথা",
    "আত্মহত্যা",
    "নিজেকে ক্ষতি করা",
    "মরে যেতে চাই",
    "মারা যাব",
    # Mixed-language variants
    "heavy bleeding হচ্ছে",
    "severe pain হচ্ছে",
    "অনেক বেশি bleeding",
    "অতিরিক্ত bleeding",
    "bleeding হচ্ছে অনেক",
    "bleeding অনেক বেশি",
]


# ==================================================
# MEDICAL RISK PATTERNS  (second priority)
# ==================================================

# --- Prescriptive/medicine intent phrases (phrase-level matching) ---
MEDICAL_RISK_PHRASES: list[str] = [
    # English intent phrases
    "what should i take",
    "what medicine should i take",
    "which medicine should i take",
    "what should i use",
    "how many tablets",
    "how much should i take",
    "what dose should i take",
    "recommended dose",
    "can you prescribe",
    "prescribe something",
    "prescription for",
    "should i take something for this",
    "can i take",
    "should i take",
    "take for period",
    "take for pain",
    # Treatment / cure phrases (English)
    "what treatment",
    "treatment plan",
    "treatment for",
    "what should i do to treat",
    "how do i treat",
    "how to treat",
    "cure for",
    "remedy for",
    "what can i take to cure",
    # Bangla intent phrases
    "কোন ওষুধ খাব",
    "কী ওষুধ খাব",
    "কি ওষুধ খাব",
    "কোন ওষুধ",
    "ওষুধ কী",
    "ওষুধ কি",
    "কতটা খাব",
    "কত ট্যাবলেট",
    "কত mg",
    "ডোজ কত",
    "ডোজ কী",
    "প্রেসক্রিপশন",
    "ওষুধ লিখে দিন",
    "কোন ওষুধ ব্যবহার করব",
    "কী খাব এই সমস্যায়",
    # Mixed Bangla-English intent variants
    "ki medicine",
    "kon medicine",
    "ki medicine khabo",
    "ki medicine nibo",
    "kono medicine",
]

# --- Medicine/drug terminology (single terms) ---
MEDICINE_TERMS: list[str] = [
    # English
    "medicine",
    "medication",
    "drug",
    "tablet",
    "capsule",
    "antibiotic",
    "painkiller",
    "dosage",
    "dose",
    "steroid",
    "hormone medicine",
    "pill",
    # Specific medicine names (input-level — must not be recommended)
    "ibuprofen",
    "paracetamol",
    "naproxen",
    "aspirin",
    "metformin",
    "mefenamic",
    # Bangla
    "ওষুধ",
    "ঔষধ",
    "মেডিসিন",
    "ট্যাবলেট",
    "ক্যাপসুল",
    "অ্যান্টিবায়োটিক",
    "ব্যথার ওষুধ",
    "ডোজ",
    "মাত্রা",
]

# --- Diagnosis request terms ---
DIAGNOSIS_REQUEST_TERMS: list[str] = [
    "do i have pcos",
    "do i have anemia",
    "do i have endometriosis",
    "have i got",
    "আমার কি pcos হয়েছে",
    "আমার কি এই রোগ হয়েছে",
    "আমার pcos আছে",
    "আমার কি cancer হয়েছে",
    "আমি কি pregnant",
]

# Combine all medical risk patterns for input check
MEDICAL_RISK_PATTERNS: list[str] = MEDICAL_RISK_PHRASES + DIAGNOSIS_REQUEST_TERMS


# ==================================================
# OUTPUT FILTER PATTERNS
# ==================================================

# --- Medicine output terms ---
MEDICINE_PATTERNS: list[str] = [
    # English
    "medicine",
    "medication",
    "drug",
    "tablet",
    "capsule",
    "antibiotic",
    "painkiller",
    "ibuprofen",
    "paracetamol",
    "naproxen",
    "aspirin",
    # Bangla
    "ওষুধ",
    "ঔষধ",
    "মেডিসিন",
    "ট্যাবলেট",
    "ক্যাপসুল",
    "অ্যান্টিবায়োটিক",
    "ব্যথার ওষুধ",
    "সিরাপ",
]

# --- Dosage patterns (regex-based, handles Bengali numerals too) ---
DOSAGE_REGEX_PATTERNS: list[str] = [
    r"\d+\s*mg",                          # 500 mg, 10mg
    r"\d+\s*ml",                           # 5 ml
    r"[০-৯]+\s*mg",                       # ৫০০ mg (Bangla numerals)
    r"\d+\s*tablets?",                     # 2 tablets
    r"\d+\s*capsules?",                    # 1 capsule
    r"[০-৯]+\s*ট্যাবলেট",                # ২ ট্যাবলেট
    r"twice\s+daily",                      # twice daily
    r"three\s+times\s+(a\s+)?day",        # three times a day
    r"once\s+a\s+day",                    # once a day
    r"দিনে\s*দুইবার",                    # দিনে দুইবার
    r"দিনে\s*তিনবার",                    # দিনে তিনবার
    r"দিনে\s*\d+\s*বার",                 # দিনে ২ বার
]

# --- Prescriptive phrases (output-level) ---
PRESCRIPTIVE_PATTERNS: list[str] = [
    # English
    "you should take",
    "take this medicine",
    "take this medication",
    "use this medication",
    "you need this medicine",
    "take two tablets",
    "take it twice daily",
    "take 500",
    "take 250",
    "take one tablet",
    # Bangla
    "আপনার এই ওষুধ খাওয়া উচিত",
    "এই ওষুধটি খান",
    "ওষুধটি দিনে দুইবার খান",
    "এই ওষুধ ব্যবহার করুন",
    "এই ওষুধ নিন",
    "দিনে দুইবার খাবেন",
    "দিনে তিনবার খাবেন",
    "এই ডোজ নিন",
    "খাবার পরে ওষুধটি নিন",
]

# --- Diagnosis claims (output-level) ---
DIAGNOSIS_PATTERNS: list[str] = [
    # English
    "you have pcos",
    "you have anemia",
    "you have endometriosis",
    "you have a condition",
    "this confirms pcos",
    "this means you have",
    "you are suffering from",
    "your symptoms indicate that you have",
    "this confirms your condition",
    # Bangla
    "আপনার pcos হয়েছে",
    "আপনার অ্যানিমিয়া হয়েছে",
    "আপনার এই রোগ হয়েছে",
    "আপনার pcos নিশ্চিত",
    "আপনার রোগ হয়েছে",
    "আপনার এই সমস্যা নিশ্চিত",
    "রোগ নির্ণয়",
    "ডায়াগনসিস",
]

# --- Medical causation claims (output-level) ---
# NOTE: These are STRONG causal claims. Association phrases like
# "may be associated with" or "আপনার লগে একটি সম্পর্ক দেখা যাচ্ছে" are SAFE.
CAUSATION_PATTERNS: list[str] = [
    # English – causal phrasing that implies medical certainty
    "causes your",
    "is caused by",
    "caused by pcos",
    "caused by anemia",
    "your cramps are because",
    "this symptom is caused by",
    "the reason for your",
    # Bangla
    "কম ঘুমের কারণে আপনার",
    "এর কারণে আপনার",
    "এই সমস্যার কারণ হলো",
    "এই লক্ষণটি এই রোগের কারণে",
    "এর ফলে আপনার",
]


# ==================================================
# PRIVATE HELPERS
# ==================================================

def _normalize(text: str) -> str:
    """
    Normalize text for comparison:
    - lowercase English characters
    - strip leading/trailing whitespace
    - collapse repeated whitespace to single space
    - preserves Bangla characters as-is
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # Only lowercase ASCII — Bangla codepoints are unaffected by .lower()
    text = text.lower()
    return text


def _matches_any_phrase(normalized_text: str, patterns: list[str]) -> bool:
    """Check whether normalized_text contains any of the given literal phrases."""
    return any(p in normalized_text for p in patterns)


def _matches_any_regex(normalized_text: str, patterns: list[str]) -> bool:
    """Check whether normalized_text matches any of the given regex patterns."""
    return any(re.search(p, normalized_text) for p in patterns)


# ==================================================
# PUBLIC API
# ==================================================

def input_risk_check(text: str) -> str:
    """
    Classify user input into a safety tier.

    Returns:
        "emergency"     — severe physical/mental health emergency
        "medical_risk"  — medical/prescriptive intent detected
        "safe"          — normal wellness conversation

    Priority:  emergency > medical_risk > safe
    """
    norm = _normalize(text)

    # --- LAYER 1: Emergency (highest priority) ---
    if _matches_any_phrase(norm, EMERGENCY_PATTERNS):
        return "emergency"

    # --- LAYER 2: Medical/prescriptive intent phrases ---
    if _matches_any_phrase(norm, MEDICAL_RISK_PATTERNS):
        return "medical_risk"

    # --- LAYER 3: Medicine / drug terms ---
    if _matches_any_phrase(norm, MEDICINE_TERMS):
        return "medical_risk"

    # --- LAYER 4: Dosage patterns via regex ---
    if _matches_any_regex(norm, DOSAGE_REGEX_PATTERNS):
        return "medical_risk"

    return "safe"


def output_post_filter(text: str) -> str:
    """
    Scan the complete LLM output for unsafe medical content.

    If ANY unsafe pattern is found, the ENTIRE response is replaced
    with SAFE_FALLBACK_MESSAGE — never partially edited.

    Returns:
        Original text if safe, or SAFE_FALLBACK_MESSAGE if unsafe.
    """
    norm = _normalize(text)

    # Check medicine names/terms
    if _matches_any_phrase(norm, MEDICINE_PATTERNS):
        return SAFE_FALLBACK_MESSAGE

    # Check prescriptive phrasing
    if _matches_any_phrase(norm, PRESCRIPTIVE_PATTERNS):
        return SAFE_FALLBACK_MESSAGE

    # Check diagnosis claims
    if _matches_any_phrase(norm, DIAGNOSIS_PATTERNS):
        return SAFE_FALLBACK_MESSAGE

    # Check strong medical causation claims
    if _matches_any_phrase(norm, CAUSATION_PATTERNS):
        return SAFE_FALLBACK_MESSAGE

    # Check dosage patterns via regex
    if _matches_any_regex(norm, DOSAGE_REGEX_PATTERNS):
        return SAFE_FALLBACK_MESSAGE

    return text
