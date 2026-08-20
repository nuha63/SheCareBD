import re

UNSAFE_KEYWORDS = [
    "medicine", "prescription", "dosage", "diagnose", "diagnosis", 
    "emergency", "suicide", "bleeding heavily", "severe pain", "treatment", "pill", "drug"
]

def classify_input(text: str) -> str:
    """
    Classifies user input into SAFE, MEDICAL_RISK, or EMERGENCY.
    """
    text_lower = text.lower()
    
    # Emergency keywords
    emergency_keywords = [
        "emergency", "suicide", "severe pain", "bleeding heavily", "heart attack", "can't breathe",
        "প্রচণ্ড ব্যথা", "অনেক বেশি bleeding", "অনেক রক্ত", "মারা যাব", "জরুরী", "তীব্র ব্যথা"
    ]
    if any(k in text_lower for k in emergency_keywords):
        return "EMERGENCY"
        
    # Medical risk keywords
    medical_keywords = [
        "medicine", "prescription", "dosage", "diagnose", "diagnosis", "treatment", "pill", "drug",
        "ওষুধ", "ঔষধ", "ডাক্তার দেখাব", "কী খাব", "pcos আছে?", "cancer", "ক্যান্সার", "pregnant", "প্রেগন্যান্ট", "রোগ", "ধরে নেই"
    ]
    if any(k in text_lower for k in medical_keywords):
        return "MEDICAL_RISK"
        
    return "SAFE"

def get_hardcoded_safe_response(classification: str) -> str:
    if classification in ["EMERGENCY", "MEDICAL_RISK"]:
        return "এই ধরনের গুরুতর উপসর্গের ক্ষেত্রে অনলাইনে পরামর্শের উপর নির্ভর না করে দ্রুত একজন qualified healthcare professional-এর সাহায্য নিন।"
    return ""

def apply_output_filter(text: str) -> str:
    """
    Post-filter for LLM generated text.
    If the LLM accidentally generates medical advice, replace the response.
    """
    # Look for bengali words related to medicine/prescription
    unsafe_bengali_words = ["ওষুধ", "ঔষধ", "প্রেসক্রিপশন", "চিকিৎসা", "ডায়াগনসিস", "রোগ নির্ণয়"]
    
    for word in unsafe_bengali_words:
        if word in text:
            return "দুঃখিত, আমি ওষুধ বা চিকিৎসার পরামর্শ দিতে পারি না। অনুগ্রহ করে ডাক্তারের সাথে কথা বলুন।"
            
    # Look for english words that slipped through
    for word in UNSAFE_KEYWORDS:
        if word in text.lower():
            return "দুঃখিত, আমি ওষুধ বা চিকিৎসার পরামর্শ দিতে পারি না। অনুগ্রহ করে ডাক্তারের সাথে কথা বলুন।"
            
    return text
