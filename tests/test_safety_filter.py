"""
Tests for backend/services/safety_filter.py
============================================
Run from the project root with:
    pytest -q backend/tests/test_safety_filter.py

Or directly:
    python backend/tests/test_safety_filter.py
"""
import sys
import os

# Allow running from project root or from backend/tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.safety_filter import (
    input_risk_check,
    output_post_filter,
    SAFE_FALLBACK_MESSAGE,
)

# ==================================================
# HELPERS
# ==================================================

def assert_eq(actual, expected, label=""):
    if actual != expected:
        raise AssertionError(
            f"[FAIL] {label}\n  Expected : {expected!r}\n  Got      : {actual!r}"
        )
    print(f"  ✅ PASS  {label}")


# ==================================================
# INPUT RISK CHECK TESTS  (≥10 required)
# ==================================================

def test_input_risk_check():
    print("\n── input_risk_check ──────────────────────────────────────────")

    # 1. Normal English wellness question → safe
    assert_eq(
        input_risk_check("How can I improve my sleep?"),
        "safe",
        "Normal English wellness → safe",
    )

    # 2. Normal Bangla wellness question → safe
    assert_eq(
        input_risk_check("কীভাবে ভালো ঘুম হতে পারে?"),
        "safe",
        "Normal Bangla wellness → safe",
    )

    # 3. Mixed Bangla-English wellness question → safe
    assert_eq(
        input_risk_check("PMS এর সময় mood কেন change হয়?"),
        "safe",
        "Mixed Bangla-English wellness → safe",
    )

    # 4. Period/cycle question → safe (must NOT block ordinary health words)
    assert_eq(
        input_risk_check("আমার period সাধারণত কতদিন চলে?"),
        "safe",
        "Period question → safe (must not over-block)",
    )

    # 4a. Pregnancy food → safe
    assert_eq(
        input_risk_check("আমি গর্ভাবস্থায় কী ধরনের খাবার খাব?"),
        "safe",
        "Pregnancy food → safe",
    )

    # 4b. Pregnancy healthy food → safe
    assert_eq(
        input_risk_check("pregnancy-তে healthy খাবার কী কী?"),
        "safe",
        "Pregnancy healthy food → safe",
    )

    # 4c. Suggest healthy food → safe
    assert_eq(
        input_risk_check("Suggest me some healthy food during pregnancy"),
        "safe",
        "Suggest healthy food → safe",
    )

    # 4d. Pregnancy exercise → safe
    assert_eq(
        input_risk_check("গর্ভাবস্থায় কী ধরনের হালকা exercise করা যায়?"),
        "safe",
        "Pregnancy exercise → safe",
    )

    # 4e. Pregnancy sleep routine → safe
    assert_eq(
        input_risk_check("pregnancy-তে sleep routine কীভাবে ভালো করা যায়?"),
        "safe",
        "Pregnancy sleep routine → safe",
    )

    # 5. 'What should I take?' → medical_risk
    assert_eq(
        input_risk_check("What should I take?"),
        "medical_risk",
        "'What should I take?' → medical_risk",
    )

    # 6. Bangla medicine question → medical_risk
    assert_eq(
        input_risk_check("কোন ওষুধ খাব?"),
        "medical_risk",
        "Bangla medicine question → medical_risk",
    )

    # 7. Medicine keyword → medical_risk
    assert_eq(
        input_risk_check("What medicine is best for cramps?"),
        "medical_risk",
        "English 'medicine' keyword → medical_risk",
    )

    # 8. Dosage question → medical_risk
    assert_eq(
        input_risk_check("How many tablets should I take?"),
        "medical_risk",
        "Dosage question → medical_risk",
    )

    # 9. Numeric dosage → medical_risk
    assert_eq(
        input_risk_check("Can I take 500 mg for pain?"),
        "medical_risk",
        "Numeric dosage pattern → medical_risk",
    )

    # 9a. 500 mg ki khabo → medical_risk
    assert_eq(
        input_risk_check("500 mg কি খাব?"),
        "medical_risk",
        "500 mg ki khabo → medical_risk",
    )

    # 10. Prescription request → medical_risk
    assert_eq(
        input_risk_check("Can you prescribe something for me?"),
        "medical_risk",
        "Prescription request → medical_risk",
    )

    # 11. Bangla diagnosis request → medical_risk
    assert_eq(
        input_risk_check("আমার কি PCOS হয়েছে?"),
        "medical_risk",
        "Bangla diagnosis request → medical_risk",
    )

    # 12. Mixed Bangla-English medical → medical_risk
    assert_eq(
        input_risk_check("ki medicine khabo period pain er jonno?"),
        "medical_risk",
        "Mixed Bangla-English medical → medical_risk",
    )

    # 12a. What treatment should I follow → medical_risk
    assert_eq(
        input_risk_check("What treatment should I follow?"),
        "medical_risk",
        "Treatment follow → medical_risk",
    )

    # 12b. Gestational diabetes diagnosis → medical_risk
    assert_eq(
        input_risk_check("আমার কি gestational diabetes হয়েছে?"),
        "medical_risk",
        "Gestational diabetes diagnosis → medical_risk",
    )

    # 13. Tablet keyword → medical_risk
    assert_eq(
        input_risk_check("কোন tablet নেব?"),
        "medical_risk",
        "Bangla tablet keyword → medical_risk",
    )

    # 14. Heavy bleeding → emergency (not medical_risk)
    assert_eq(
        input_risk_check("I have heavy bleeding"),
        "emergency",
        "Heavy bleeding → emergency",
    )

    # 15. Fainting → emergency
    assert_eq(
        input_risk_check("I am fainting, what should I do?"),
        "emergency",
        "Fainting → emergency",
    )

    # 16. Severe pain in Bangla → emergency
    assert_eq(
        input_risk_check("আমার প্রচণ্ড ব্যথা হচ্ছে"),
        "emergency",
        "Bangla severe pain → emergency",
    )

    # 17. Unconscious → emergency
    assert_eq(
        input_risk_check("She is unconscious, help!"),
        "emergency",
        "Unconscious → emergency",
    )

    # 18. Suicidal thoughts → emergency
    assert_eq(
        input_risk_check("I am having suicidal thoughts"),
        "emergency",
        "Suicidal thoughts → emergency",
    )

    # 19. Shorthness of breath in Bangla → emergency
    assert_eq(
        input_risk_check("আমার শ্বাসকষ্ট হচ্ছে"),
        "emergency",
        "Bangla shortness of breath → emergency",
    )

    # 20. PRIORITY: emergency beats medical_risk
    assert_eq(
        input_risk_check("আমার খুব বেশি রক্তপাত হচ্ছে, কোন ওষুধ খাব?"),
        "emergency",
        "Emergency + medicine request → emergency (priority test)",
    )

    # 21. PMS word alone → safe
    assert_eq(
        input_risk_check("What lifestyle habits may help during PMS?"),
        "safe",
        "PMS education question → safe (must not over-block)",
    )

    # 22. Hydration question → safe
    assert_eq(
        input_risk_check("How can I stay hydrated?"),
        "safe",
        "Hydration question → safe",
    )


# ==================================================
# OUTPUT POST FILTER TESTS  (≥10 required)
# ==================================================

def test_output_post_filter():
    print("\n── output_post_filter ────────────────────────────────────────")

    # 1. Safe English wellness response → unchanged
    safe_en = (
        "Drinking more water and maintaining a consistent sleep schedule "
        "can support your overall wellness."
    )
    assert_eq(output_post_filter(safe_en), safe_en, "Safe English response → unchanged")

    # 2. Safe Bangla wellness response → unchanged
    safe_bn = (
        "আপনার লগ অনুযায়ী পর্যাপ্ত ঘুম ও hydration-এর দিকে নজর রাখা সহায়ক হতে পারে।"
    )
    assert_eq(output_post_filter(safe_bn), safe_bn, "Safe Bangla response → unchanged")

    # 3. Safe association-based Bangla response → unchanged
    safe_assoc = (
        "আপনার লগে কম ঘুম এবং ক্লান্তি একই সময়ে বেশি দেখা যাচ্ছে। "
        "এটি একটি সম্পর্ক হতে পারে।"
    )
    assert_eq(output_post_filter(safe_assoc), safe_assoc, "Safe association statement → unchanged")

    # 4. Medicine-related response → replaced
    assert_eq(
        output_post_filter("You can take some medicine for the pain."),
        SAFE_FALLBACK_MESSAGE,
        "Medicine reference in output → replaced",
    )

    # 5. Numeric dosage pattern → replaced
    assert_eq(
        output_post_filter("Take 500 mg of this supplement daily."),
        SAFE_FALLBACK_MESSAGE,
        "Numeric dosage (500 mg) → replaced",
    )

    # 6. Bangla dosage pattern → replaced
    assert_eq(
        output_post_filter("দিনে দুইবার খাবেন।"),
        SAFE_FALLBACK_MESSAGE,
        "Bangla dosage phrase → replaced",
    )

    # 7. Prescriptive phrase → replaced
    assert_eq(
        output_post_filter("আপনার এই ওষুধ খাওয়া উচিত।"),
        SAFE_FALLBACK_MESSAGE,
        "Bangla prescriptive phrase → replaced",
    )

    # 8. English prescriptive phrase → replaced
    assert_eq(
        output_post_filter("You should take two tablets in the morning."),
        SAFE_FALLBACK_MESSAGE,
        "English prescriptive phrase → replaced",
    )

    # 9. Diagnosis claim → replaced
    assert_eq(
        output_post_filter("Based on your symptoms, you have PCOS."),
        SAFE_FALLBACK_MESSAGE,
        "Diagnosis claim 'you have pcos' → replaced",
    )

    # 10. Bangla diagnosis claim → replaced
    assert_eq(
        output_post_filter("আপনার PCOS হয়েছে, তাই এই সমস্যা হচ্ছে।"),
        SAFE_FALLBACK_MESSAGE,
        "Bangla diagnosis claim → replaced",
    )

    # 11. Causal medical statement → replaced
    assert_eq(
        output_post_filter("Lack of sleep causes your cramps."),
        SAFE_FALLBACK_MESSAGE,
        "Causation claim 'causes your' → replaced",
    )

    # 12. Bangla causation claim → replaced
    assert_eq(
        output_post_filter("কম ঘুমের কারণে আপনার cramps বাড়ছে।"),
        SAFE_FALLBACK_MESSAGE,
        "Bangla causation claim → replaced",
    )

    # 13. Mixed Bangla-English unsafe response → replaced
    assert_eq(
        output_post_filter("আপনার সমস্যার জন্য এই medicine নিন।"),
        SAFE_FALLBACK_MESSAGE,
        "Mixed Bangla-English unsafe response → replaced",
    )

    # 14. Multi-sentence response — unsafe part causes ENTIRE replacement
    multi_sentence = (
        "আপনার লগ অনুযায়ী সব ঠিক আছে। "
        "তবে আপনার cramps কমানোর জন্য 500 mg medicine দিনে দুইবার নিন।"
    )
    assert_eq(
        output_post_filter(multi_sentence),
        SAFE_FALLBACK_MESSAGE,
        "Multi-sentence — ENTIRE response replaced (not partial)",
    )

    # 15. Tablet count pattern → replaced
    assert_eq(
        output_post_filter("Take 2 tablets with water each morning."),
        SAFE_FALLBACK_MESSAGE,
        "Tablet count pattern → replaced",
    )

    # 16. Antibiotic mention → replaced
    assert_eq(
        output_post_filter("You may need an antibiotic for this."),
        SAFE_FALLBACK_MESSAGE,
        "Antibiotic mention → replaced",
    )

    # 17. Diagnosis claim variant → replaced
    assert_eq(
        output_post_filter("This confirms PCOS based on your data."),
        SAFE_FALLBACK_MESSAGE,
        "Diagnosis confirmation → replaced",
    )


# ==================================================
# RUNNER
# ==================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SheCare BD — Safety Filter Test Suite")
    print("=" * 60)

    errors = []

    for fn in [test_input_risk_check, test_output_post_filter]:
        try:
            fn()
        except AssertionError as e:
            errors.append(str(e))
            print(str(e))

    print("\n" + "=" * 60)
    if errors:
        print(f"❌ {len(errors)} test(s) FAILED.")
    else:
        total = 22 + 17  # 22 input tests + 17 output tests
        print(f"🎉 All {total} assertions passed successfully!")
    print("=" * 60)
