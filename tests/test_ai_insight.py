"""
Tests for POST /api/ai/insight   (Step 3)
==========================================
All tests use mocking — NO real Gemini API calls are made.

Run from the project root:
    python backend/tests/test_ai_insight.py

Or with pytest (if available):
    pytest -q backend/tests/test_ai_insight.py
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Allow running from project root or backend/tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.ai_routes import (
    generate_insight,
    InsightPayload,
    InsightResponse,
    EMERGENCY_SAFE_RESPONSE,
    MEDICAL_RISK_SAFE_RESPONSE,
    _build_analytics_summary,
    _build_deterministic_fallback,
    INSIGHT_FULL_SYSTEM_PROMPT,
)
from services.llm_service import REQUIRED_DISCLAIMER, LLM_FALLBACK_MESSAGE
from services.safety_filter import SAFE_FALLBACK_MESSAGE


# ==================================================
# FIXTURES
# ==================================================

def _normal_payload(**overrides) -> InsightPayload:
    """A standard, safe analytics payload."""
    data = {
        "mood_average": 3.5,
        "average_sleep": 6.0,
        "cycle_average": 29,
        "frequent_symptoms": ["cramps", "fatigue"],
    }
    data.update(overrides)
    return InsightPayload(**data)


def _make_gemini_text(text: str) -> str:
    """Ensure a response contains the disclaimer."""
    if REQUIRED_DISCLAIMER not in text:
        return text + "\n\n" + REQUIRED_DISCLAIMER
    return text


# ==================================================
# TESTS
# ==================================================

class TestInsightEndpoint(unittest.TestCase):

    # ── Test 1 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_safe_payload_calls_gemini_once(self, mock_generate):
        """Normal analytics payload → Gemini is called exactly once."""
        mock_generate.return_value = _make_gemini_text(
            "আপনার লগে ঘুম ও ক্লান্তির একটি সম্পর্ক দেখা যাচ্ছে।"
        )

        payload = _normal_payload()
        result = generate_insight(payload)

        mock_generate.assert_called_once()
        self.assertEqual(result.source, "gemini")
        self.assertTrue(result.success)

    # ── Test 2 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.output_post_filter")
    @patch("api.ai_routes.generate")
    def test_gemini_output_passes_through_output_post_filter(
        self, mock_generate, mock_post_filter
    ):
        """Gemini output is always passed through output_post_filter."""
        raw = _make_gemini_text("পর্যাপ্ত ঘুম সহায়ক হতে পারে।")
        mock_generate.return_value = raw
        mock_post_filter.return_value = raw  # safe — returns unchanged

        generate_insight(_normal_payload())

        mock_post_filter.assert_called_once_with(raw)

    # ── Test 3 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_gemini_failure_returns_deterministic_fallback(self, mock_generate):
        """Gemini LLM failure (returns LLM_FALLBACK_MESSAGE) → deterministic fallback."""
        mock_generate.return_value = LLM_FALLBACK_MESSAGE

        payload = _normal_payload()
        result = generate_insight(payload)

        self.assertEqual(result.source, "fallback")
        self.assertTrue(result.success)
        self.assertIn(REQUIRED_DISCLAIMER, result.insight)
        # Fallback must NOT be the raw LLM error message
        self.assertNotEqual(result.insight, LLM_FALLBACK_MESSAGE)

    # ── Test 4 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_empty_gemini_response_triggers_fallback(self, mock_generate):
        """
        Simulate generate() returning LLM_FALLBACK_MESSAGE
        (which is what llm_service returns on empty response).
        → deterministic fallback returned.
        """
        mock_generate.return_value = LLM_FALLBACK_MESSAGE

        result = generate_insight(_normal_payload())

        self.assertEqual(result.source, "fallback")
        self.assertNotIn("AI wellness insight তৈরি করা সম্ভব হচ্ছে না", result.insight)

    # ── Test 5 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_unsafe_gemini_output_triggers_deterministic_fallback(self, mock_generate):
        """
        Gemini returns text containing medicine/dosage → output_post_filter rejects it
        → deterministic fallback returned with source="fallback".
        """
        unsafe = "আপনার ব্যথার জন্য 500 mg ওষুধ দিনে দুইবার নিন।"
        mock_generate.return_value = unsafe  # simulate unsafe output

        result = generate_insight(_normal_payload())

        # output_post_filter should have caught this → fallback
        self.assertEqual(result.source, "fallback")
        self.assertNotIn("500 mg", result.insight)
        self.assertNotIn("ওষুধ দিনে দুইবার", result.insight)

    # ── Test 6 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_medical_risk_input_skips_gemini(self, mock_generate):
        """
        Analytics summary containing a medical-risk phrase → Gemini NOT called.
        source = "safety_fallback".
        """
        # Inject a medical-risk symptom description into the summary
        # The input_risk_check operates on the analytics summary string.
        # We simulate this by using a symptom list containing a medicine request phrase.
        # Since input_risk_check checks the summary, we need to craft a payload whose
        # _build_analytics_summary output contains a medical-risk phrase.
        # The easiest approach: mock input_risk_check directly.
        with patch("api.ai_routes.input_risk_check", return_value="medical_risk"):
            result = generate_insight(_normal_payload())

        mock_generate.assert_not_called()
        self.assertEqual(result.source, "safety_fallback")
        self.assertEqual(result.insight, MEDICAL_RISK_SAFE_RESPONSE)

    # ── Test 7 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_emergency_input_skips_gemini(self, mock_generate):
        """Emergency phrase in analytics summary → Gemini NOT called."""
        with patch("api.ai_routes.input_risk_check", return_value="emergency"):
            result = generate_insight(_normal_payload())

        mock_generate.assert_not_called()
        self.assertEqual(result.source, "safety_fallback")
        self.assertEqual(result.insight, EMERGENCY_SAFE_RESPONSE)

    # ── Test 8 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_response_always_contains_disclaimer(self, mock_generate):
        """All responses (gemini, fallback, safety_fallback) include the disclaimer."""
        # Gemini path
        mock_generate.return_value = _make_gemini_text("wellness tip.")
        gemini_result = generate_insight(_normal_payload())
        self.assertIn(REQUIRED_DISCLAIMER, gemini_result.insight)

        # Fallback path
        mock_generate.return_value = LLM_FALLBACK_MESSAGE
        fallback_result = generate_insight(_normal_payload())
        self.assertIn(REQUIRED_DISCLAIMER, fallback_result.insight)

        # Safety fallback path
        with patch("api.ai_routes.input_risk_check", return_value="emergency"):
            safety_result = generate_insight(_normal_payload())
        self.assertIn(REQUIRED_DISCLAIMER, safety_result.insight)

    # ── Test 9 ────────────────────────────────────────────────────────────
    def test_insight_payload_rejects_pii_fields(self):
        """InsightPayload must NOT accept PII fields like name, phone, journal."""
        # Extra fields should be silently ignored (Pydantic default) or rejected.
        # The key assertion: raw DailyLog fields are not defined on InsightPayload.
        payload_fields = set(InsightPayload.model_fields.keys())

        pii_fields = {"name", "phone", "email", "journal", "user_id", "period"}
        for field in pii_fields:
            self.assertNotIn(
                field,
                payload_fields,
                f"PII field '{field}' must not be accepted by InsightPayload",
            )

        # Only the 4 analytics fields are allowed
        allowed = {"mood_average", "average_sleep", "cycle_average", "frequent_symptoms"}
        self.assertEqual(payload_fields, allowed)

    # ── Test 10 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_source_field_values_are_correct(self, mock_generate):
        """Verify correct source values: gemini, fallback, safety_fallback."""
        # gemini
        mock_generate.return_value = _make_gemini_text("wellness tip.")
        r = generate_insight(_normal_payload())
        self.assertEqual(r.source, "gemini")

        # fallback (LLM fails)
        mock_generate.return_value = LLM_FALLBACK_MESSAGE
        r = generate_insight(_normal_payload())
        self.assertEqual(r.source, "fallback")

        # safety_fallback (emergency)
        with patch("api.ai_routes.input_risk_check", return_value="emergency"):
            r = generate_insight(_normal_payload())
        self.assertEqual(r.source, "safety_fallback")

        # safety_fallback (medical_risk)
        with patch("api.ai_routes.input_risk_check", return_value="medical_risk"):
            r = generate_insight(_normal_payload())
        self.assertEqual(r.source, "safety_fallback")

    # ── Test 11 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_normal_symptoms_not_over_blocked(self, mock_generate):
        """
        Ordinary symptom categories (cramps, fatigue, headache, bloating)
        must NOT trigger safety_fallback.
        """
        normal_gemini_response = _make_gemini_text(
            "আপনার লগে cramps ও fatigue একই সময়ে বেশি দেখা যাচ্ছে।"
        )
        mock_generate.return_value = normal_gemini_response

        for symptoms in [
            ["cramps"],
            ["fatigue"],
            ["headache"],
            ["bloating"],
            ["acne"],
            ["cramps", "fatigue", "headache"],
        ]:
            payload = _normal_payload(frequent_symptoms=symptoms)
            result = generate_insight(payload)
            self.assertNotEqual(
                result.source,
                "safety_fallback",
                f"Ordinary symptoms {symptoms} must NOT trigger safety_fallback",
            )

    # ── Test 12 ───────────────────────────────────────────────────────────
    def test_insight_payload_field_validation(self):
        """InsightPayload enforces sensible field ranges."""
        from pydantic import ValidationError

        # mood_average out of range
        with self.assertRaises(ValidationError):
            InsightPayload(mood_average=10.0)  # > 5.0

        # average_sleep out of range
        with self.assertRaises(ValidationError):
            InsightPayload(average_sleep=30.0)  # > 24.0

        # cycle_average out of range
        with self.assertRaises(ValidationError):
            InsightPayload(cycle_average=0)  # < 1

        # symptom label too long
        with self.assertRaises(ValidationError):
            InsightPayload(frequent_symptoms=["x" * 81])

    # ── Test 13 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_system_prompt_passed_contains_insight_instructions(self, mock_generate):
        """INSIGHT_FULL_SYSTEM_PROMPT is passed to generate() for the insight route."""
        mock_generate.return_value = _make_gemini_text("wellness.")

        generate_insight(_normal_payload())

        call_kwargs = mock_generate.call_args
        passed_system_prompt = call_kwargs[1].get("system_prompt") or call_kwargs[0][0]
        self.assertIn("INSIGHT-SPECIFIC INSTRUCTIONS", passed_system_prompt)

    # ── Test 14 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_deterministic_fallback_uses_actual_payload_values(self, mock_generate):
        """Deterministic fallback dynamically includes real values from the payload."""
        mock_generate.return_value = LLM_FALLBACK_MESSAGE

        payload = InsightPayload(
            mood_average=2.5,
            average_sleep=5.5,
            cycle_average=28,
            frequent_symptoms=["ব্যথা", "ক্লান্তি"],
        )
        result = generate_insight(payload)

        self.assertEqual(result.source, "fallback")
        # Should contain actual sleep value (5.5)
        self.assertIn("5.5", result.insight)

    # ── Test 15 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_success_field_always_true(self, mock_generate):
        """InsightResponse.success is always True (errors handled gracefully)."""
        # Gemini path
        mock_generate.return_value = _make_gemini_text("ok.")
        self.assertTrue(generate_insight(_normal_payload()).success)

        # Fallback path
        mock_generate.return_value = LLM_FALLBACK_MESSAGE
        self.assertTrue(generate_insight(_normal_payload()).success)

        # Safety path
        with patch("api.ai_routes.input_risk_check", return_value="emergency"):
            self.assertTrue(generate_insight(_normal_payload()).success)


# ==================================================
# ANALYTICS SUMMARY HELPER TESTS
# ==================================================

class TestAnalyticsSummary(unittest.TestCase):

    def test_summary_includes_all_fields(self):
        payload = _normal_payload()
        summary = _build_analytics_summary(payload)
        self.assertIn("mood", summary.lower())
        self.assertIn("sleep", summary.lower())
        self.assertIn("cycle", summary.lower())
        self.assertIn("cramps", summary)
        self.assertIn("fatigue", summary)

    def test_summary_with_no_symptoms(self):
        payload = InsightPayload(frequent_symptoms=[])
        summary = _build_analytics_summary(payload)
        self.assertIn("No specific symptoms", summary)

    def test_summary_does_not_contain_pii(self):
        payload = _normal_payload()
        summary = _build_analytics_summary(payload)
        # No names, no phone numbers, no email
        for pii_marker in ["@", "01", "name", "phone"]:
            self.assertNotIn(pii_marker, summary.lower())


# ==================================================
# DETERMINISTIC FALLBACK TESTS
# ==================================================

class TestDeterministicFallback(unittest.TestCase):

    def test_fallback_contains_disclaimer(self):
        payload = _normal_payload()
        result = _build_deterministic_fallback(payload)
        self.assertIn(REQUIRED_DISCLAIMER, result)

    def test_fallback_no_causal_language(self):
        payload = _normal_payload()
        result = _build_deterministic_fallback(payload)
        # Must not contain causal claims
        self.assertNotIn("কারণে", result)
        self.assertNotIn("causes", result.lower())

    def test_fallback_uses_real_sleep_value(self):
        payload = InsightPayload(average_sleep=7.3, frequent_symptoms=["fatigue"])
        result = _build_deterministic_fallback(payload)
        self.assertIn("7.3", result)

    def test_fallback_with_empty_payload(self):
        payload = InsightPayload()
        result = _build_deterministic_fallback(payload)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertIn(REQUIRED_DISCLAIMER, result)


# ==================================================
# RUNNER
# ==================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SheCare BD — AI Insight Endpoint Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestInsightEndpoint, TestAnalyticsSummary, TestDeterministicFallback]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 60)
    if result.wasSuccessful():
        print(f"🎉 All {result.testsRun} tests passed successfully!")
    else:
        print(f"❌ {len(result.failures)} failure(s), {len(result.errors)} error(s).")
    print("=" * 60)
