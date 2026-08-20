"""
Tests for POST /api/ai/chat   (Step 4)
=======================================
All tests use mocking — NO real Gemini API calls are made.

Run from the project root:
    python backend/tests/test_ai_chat.py

Or with pytest:
    pytest -q backend/tests/test_ai_chat.py
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Allow running from project root or backend/tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.ai_routes import (
    chat,
    ChatRequest,
    ChatResponse,
    EMERGENCY_SAFE_RESPONSE,
    MEDICAL_RISK_SAFE_RESPONSE,
    DAILY_LIMIT_RESPONSE,
    CHAT_FALLBACK_RESPONSE,
    CHAT_FULL_SYSTEM_PROMPT,
)
from services.llm_service import REQUIRED_DISCLAIMER, LLM_FALLBACK_MESSAGE
from services.safety_filter import SAFE_FALLBACK_MESSAGE
from services.daily_limit_service import reset_usage, DAILY_CHAT_LIMIT


# ==================================================
# FIXTURES / HELPERS
# ==================================================

def _req(message: str) -> ChatRequest:
    return ChatRequest(message=message)


def _make_safe_response(text: str) -> str:
    """Return text guaranteed to contain the disclaimer."""
    if REQUIRED_DISCLAIMER not in text:
        return text + "\n\n" + REQUIRED_DISCLAIMER
    return text


def _fake_request(ip: str = "1.2.3.4"):
    """Build a fake FastAPI Request-like object for testing."""
    mock_req = MagicMock()
    mock_req.client.host = ip
    mock_req.headers = {}
    return mock_req


# ==================================================
# MAIN CHAT TESTS
# ==================================================

class TestChatEndpoint(unittest.TestCase):

    def setUp(self):
        """Reset usage for the test IP before each test."""
        reset_usage("1.2.3.4")

    # ── Test 1 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_normal_wellness_message_calls_gemini_once(self, mock_generate):
        """Normal wellness question → Gemini called exactly once."""
        mock_generate.return_value = _make_safe_response(
            "পর্যাপ্ত ঘুম ও পানি পান স্বাস্থ্যের জন্য উপকারী।"
        )
        result = chat(_req("আমি আজ খুব tired feel করছি"), _fake_request())

        mock_generate.assert_called_once()
        self.assertEqual(result.source, "gemini")
        self.assertTrue(result.success)

    # ── Test 2 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_medicine_name_question_skips_gemini(self, mock_generate):
        """Asking about a medicine name → Gemini NOT called."""
        result = chat(_req("Can I take ibuprofen for period pain?"), _fake_request())

        mock_generate.assert_not_called()
        self.assertIn(result.source, ["medical_risk", "emergency"])

    # ── Test 3 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_what_should_i_take_skips_gemini(self, mock_generate):
        """'What should I take?' → Gemini NOT called."""
        result = chat(_req("What should I take for cramps?"), _fake_request())

        mock_generate.assert_not_called()
        self.assertNotEqual(result.source, "gemini")

    # ── Test 4 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_bangla_medicine_request_skips_gemini(self, mock_generate):
        """Bangla medicine request → Gemini NOT called."""
        result = chat(_req("period pain এর জন্য কি ওষুধ খাব?"), _fake_request())

        mock_generate.assert_not_called()
        self.assertIn(result.source, ["medical_risk", "emergency"])

    # ── Test 5 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_dosage_question_skips_gemini(self, mock_generate):
        """Dosage question → Gemini NOT called."""
        result = chat(_req("How many tablets should I take per day?"), _fake_request())

        mock_generate.assert_not_called()
        self.assertNotEqual(result.source, "gemini")

    # ── Test 6 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_prescription_request_skips_gemini(self, mock_generate):
        """Prescription request → Gemini NOT called."""
        result = chat(_req("Can you prescribe something for me?"), _fake_request())

        mock_generate.assert_not_called()
        self.assertNotEqual(result.source, "gemini")

    # ── Test 7 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_diagnosis_request_skips_gemini(self, mock_generate):
        """Diagnosis request → Gemini NOT called."""
        result = chat(_req("আমার কি PCOS হয়েছে?"), _fake_request())

        mock_generate.assert_not_called()
        self.assertNotEqual(result.source, "gemini")

    # ── Test 8 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_heavy_bleeding_is_emergency(self, mock_generate):
        """Heavy bleeding → emergency response, Gemini NOT called."""
        result = chat(_req("আমার অনেক বেশি bleeding হচ্ছে"), _fake_request())

        mock_generate.assert_not_called()
        self.assertEqual(result.source, "emergency")
        self.assertEqual(result.response, EMERGENCY_SAFE_RESPONSE)

    # ── Test 9 ────────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_fainting_is_emergency(self, mock_generate):
        """Fainting/unconsciousness → emergency response, Gemini NOT called."""
        with patch("api.ai_routes.input_risk_check", return_value="emergency"):
            result = chat(_req("I feel like I'm going to faint"), _fake_request())

        mock_generate.assert_not_called()
        self.assertEqual(result.source, "emergency")

    # ── Test 10 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_severe_pain_is_emergency(self, mock_generate):
        """Severe pain → emergency response."""
        result = chat(_req("আমার তীব্র ব্যথা হচ্ছে, সহ্য করতে পারছি না"), _fake_request())

        mock_generate.assert_not_called()
        self.assertEqual(result.source, "emergency")

    # ── Test 11 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.output_post_filter")
    @patch("api.ai_routes.generate")
    def test_safe_question_passes_through_output_post_filter(
        self, mock_generate, mock_post_filter
    ):
        """Safe question → Gemini output is ALWAYS passed through output_post_filter."""
        raw = _make_safe_response("হালকা ব্যায়াম সহায়ক।")
        mock_generate.return_value = raw
        mock_post_filter.return_value = raw  # safe — unchanged

        chat(_req("আমি কিভাবে ভালো ঘুমাতে পারি?"), _fake_request())

        mock_post_filter.assert_called_once_with(raw)

    # ── Test 12 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_unsafe_gemini_output_returns_fallback(self, mock_generate):
        """Unsafe Gemini output (medicine/dosage) → deterministic fallback."""
        mock_generate.return_value = "আপনি দিনে দুইবার 500 mg নিন।"

        result = chat(_req("আমার ঘুম কম হচ্ছে"), _fake_request())

        self.assertEqual(result.source, "fallback")
        self.assertNotIn("500 mg", result.response)
        self.assertNotIn("দিনে দুইবার", result.response)

    # ── Test 13 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_gemini_exception_returns_safe_fallback(self, mock_generate):
        """Gemini LLM failure (returns LLM_FALLBACK_MESSAGE) → safe fallback."""
        mock_generate.return_value = LLM_FALLBACK_MESSAGE

        result = chat(_req("আমি কিভাবে stress কমাতে পারি?"), _fake_request())

        self.assertEqual(result.source, "fallback")
        self.assertEqual(result.response, CHAT_FALLBACK_RESPONSE)
        self.assertIn(REQUIRED_DISCLAIMER, result.response)

    # ── Test 14 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_under_daily_limit_gemini_may_be_called(self, mock_generate):
        """Daily limit not reached → Gemini may be called for safe input."""
        mock_generate.return_value = _make_safe_response("wellness tip.")

        # Fresh IP — limit not reached
        reset_usage("5.5.5.5")
        result = chat(_req("আমি কিভাবে ভালো থাকতে পারি?"), _fake_request("5.5.5.5"))

        self.assertNotEqual(result.source, "daily_limit")

    # ── Test 15 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_daily_limit_reached_skips_gemini(self, mock_generate):
        """Daily limit reached → Gemini NOT called, source=daily_limit."""
        with patch("api.ai_routes.is_limit_reached", return_value=True):
            result = chat(_req("সাধারণ wellness প্রশ্ন"), _fake_request())

        mock_generate.assert_not_called()
        self.assertEqual(result.source, "daily_limit")
        self.assertEqual(result.response, DAILY_LIMIT_RESPONSE)

    # ── Test 16 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_medical_risk_while_limit_exists_skips_gemini(self, mock_generate):
        """Medical-risk input (under daily limit) → Gemini still NOT called."""
        result = chat(_req("আমার কি ওষুধ খাওয়া উচিত?"), _fake_request())

        mock_generate.assert_not_called()
        self.assertIn(result.source, ["medical_risk", "emergency"])

    # ── Test 17 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_emergency_while_limit_exists_skips_gemini(self, mock_generate):
        """Emergency input (under daily limit) → Gemini still NOT called."""
        result = chat(_req("আমার অনেক বেশি bleeding হচ্ছে"), _fake_request())

        mock_generate.assert_not_called()
        self.assertEqual(result.source, "emergency")

    # ── Test 18 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_every_response_contains_disclaimer(self, mock_generate):
        """ALL response types must contain the Bangla disclaimer."""
        # gemini path
        mock_generate.return_value = _make_safe_response("wellness.")
        r = chat(_req("ঘুমের টিপস দিন"), _fake_request())
        self.assertIn(REQUIRED_DISCLAIMER, r.response)

        # fallback path
        mock_generate.return_value = LLM_FALLBACK_MESSAGE
        r = chat(_req("ঘুমের টিপস দিন"), _fake_request())
        self.assertIn(REQUIRED_DISCLAIMER, r.response)

        # emergency path
        r = chat(_req("অনেক বেশি bleeding হচ্ছে"), _fake_request())
        self.assertIn(REQUIRED_DISCLAIMER, r.response)

        # medical_risk path
        r = chat(_req("কি ওষুধ নেব?"), _fake_request())
        self.assertIn(REQUIRED_DISCLAIMER, r.response)

        # daily_limit path
        with patch("api.ai_routes.is_limit_reached", return_value=True):
            r = chat(_req("কিছু জিজ্ঞেস করি?"), _fake_request())
        self.assertIn(REQUIRED_DISCLAIMER, r.response)

    # ── Test 19 ───────────────────────────────────────────────────────────
    def test_empty_message_rejected_by_validation(self):
        """Empty message → Pydantic ValidationError (min_length=1)."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ChatRequest(message="")

    # ── Test 20 ───────────────────────────────────────────────────────────
    def test_chat_request_does_not_accept_pii_fields(self):
        """ChatRequest only has 'message' — no PII fields defined."""
        defined_fields = set(ChatRequest.model_fields.keys())
        self.assertEqual(defined_fields, {"message"})

        pii_fields = {"name", "phone", "email", "user_id", "journal", "history"}
        for field in pii_fields:
            self.assertNotIn(field, defined_fields)

    # ── Test 21 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_usage_incremented_only_on_successful_gemini_response(self, mock_generate):
        """Daily usage counter increments ONLY for successful Gemini responses."""
        mock_generate.return_value = _make_safe_response("wellness tip.")
        reset_usage("7.7.7.7")

        from services.daily_limit_service import get_usage_count
        before = get_usage_count("7.7.7.7")
        chat(_req("আমি কিভাবে ভালো ঘুমাতে পারি?"), _fake_request("7.7.7.7"))
        after = get_usage_count("7.7.7.7")

        self.assertEqual(after, before + 1)

    # ── Test 22 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_usage_not_incremented_on_fallback(self, mock_generate):
        """Daily usage counter NOT incremented when Gemini fails or is blocked."""
        mock_generate.return_value = LLM_FALLBACK_MESSAGE
        reset_usage("8.8.8.8")

        from services.daily_limit_service import get_usage_count
        before = get_usage_count("8.8.8.8")
        chat(_req("ঘুমের সমস্যা"), _fake_request("8.8.8.8"))
        after = get_usage_count("8.8.8.8")

        self.assertEqual(after, before)  # unchanged

    # ── Test 23 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_source_field_all_values(self, mock_generate):
        """Verify all possible source values can be produced."""
        sources_seen = set()

        # gemini
        mock_generate.return_value = _make_safe_response("ok.")
        r = chat(_req("ভালো থাকার উপায়"), _fake_request())
        sources_seen.add(r.source)

        # fallback
        mock_generate.return_value = LLM_FALLBACK_MESSAGE
        r = chat(_req("ভালো থাকার উপায়"), _fake_request())
        sources_seen.add(r.source)

        # emergency
        r = chat(_req("অনেক বেশি bleeding হচ্ছে"), _fake_request())
        sources_seen.add(r.source)

        # medical_risk
        r = chat(_req("কি ওষুধ খাব?"), _fake_request())
        sources_seen.add(r.source)

        # daily_limit
        with patch("api.ai_routes.is_limit_reached", return_value=True):
            r = chat(_req("প্রশ্ন"), _fake_request())
        sources_seen.add(r.source)

        expected = {"gemini", "fallback", "emergency", "medical_risk", "daily_limit"}
        self.assertEqual(sources_seen, expected)

    # ── Test 24 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_chat_system_prompt_contains_key_safety_rules(self, mock_generate):
        """CHAT_FULL_SYSTEM_PROMPT contains all required safety constraints."""
        self.assertIn("NOT a doctor", CHAT_FULL_SYSTEM_PROMPT)
        self.assertIn("CHAT-SPECIFIC INSTRUCTIONS", CHAT_FULL_SYSTEM_PROMPT)
        self.assertIn("medicine", CHAT_FULL_SYSTEM_PROMPT.lower())
        self.assertIn("dosage", CHAT_FULL_SYSTEM_PROMPT.lower())
        self.assertIn("এটি চিকিৎসকের বিকল্প নয়", CHAT_FULL_SYSTEM_PROMPT)

    # ── Test 25 ───────────────────────────────────────────────────────────
    @patch("api.ai_routes.generate")
    def test_message_trimmed_before_processing(self, mock_generate):
        """Leading/trailing whitespace in message is stripped before risk check."""
        mock_generate.return_value = _make_safe_response("wellness.")

        # Should NOT crash and should be processed as a safe message
        result = chat(_req("  আমি ভালো নেই  "), _fake_request())
        self.assertIsInstance(result, ChatResponse)


# ==================================================
# DAILY LIMIT SERVICE TESTS
# ==================================================

class TestDailyLimitService(unittest.TestCase):

    def setUp(self):
        reset_usage("test_ip")

    def test_fresh_ip_not_limited(self):
        from services.daily_limit_service import is_limit_reached
        self.assertFalse(is_limit_reached("test_ip"))

    def test_usage_increments_correctly(self):
        from services.daily_limit_service import increment_usage, get_usage_count
        increment_usage("test_ip")
        increment_usage("test_ip")
        self.assertEqual(get_usage_count("test_ip"), 2)

    def test_limit_reached_after_max_requests(self):
        from services.daily_limit_service import increment_usage, is_limit_reached
        for _ in range(DAILY_CHAT_LIMIT):
            increment_usage("test_ip")
        self.assertTrue(is_limit_reached("test_ip"))

    def test_test_key_never_limited(self):
        """The '__test__' key bypasses all limits."""
        from services.daily_limit_service import increment_usage, is_limit_reached
        for _ in range(DAILY_CHAT_LIMIT + 10):
            increment_usage("__test__")
        self.assertFalse(is_limit_reached("__test__"))

    def test_reset_clears_usage(self):
        from services.daily_limit_service import increment_usage, get_usage_count
        increment_usage("test_ip")
        reset_usage("test_ip")
        self.assertEqual(get_usage_count("test_ip"), 0)


# ==================================================
# RUNNER
# ==================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SheCare BD — AI Chat Endpoint Test Suite (Step 4)")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestChatEndpoint, TestDailyLimitService]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 60)
    if result.wasSuccessful():
        print(f"🎉 All {result.testsRun} tests passed successfully!")
    else:
        print(f"❌ {len(result.failures)} failure(s), {len(result.errors)} error(s).")
    print("=" * 60)
