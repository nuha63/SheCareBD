"""
Tests for backend/services/llm_service.py  (google-genai SDK)
==============================================================
All tests use mocking — NO real Gemini API calls are made.

Mock structure for the new google-genai SDK:
    genai.Client(api_key=...) → mock_client
    mock_client.models.generate_content(...) → mock_response
    mock_response.text → str

Run from the project root:
    python backend/tests/test_llm_service.py

Or with pytest:
    pytest -q backend/tests/test_llm_service.py
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Allow running from project root or backend/tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.llm_service import (
    generate,
    SHARED_WELLNESS_SYSTEM_PROMPT,
    REQUIRED_DISCLAIMER,
    LLM_FALLBACK_MESSAGE,
    GEMINI_MODEL,
    _ensure_disclaimer,
    _extract_text,
)


# ==================================================
# HELPERS
# ==================================================

def _make_mock_response(text):
    """Build a fake Gemini response object with a .text attribute."""
    mock_resp = MagicMock()
    mock_resp.text = text
    return mock_resp


def _make_mock_client(response_text):
    """
    Build a mock genai.Client where client.models.generate_content()
    returns a mock response with the given text.
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(response_text)
    return mock_client


# ==================================================
# TEST SUITE
# ==================================================

class TestLlmService(unittest.TestCase):

    # ── Test 1 ────────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_successful_response_with_disclaimer_returned(self, mock_genai):
        """generate() returns expected wellness text when Gemini succeeds."""
        safe_text = (
            "আপনার লগে দেখা যাচ্ছে পর্যাপ্ত ঘুম না হলে ক্লান্তি বেশি থাকে। "
            + REQUIRED_DISCLAIMER
        )
        mock_genai.Client.return_value = _make_mock_client(safe_text)

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="আমার ঘুম কম হলে কী হতে পারে?",
        )

        self.assertEqual(result, safe_text)
        self.assertIsInstance(result, str)

    # ── Test 2 ────────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_empty_response_returns_fallback(self, mock_genai):
        """generate() returns LLM_FALLBACK_MESSAGE when Gemini returns empty text."""
        mock_genai.Client.return_value = _make_mock_client("")

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="test",
        )

        self.assertEqual(result, LLM_FALLBACK_MESSAGE)

    # ── Test 3 ────────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_none_response_text_returns_fallback(self, mock_genai):
        """generate() returns LLM_FALLBACK_MESSAGE when response.text is None."""
        mock_genai.Client.return_value = _make_mock_client(None)

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="test",
        )

        self.assertEqual(result, LLM_FALLBACK_MESSAGE)

    # ── Test 4 ────────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_api_exception_returns_fallback(self, mock_genai):
        """generate() returns LLM_FALLBACK_MESSAGE when Gemini raises an exception."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Network error")
        mock_genai.Client.return_value = mock_client

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="test",
        )

        self.assertEqual(result, LLM_FALLBACK_MESSAGE)
        # Exception message must NOT be exposed
        self.assertNotIn("Network error", result)

    # ── Test 5 ────────────────────────────────────────────────────────────
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_fallback(self):
        """generate() returns LLM_FALLBACK_MESSAGE when GEMINI_API_KEY is absent."""
        os.environ.pop("GEMINI_API_KEY", None)

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="test",
        )

        self.assertEqual(result, LLM_FALLBACK_MESSAGE)
        self.assertNotIn("fake", result)
        self.assertNotIn("key", result.lower())

    # ── Test 6 ────────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_missing_disclaimer_is_appended(self, mock_genai):
        """If Gemini forgets the disclaimer, generate() appends it automatically."""
        text_without_disclaimer = "পর্যাপ্ত ঘুম ও পানি পান সুস্বাস্থ্যের জন্য সহায়ক।"
        mock_genai.Client.return_value = _make_mock_client(text_without_disclaimer)

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="ঘুমের টিপস দিন",
        )

        self.assertIn(REQUIRED_DISCLAIMER, result)
        self.assertIn(text_without_disclaimer, result)

    # ── Test 7 ────────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_existing_disclaimer_not_duplicated(self, mock_genai):
        """If Gemini already includes the disclaimer, it must NOT be added twice."""
        text_with_disclaimer = (
            "পর্যাপ্ত ঘুম ও পানি পান সুস্বাস্থ্যের জন্য সহায়ক।\n\n"
            + REQUIRED_DISCLAIMER
        )
        mock_genai.Client.return_value = _make_mock_client(text_with_disclaimer)

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="test",
        )

        count = result.count(REQUIRED_DISCLAIMER)
        self.assertEqual(count, 1, f"Disclaimer duplicated! Found {count} times.")

    # ── Test 8 ────────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_system_prompt_passed_to_gemini(self, mock_genai):
        """The system_prompt is included in the contents sent to Gemini."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_mock_response(
            "wellness tip. " + REQUIRED_DISCLAIMER
        )
        mock_genai.Client.return_value = mock_client

        custom_prompt = "CUSTOM_SYSTEM_INSTRUCTION"
        generate(system_prompt=custom_prompt, user_content="user question")

        call_args = mock_client.models.generate_content.call_args
        combined_prompt = call_args[1].get("contents") or call_args[0][0]
        self.assertIn(custom_prompt, combined_prompt)

    # ── Test 9 ────────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_user_content_passed_to_gemini(self, mock_genai):
        """The user_content is included in the contents sent to Gemini."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_mock_response(
            "response. " + REQUIRED_DISCLAIMER
        )
        mock_genai.Client.return_value = mock_client

        user_msg = "UNIQUE_USER_MESSAGE_XYZ"
        generate(system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT, user_content=user_msg)

        call_args = mock_client.models.generate_content.call_args
        combined_prompt = call_args[1].get("contents") or call_args[0][0]
        self.assertIn(user_msg, combined_prompt)

    # ── Test 10 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_return_type_is_always_str(self, mock_genai):
        """generate() always returns str — never a raw SDK object."""
        mock_genai.Client.return_value = _make_mock_client(
            "some text " + REQUIRED_DISCLAIMER
        )

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="test",
        )

        self.assertIsInstance(result, str)
        self.assertNotIsInstance(result, MagicMock)

    # ── Test 11 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_api_key_passed_to_client(self, mock_genai):
        """
        Verify that the API key is read from environment and passed to genai.Client.
        Not hardcoded anywhere in source.
        """
        mock_genai.Client.return_value = _make_mock_client(
            "ok " + REQUIRED_DISCLAIMER
        )

        generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="test",
        )

        # genai.Client must have been called with the env key
        mock_genai.Client.assert_called_once_with(api_key="fake-test-key")

    # ── Test 12 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_unsafe_llm_output_is_returned_unfiltered_for_caller(self, mock_genai):
        """
        generate() does NOT call output_post_filter() itself.
        An unsafe LLM output is returned as-is so the CALLER can filter it.
        """
        unsafe_text = (
            "আপনার ব্যথার জন্য একটি ওষুধ দিনে দুইবার খেতে পারেন। "
            + REQUIRED_DISCLAIMER
        )
        mock_genai.Client.return_value = _make_mock_client(unsafe_text)

        result = generate(
            system_prompt=SHARED_WELLNESS_SYSTEM_PROMPT,
            user_content="আমার কী করা উচিত?",
        )

        self.assertEqual(result, unsafe_text)

        # Confirm output_post_filter() would catch this if called
        from services.safety_filter import output_post_filter, SAFE_FALLBACK_MESSAGE
        filtered = output_post_filter(result)
        self.assertEqual(filtered, SAFE_FALLBACK_MESSAGE)

    # ── Test 13 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_correct_gemini_model_used(self, mock_genai):
        """Gemini model name comes from GEMINI_MODEL constant, not hardcoded inline."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_mock_response(
            "response " + REQUIRED_DISCLAIMER
        )
        mock_genai.Client.return_value = mock_client

        generate(system_prompt="sys", user_content="user")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs.get("model"), GEMINI_MODEL)

    # ── Test 14 ───────────────────────────────────────────────────────────
    def test_shared_wellness_prompt_contains_key_safety_rules(self):
        """SHARED_WELLNESS_SYSTEM_PROMPT contains all required safety constraints."""
        prompt = SHARED_WELLNESS_SYSTEM_PROMPT
        self.assertIn("NOT a doctor", prompt)
        self.assertIn("medicine", prompt.lower())
        self.assertIn("dosage", prompt.lower())
        self.assertIn("এটি চিকিৎসকের বিকল্প নয়", prompt)
        self.assertIn("চিকিৎসকের পরামর্শ", prompt)

    # ── Test 15 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_whitespace_only_response_returns_fallback(self, mock_genai):
        """generate() treats a whitespace-only response as empty and falls back."""
        mock_genai.Client.return_value = _make_mock_client("   \n\t  ")

        result = generate(system_prompt="sys", user_content="user")
        self.assertEqual(result, LLM_FALLBACK_MESSAGE)

    # ── Test 16 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_client_creation_failure_returns_fallback(self, mock_genai):
        """If genai.Client() raises an exception, fallback is returned."""
        mock_genai.Client.side_effect = Exception("Init error")

        result = generate(system_prompt="sys", user_content="user")
        self.assertEqual(result, LLM_FALLBACK_MESSAGE)
        self.assertNotIn("Init error", result)

    # ── Test 17 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_generate_content_called_with_model_constant(self, mock_genai):
        """generate_content is called with model=GEMINI_MODEL keyword arg."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_mock_response(
            "ok " + REQUIRED_DISCLAIMER
        )
        mock_genai.Client.return_value = mock_client

        generate(system_prompt="s", user_content="u")

        mock_client.models.generate_content.assert_called_once()
        kwargs = mock_client.models.generate_content.call_args[1]
        self.assertIn("model", kwargs)
        self.assertIn("contents", kwargs)

    # ── Test 18 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_prompt_separator_between_system_and_user(self, mock_genai):
        """System prompt and user content are separated by '---' in the combined prompt."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_mock_response(
            "ok " + REQUIRED_DISCLAIMER
        )
        mock_genai.Client.return_value = mock_client

        generate(system_prompt="SYSPROMPT", user_content="USERCONTENT")

        kwargs = mock_client.models.generate_content.call_args[1]
        combined = kwargs["contents"]
        self.assertIn("SYSPROMPT", combined)
        self.assertIn("USERCONTENT", combined)
        self.assertIn("---", combined)

    # ── Test 19 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_no_old_sdk_configure_called(self, mock_genai):
        """New SDK uses genai.Client(), not genai.configure()."""
        mock_genai.Client.return_value = _make_mock_client(
            "ok " + REQUIRED_DISCLAIMER
        )
        generate(system_prompt="s", user_content="u")

        # configure() must NOT be called in the new SDK
        mock_genai.configure.assert_not_called()

    # ── Test 20 ───────────────────────────────────────────────────────────
    @patch("services.llm_service.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"})
    def test_no_generative_model_called(self, mock_genai):
        """New SDK uses client.models.generate_content(), not genai.GenerativeModel()."""
        mock_genai.Client.return_value = _make_mock_client(
            "ok " + REQUIRED_DISCLAIMER
        )
        generate(system_prompt="s", user_content="u")

        # GenerativeModel must NOT be called in the new SDK
        mock_genai.GenerativeModel.assert_not_called()


# ==================================================
# UNIT TESTS FOR PRIVATE HELPERS
# ==================================================

class TestHelpers(unittest.TestCase):

    def test_ensure_disclaimer_appends_when_missing(self):
        text = "পর্যাপ্ত ঘুম দরকার।"
        result = _ensure_disclaimer(text)
        self.assertIn(REQUIRED_DISCLAIMER, result)
        self.assertIn(text, result)

    def test_ensure_disclaimer_no_duplicate(self):
        text = "ভালো পরামর্শ। " + REQUIRED_DISCLAIMER
        result = _ensure_disclaimer(text)
        self.assertEqual(result.count(REQUIRED_DISCLAIMER), 1)

    def test_extract_text_with_valid_response(self):
        mock_resp = _make_mock_response("some valid text")
        self.assertEqual(_extract_text(mock_resp), "some valid text")

    def test_extract_text_with_empty_string(self):
        mock_resp = _make_mock_response("")
        self.assertIsNone(_extract_text(mock_resp))

    def test_extract_text_with_none(self):
        mock_resp = _make_mock_response(None)
        self.assertIsNone(_extract_text(mock_resp))


# ==================================================
# RUNNER
# ==================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SheCare BD — LLM Service Test Suite  (google-genai SDK)")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestLlmService))
    suite.addTests(loader.loadTestsFromTestCase(TestHelpers))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 60)
    if result.wasSuccessful():
        total = result.testsRun
        print(f"🎉 All {total} tests passed successfully!")
    else:
        print(f"❌ {len(result.failures)} failure(s), {len(result.errors)} error(s).")
    print("=" * 60)
