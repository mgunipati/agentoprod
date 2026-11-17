"""Checks on app/guardrails/finance-guardrail.json that Bedrock would otherwise reject at deploy time."""

import json
import re
import unittest
from pathlib import Path

DEFINITION = json.loads(
    (Path(__file__).resolve().parents[1] / "app" / "guardrails" / "finance-guardrail.json").read_text()
)


class GuardrailDefinitionTests(unittest.TestCase):
    def test_blocked_messages_fit_bedrock_limits(self) -> None:
        for key in ("blockedInputMessaging", "blockedOutputsMessaging"):
            self.assertTrue(0 < len(DEFINITION[key]) <= 500, key)

    def test_prompt_attack_checks_input_only(self) -> None:
        # Bedrock requires PROMPT_ATTACK outputStrength NONE: it applies to user input.
        filters = {f["type"]: f for f in DEFINITION["contentPolicyConfig"]["filtersConfig"]}
        self.assertEqual(filters["PROMPT_ATTACK"]["inputStrength"], "HIGH")
        self.assertEqual(filters["PROMPT_ATTACK"]["outputStrength"], "NONE")
        self.assertLessEqual({"HATE", "SEXUAL", "VIOLENCE", "INSULTS", "MISCONDUCT"}, filters.keys())

    def test_denied_topics_fit_bedrock_limits(self) -> None:
        topics = DEFINITION["topicPolicyConfig"]["topicsConfig"]
        self.assertIn("PersonalInvestmentAdvice", [t["name"] for t in topics])
        for topic in topics:
            self.assertEqual(topic["type"], "DENY")
            self.assertLessEqual(len(topic["definition"]), 200)
            self.assertLessEqual(len(topic["examples"]), 5)
            self.assertTrue(all(len(example) <= 100 for example in topic["examples"]))

    def test_payment_details_are_blocked_and_contacts_masked(self) -> None:
        actions = {e["type"]: e["action"] for e in DEFINITION["sensitiveInformationPolicyConfig"]["piiEntitiesConfig"]}
        self.assertEqual(actions["CREDIT_DEBIT_CARD_NUMBER"], "BLOCK")
        self.assertEqual(actions["EMAIL"], "ANONYMIZE")
        self.assertEqual(actions["PHONE"], "ANONYMIZE")

    def test_pan_regex_matches_a_pan_and_not_tickers_or_prices(self) -> None:
        pattern = re.compile(DEFINITION["sensitiveInformationPolicyConfig"]["regexesConfig"][0]["pattern"])
        self.assertTrue(pattern.search("My PAN is ABCDE1234F."))
        for text in ("RELIANCE.NS closed at 2,950.10 INR", "TCS.NS", "AAPL 189.5 USD"):
            self.assertIsNone(pattern.search(text), text)


if __name__ == "__main__":
    unittest.main()
