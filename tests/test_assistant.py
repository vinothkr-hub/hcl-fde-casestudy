import json
import unittest
from pathlib import Path

from assistant import ABSTAIN_MESSAGE, answer_question, load_documents


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "starter_kit" / "data" / "example_questions.json"


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents()
        with QUESTIONS_PATH.open(encoding="utf-8") as file:
            cls.examples = json.load(file)["example_questions"]

    def test_example_questions_retrieve_expected_documents(self) -> None:
        for example in self.examples:
            with self.subTest(question_id=example["question_id"]):
                answer = answer_question(example["question"], self.documents)
                self.assertEqual(example["expected_doc_id"], answer.doc_id)
                if answer.doc_id:
                    self.assertTrue(answer.text)

    def test_unsupported_question_abstains(self) -> None:
        answer = answer_question(
            "Can I connect Solstice to Salesforce?", self.documents
        )
        self.assertIsNone(answer.doc_id)
        self.assertEqual(ABSTAIN_MESSAGE, answer.text)

    def test_answer_contains_specific_documented_detail(self) -> None:
        answer = answer_question(
            "What's the maximum number of rows I can export to CSV?",
            self.documents,
        )
        self.assertIn("10,000 rows", answer.text)

    def test_password_reset_question_returns_instructions_not_expiry(self) -> None:
        answer = answer_question("How do I reset my password?", self.documents)
        self.assertIn("Forgot password?", answer.text)
        self.assertNotIn("expire", answer.text.lower())

    def test_question_without_content_words_abstains(self) -> None:
        answer = answer_question("How can I do it?", self.documents)
        self.assertIsNone(answer.doc_id)
        self.assertEqual(ABSTAIN_MESSAGE, answer.text)


if __name__ == "__main__":
    unittest.main()
