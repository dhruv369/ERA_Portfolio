import json
import shutil
import unittest
from pathlib import Path

from pipeline import MAX_INPUT_TOKENS, MinHashLSH, process_rows, word_tokens


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.output = Path(__file__).parent / ".test-output" / self._testMethodName
        self.output.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.output, ignore_errors=True)
        try:
            self.output.parent.rmdir()
        except OSError:
            pass

    def test_whitespace_token_definition(self):
        self.assertEqual(word_tokens("  one\t two\nthree  "), ["one", "two", "three"])

    def test_pipeline_metrics_and_cleaning(self):
        rows = [
            {
                "idx": 1,
                "language": "hi",
                "src": "Hello there friend",
                "tgt": "नमस्ते दुनिया मित्र",
            },
            {
                "idx": 2,
                "language": "hi",
                "src": "Hello there friend",
                "tgt": "नमस्ते दुनिया मित्र",
            },
            {
                "idx": 3,
                "language": "hi",
                "src": "Good morning [USER]",
                "tgt": "नमस्ते <|endoftext|>",
            },
            {
                "idx": 4,
                "language": "hi",
                "src": "Contact jane@example.com or +1 555 123 4567",
                "tgt": "संपर्क करें",
            },
            {
                "idx": 5,
                "language": "hi",
                "src": "Subscribe to continue reading",
                "tgt": "अधिक पढ़ें",
            },
            {"idx": 6, "language": "hi", "src": "OK", "tgt": "ठीक"},
            {
                "idx": 7,
                "language": "hi",
                "src": "Cafe\u0301 guide today",
                "tgt": "नमस्ते\u200d दुनिया आज",
            },
        ]
        stats = process_rows(rows, self.output, target_tokens=1000)
        cleaned = [
            json.loads(line)
            for line in (self.output / "cleaned.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(stats["slice"]["input_documents"], 7)
        self.assertEqual(stats["slice"]["output_documents"], 4)
        self.assertEqual(stats["stages"]["boilerplate"]["dropped_documents"], 1)
        self.assertEqual(stats["stages"]["quality"]["dropped_documents"], 1)
        self.assertEqual(stats["stages"]["minhash_lsh"]["dropped_documents"], 1)
        self.assertAlmostEqual(stats["deduplication"]["duplicate_rate"], 0.2)
        self.assertEqual(stats["special_tag_occurrences_removed"], 2)
        self.assertEqual(stats["parameters"]["pii"]["counts"]["email"], 1)
        self.assertEqual(stats["parameters"]["pii"]["counts"]["phone"], 1)
        self.assertTrue(any("\u200d" in row["tgt"] for row in cleaned))
        self.assertFalse(any("\u0301" in row["src"] for row in cleaned))
        pii_row = next(row for row in cleaned if "[EMAIL]" in row["src"])
        self.assertNotIn("jane@example.com", pii_row["src"])
        self.assertEqual(
            stats["stages"]["decontamination"]["status"],
            "not_run_no_holdout_file",
        )

    def test_target_budget_bounds_input_not_clean_output(self):
        rows = [
            {"idx": 1, "language": "hi", "src": "one two", "tgt": "तीन चार"},
            {"idx": 2, "language": "hi", "src": "five six", "tgt": "सात आठ"},
        ]
        stats = process_rows(rows, self.output, target_tokens=4)
        self.assertEqual(stats["slice"]["input_tokens"], 4)
        self.assertEqual(stats["slice"]["input_documents"], 1)

    def test_target_budget_cannot_exceed_ten_million(self):
        with self.assertRaisesRegex(ValueError, "10,000,000"):
            process_rows([], self.output, target_tokens=MAX_INPUT_TOKENS + 1)

    def test_stage_can_be_disabled(self):
        rows = [
            {
                "idx": 1,
                "language": "hi",
                "src": "Subscribe to continue reading",
                "tgt": "अधिक पढ़ें",
            }
        ]
        stats = process_rows(
            rows, self.output, disabled={"boilerplate", "minhash_lsh"}
        )
        result = json.loads(
            (self.output / "cleaned.jsonl").read_text(encoding="utf-8")
        )
        self.assertEqual(result["idx"], 1)
        self.assertEqual(stats["stages"]["boilerplate"]["status"], "not_applied")
        self.assertEqual(stats["stages"]["boilerplate"]["output_documents"], 1)
        self.assertEqual(stats["stages"]["minhash_lsh"]["status"], "not_applied")

    def test_minhash_threshold_validation(self):
        with self.assertRaises(ValueError):
            MinHashLSH(self.output / "index.sqlite3", threshold=0)

    def test_decontamination_runs_only_with_holdout(self):
        phrase = "A reserved benchmark sentence appears only in this test fixture."
        holdout = self.output / "holdout.txt"
        holdout.write_text(phrase + "\n", encoding="utf-8")
        rows = [
            {
                "idx": 1,
                "language": "hi",
                "src": phrase,
                "tgt": "यह एक पर्याप्त लंबा परीक्षण वाक्य है।",
            }
        ]
        stats = process_rows(
            rows, self.output, target_tokens=100, holdout_file=holdout
        )
        self.assertEqual(
            stats["stages"]["decontamination"]["status"], "applied"
        )
        self.assertEqual(stats["stages"]["decontamination"]["dropped_documents"], 1)
        self.assertEqual(stats["stages"]["decontamination"]["findings"], 1)
        self.assertEqual(stats["slice"]["output_documents"], 0)


if __name__ == "__main__":
    unittest.main()
