"""Regression checks for meaning-preserving cleaning and identifier handling."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cleaning import clean_text,normalize_accession,normalize_identifier,tokenize_tfidf

class CleaningTests(unittest.TestCase):
    def test_identifier_padding_and_invalid(self):
        self.assertEqual(normalize_identifier("1234",6),"001234")
        self.assertEqual(normalize_identifier("0001234.0"),"1234")
        for value in [None,"<NA>","nan","0","12x","1234567"]:
            self.assertIsNone(normalize_identifier(value,6))

    def test_accession(self):
        self.assertEqual(normalize_accession("0001193125-20-123456"),"000119312520123456")
        self.assertIsNone(normalize_accession("123"))

    def test_preserve_negation_money_and_bare_table_numbers(self):
        text,info=clean_text("Table of Contents\n\nWe do not plan to spend $10 million in 2027.\n\n123\n\nPage 7",0)
        self.assertIn("not",text)
        self.assertIn("$10 million in 2027",text)
        self.assertIn("123",text)
        self.assertNotIn("Table of Contents",text)
        self.assertEqual(info["removed_noise_lines"],2)

    def test_actual_later_heading_is_trimmed(self):
        body="We plan to expand equipment. "*120
        text,info=clean_text(body+"\n\nItem 8. Financial Statements\nOther section",0)
        self.assertNotIn("Other section",text)
        self.assertGreater(info["boundary_trim_characters"],0)

    def test_reference_is_not_heading(self):
        raw="See Item 8. Financial Statements for details. "*100
        text,info=clean_text(raw,0)
        self.assertEqual(info["boundary_trim_characters"],0)
        self.assertIn("See Item 8",text)

    def test_early_heading_is_flagged_not_removed(self):
        text,info=clean_text("Item 8. Financial Statements\nShort index",0)
        self.assertIn("Short index",text)
        self.assertIn("early_boundary",info["quality_flags"])

    def test_long_repeat_dedup_keeps_short_numbers(self):
        block="A specific capital investment project description. "*15
        text,info=clean_text(block+"\n\n"+block+"\n\n100\n\n100",0)
        self.assertEqual(info["removed_duplicate_long_blocks"],1)
        self.assertEqual(text.count("100"),2)

    def test_tfidf_view_preserves_negations(self):
        tokens=tokenize_tfidf("We will not increase spending without financing in 2027.")
        self.assertIn("not",tokens)
        self.assertIn("without",tokens)
        self.assertIn("will",tokens)
        self.assertIn("2027",tokens)
        self.assertNotIn("we",tokens)

if __name__=="__main__":
    unittest.main()
