import unittest
from src.preprocess import (
    remove_stopwords, convert_unicode, is_valid_vietnamese_word,
    standardize_word_typing, standardize_sentence_typing, normalize_acronyms,
    remove_unnecessary_characters, preprocess_fn
)


class TestPreprocessFunctions(unittest.TestCase):
    def test_remove_stopwords_removes_stopwords(self):
        self.assertEqual(remove_stopwords("không đầy test"), "test")

    def test_remove_stopwords_keeps_non_stopwords(self):
        self.assertEqual(remove_stopwords("sợ ghê"), "sợ ghê")

    def test_convert_unicode_converts_correctly(self):
        self.assertEqual(convert_unicode("à"), "à")

    def test_convert_unicode_keeps_non_vietnamese(self):
        self.assertEqual(convert_unicode("hello"), "hello")

    def test_is_valid_vietnamese_word_valid_word(self):
        self.assertTrue(is_valid_vietnamese_word("hòa"))

    def test_is_valid_vietnamese_word_invalid_word(self):
        self.assertFalse(is_valid_vietnamese_word("the sea is blue"))

    def test_standardize_word_typing_standardizes_correctly(self):
        self.assertEqual(standardize_word_typing("hoà"), "hòa")

    def test_standardize_word_typing_keeps_non_vietnamese(self):
        self.assertEqual(standardize_word_typing("hello"), "hello")

    def test_standardize_sentence_typing_standardizes_sentence(self):
        self.assertEqual(standardize_sentence_typing("hoà bình"), "hòa bình")

    def test_standardize_sentence_typing_keeps_non_vietnamese_sentence(self):
        self.assertEqual(standardize_sentence_typing("hello world"), "hello world")

    def test_normalize_acronyms_normalizes_correctly(self):
        self.assertEqual(normalize_acronyms("ctrai"), "con trai")

    def test_normalize_acronyms_keeps_non_acronyms(self):
        self.assertEqual(normalize_acronyms("hello"), "hello")

    def test_remove_unnecessary_characters_removes_correctly(self):
        self.assertEqual(remove_unnecessary_characters("hello!!!"), "hello")

    def test_remove_unnecessary_characters_keeps_valid_characters(self):
        self.assertEqual(remove_unnecessary_characters("hello world"), "hello world")

    def test_preprocess_fn_processes_text_correctly(self):
        self.assertEqual(preprocess_fn("hoà bình bme"), "hòa bình bố_mẹ")

    def test_teencode_conversion(self):
        self.assertEqual(normalize_acronyms("ctrai"), "con trai")
        self.assertEqual(normalize_acronyms("khôg"), "không")

    def test_preprocess_fn_with_teencode_and_stopwords(self):
        self.assertEqual(preprocess_fn("ctrai không để"), "trai")


if __name__ == '__main__':
    unittest.main()
