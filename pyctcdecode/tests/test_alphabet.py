# Copyright 2021-present Kensho Technologies, LLC.
import unittest

from ..alphabet import Alphabet, _normalize_alphabet, _normalize_bpe_alphabet


def _approx_beams(beams, precis=5):
    """Return beams with scores rounded."""
    return [tuple(list(b[:-1]) + [round(b[-1], precis)]) for b in beams]


class TestModelHelpers(unittest.TestCase):
    def test_normalize_alphabet(self):
        alphabet_list = [" ", "a", "b", ""]
        norm_alphabet = _normalize_alphabet(alphabet_list)
        expected_alphabet = [" ", "a", "b", ""]
        self.assertListEqual(norm_alphabet, expected_alphabet)

        # missing blank char
        alphabet_list = [" ", "a", "b"]
        norm_alphabet = _normalize_alphabet(alphabet_list)
        expected_alphabet = [" ", "a", "b", ""]
        self.assertListEqual(norm_alphabet, expected_alphabet)

        # invalid input
        alphabet_list = [" ", "a", "bb"]
        with self.assertRaises(ValueError):
            _normalize_alphabet(alphabet_list)

    def test_normalize_alphabet_bpe(self):
        # style ▁ input
        alphabet_list = ["▁⁇▁", "▁B", "ugs", "▁", "▁bunny", ""]
        norm_alphabet = _normalize_bpe_alphabet(alphabet_list)
        expected_alphabet = ["▁⁇▁", "▁B", "ugs", "▁", "▁bunny", ""]
        self.assertListEqual(norm_alphabet, expected_alphabet)

        # style ▁ with missing blank char
        alphabet_list = ["▁⁇▁", "▁B", "ugs"]
        norm_alphabet = _normalize_bpe_alphabet(alphabet_list)
        expected_alphabet = ["▁⁇▁", "▁B", "ugs", ""]
        self.assertListEqual(norm_alphabet, expected_alphabet)

        # other unk style
        alphabet_list = ["<unk>", "▁B", "ugs"]
        norm_alphabet = _normalize_bpe_alphabet(alphabet_list)
        expected_alphabet = ["▁⁇▁", "▁B", "ugs", ""]
        self.assertListEqual(norm_alphabet, expected_alphabet)

        # style ## input
        alphabet_list = ["B", "##ugs", ""]
        norm_alphabet = _normalize_bpe_alphabet(alphabet_list)
        expected_alphabet = ["▁⁇▁", "▁B", "ugs", ""]
        self.assertListEqual(norm_alphabet, expected_alphabet)

        # style ## with single ▁ char
        alphabet_list = ["B", "##ugs", "▁", ""]
        norm_alphabet = _normalize_bpe_alphabet(alphabet_list)
        expected_alphabet = ["▁⁇▁", "▁B", "ugs", "▁", ""]
        self.assertListEqual(norm_alphabet, expected_alphabet)

        # invalid input
        alphabet_list = ["B", "##ugs", "▁bunny", ""]
        with self.assertRaises(ValueError):
            _normalize_bpe_alphabet(alphabet_list)

    def test_alphabets(self):
        label_list = [" ", "a", "b", ""]
        alphabet = Alphabet.build_alphabet(label_list)
        expected_labels = [" ", "a", "b", ""]
        self.assertFalse(alphabet.is_bpe)
        self.assertListEqual(alphabet.labels, expected_labels)

        label_list = ["B", "##ugs", ""]
        alphabet_bpe = Alphabet.build_bpe_alphabet(label_list)
        expected_labels = ["▁⁇▁", "▁B", "ugs", ""]
        self.assertTrue(alphabet_bpe.is_bpe)
        self.assertListEqual(alphabet_bpe.labels, expected_labels)
