# Copyright 2021-present Kensho Technologies, LLC.
import json
import unittest

from ..alphabet import Alphabet, _normalize_bpe_alphabet, _normalize_regular_alphabet


def _approx_beams(beams, precis=5):
    """Return beams with scores rounded."""
    return [tuple(list(b[:-1]) + [round(b[-1], precis)]) for b in beams]


KNOWN_MAPPINGS = [
    (
        [" ", "a", "b"],
        [" ", "a", "b", ""],
        False,
    ),  # nemo
    (
        ["<pad>", "<s>", "</s>", "<unk>", "|", "A", "B"],
        ["", "<s>", "</s>", "⁇", " ", "A", "B"],
        False,
    ),  # huggingface
    (
        ["<unk>", "▁", "##a", "##b", "a", "b"],
        ["▁⁇▁", "▁", "a", "b", "▁a", "▁b", ""],
        True,
    ),  # nemo-bpe
]

TEST_MAPPINGS = [
    (
        [" ", "a", "b", ""],
        [" ", "a", "b", ""],
    ),  # make sure ctc blank doesn"t get added if exists
]

BPE_TEST_MAPPINGS = [
    (
        ["▁⁇▁", "▁", "a", "b", "▁a", "▁b"],
        ["▁⁇▁", "▁", "a", "b", "▁a", "▁b", ""],
    ),  # bpe in correct form
    (
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "##a", "##b", "a", "b"],
        ["", "▁⁇▁", "[CLS]", "[SEP]", "[MASK]", "a", "b", "▁a", "▁b"],
    ),  # other special tokens
]


class TestModelHelpers(unittest.TestCase):
    def test_normalize_alphabet(self):
        for labels, expected_labels in TEST_MAPPINGS:
            norm_labels = _normalize_regular_alphabet(labels)
            self.assertListEqual(norm_labels, expected_labels)

    def test_normalize_alphabet_bpe(self):
        for labels, expected_labels in BPE_TEST_MAPPINGS:
            norm_labels = _normalize_bpe_alphabet(labels)
            self.assertListEqual(norm_labels, expected_labels)

    def test_alphabets(self):
        for labels, expected_labels, expected_is_bpe in KNOWN_MAPPINGS:
            alphabet = Alphabet.build_alphabet(labels)
            self.assertListEqual(alphabet.labels, expected_labels)
            self.assertEqual(alphabet.is_bpe, expected_is_bpe)

    def test_asserts(self):
        # duplication
        label_list = ["a", "a", "b", "c"]
        with self.assertRaises(ValueError):
            Alphabet.build_alphabet(label_list)
        # bpe with space
        label_list = ["▁a", " "]
        with self.assertRaises(ValueError):
            Alphabet.build_alphabet(label_list)


class TestAlphabetSerialization(unittest.TestCase):
    def test_dumps_and_loads(self):
        for labels, _, _ in KNOWN_MAPPINGS:
            alphabet = Alphabet.build_alphabet(labels)
            recovered_alphabet = Alphabet.loads(alphabet.dumps())
            self.assertEqual(alphabet.labels, recovered_alphabet.labels)
            self.assertEqual(alphabet.is_bpe, recovered_alphabet.is_bpe)

        bad_param_choices = [
            {"labels": ["a", "b", "c"]},  # missing "is_bpe"
            {"labels": ["a", "b", "c"], "is_bpe": True, "extra": 1},  # extra key
        ]
        for bad_params in bad_param_choices:
            with self.assertRaises(ValueError):
                Alphabet.loads(json.dumps(bad_params))
