# Copyright 2021-present Kensho Technologies, LLC.
import json
import math
import os
import unittest

import kenlm  # type: ignore
import numpy as np

from ..alphabet import Alphabet
from ..decoder import (
    BeamSearchDecoderCTC,
    _merge_beams,
    _normalize_whitespace,
    _prune_history,
    _sort_and_trim_beams,
    _sum_log_scores,
)
from ..language_model import LanguageModel


def _random_matrix(rows: int, cols: int) -> np.ndarray:
    return np.random.normal(size=(rows, cols))


def _random_logits(rows: int, cols: int) -> np.ndarray:
    """Sample random logit matrix of given dimension."""
    xs = np.exp(_random_matrix(rows, cols))
    ps = (xs.T / np.sum(xs, axis=1)).T
    logits = np.log(ps)
    return logits


def _random_libri_logits(N: int) -> np.ndarray:
    return _random_logits(N, len(LIBRI_LABELS) + 1)


def _approx_beams(beams, precis=5):
    """Return beams with scores rounded."""
    return [tuple(list(b[:-1]) + [round(b[-1], precis)]) for b in beams]


def _approx_lm_beams(beams, precis=5):
    """Return beams with scores rounded."""
    simple_beams = []
    for text, _, frames, s1, s2 in beams:
        simple_beams.append((text, frames, round(s1, precis), round(s2, precis)))
    return simple_beams


def _greedy_decode(logits, alphabet):
    label_dict = {n: c for n, c in enumerate(alphabet.labels)}
    prev_c = None
    out = []
    for n in logits.argmax(axis=1):
        c = label_dict[n]
        if c != prev_c:
            out.append(c)
    return "".join(out)


class TestDecoderHelpers(unittest.TestCase):
    def test_normalize_whitespace(self):
        # empty input
        out = _normalize_whitespace("")
        self.assertEqual(out, "")

        out = _normalize_whitespace(" This  is super. ")
        self.assertEqual(out, "This is super.")

    def test_sum_log_scores(self):
        out = _sum_log_scores(0, 0)
        expected_out = math.log(math.exp(0) + math.exp(0))
        self.assertEqual(out, expected_out)

        out = _sum_log_scores(1 - math.log(2), 1 - math.log(2))
        self.assertEqual(out, 1.0)

    def test_sort_and_trim_beams(self):
        beams = _sort_and_trim_beams([(-3,), (-9,), (-5,)], 2)
        expected_beams = [(-3,), (-5,)]
        self.assertListEqual(beams, expected_beams)

    def test_merge_beams(self):
        beams = [
            ("Batman and", "", "Robi", "i", [], (-1, -1), -1),
            ("Batman and", "Robin", "", "", [], (-1, -1), -1),
            ("Batman and", "", "Robi", "", [], (-1, -1), -1),
            ("Batman and", "", "Robi", "", [], (-1, -1), -1),
            ("Batman &", "", "Robi", "", [], (-1, -1), -1),
        ]
        merged_beams = _merge_beams(beams)
        expected_merged_beams = [
            ("Batman and", "", "Robi", "i", [], (-1, -1), -1),
            ("Batman and", "Robin", "", "", [], (-1, -1), -1),
            ("Batman and", "", "Robi", "", [], (-1, -1), math.log(2 * math.exp(-1))),
            ("Batman &", "", "Robi", "", [], (-1, -1), -1),
        ]
        self.assertListEqual(_approx_beams(merged_beams), _approx_beams(expected_merged_beams))

    def test_prune_history(self):
        beams = [
            ("A Gandalf owns", "", "potatoes", "s", [], (-1, -1), -1, -1),
            ("B Gandalf owns", "", "potatoes", "", [], (-1, -1), -1, -1),
            ("C Gandalf owns", "", "potatoes", "s", [], (-1, -1), -1, -1),
            ("D Gandalf sells", "", "yeast", "", [], (-1, -1), -1, -1),
            ("E Gandalf owns", "", "yeast", "", [], (-1, -1), -1, -1),
        ]
        pruned_beams = _prune_history(beams, 3)
        expected_pruned_beams = [
            ("A Gandalf owns", "", "potatoes", "s", [], (-1, -1), -1),
            ("B Gandalf owns", "", "potatoes", "", [], (-1, -1), -1),
            ("D Gandalf sells", "", "yeast", "", [], (-1, -1), -1),
            ("E Gandalf owns", "", "yeast", "", [], (-1, -1), -1),
        ]
        self.assertListEqual(_approx_beams(pruned_beams), _approx_beams(expected_pruned_beams))


CUR_PATH = os.path.dirname(os.path.abspath(__file__))

# libri files
with open(os.path.join(CUR_PATH, "sample_data", "libri_logits.json")) as f:
    LIBRI_LOGITS = np.array(json.load(f))
LIBRI_LABELS = [
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
]

# basic 2-gram kenlm model trained with 'bugs bunny'
KENLM_MODEL_PATH = os.path.join(CUR_PATH, "sample_data", "bugs_bunny_kenlm.arpa")
TEST_KENLM_MODEL = kenlm.Model(KENLM_MODEL_PATH)

SAMPLE_LABELS = [" ", "b", "g", "n", "s", "u", "y", ""]
SAMPLE_VOCAB = {c: n for n, c in enumerate(SAMPLE_LABELS)}

BUGS_PROBS = np.zeros((4, len(SAMPLE_VOCAB)))
BUGS_PROBS[0][SAMPLE_VOCAB.get("b")] = 1
BUGS_PROBS[1][SAMPLE_VOCAB.get("u")] = 1
BUGS_PROBS[2][SAMPLE_VOCAB.get("g")] = 1
BUGS_PROBS[3][SAMPLE_VOCAB.get("s")] = 1

BUNNY_PROBS = np.zeros((6, len(SAMPLE_VOCAB)))
BUNNY_PROBS[0][SAMPLE_VOCAB.get("b")] = 1
BUNNY_PROBS[1][SAMPLE_VOCAB.get("u")] = 1
BUNNY_PROBS[2][SAMPLE_VOCAB.get("n")] = 1
BUNNY_PROBS[3][SAMPLE_VOCAB.get("")] = 1
BUNNY_PROBS[4][SAMPLE_VOCAB.get("n")] = 1
BUNNY_PROBS[5][SAMPLE_VOCAB.get("y")] = 1

BLANK_PROBS = np.zeros((1, len(SAMPLE_VOCAB)))
BLANK_PROBS[0][SAMPLE_VOCAB.get("")] = 1
SPACE_PROBS = np.zeros((1, len(SAMPLE_VOCAB)))
SPACE_PROBS[0][SAMPLE_VOCAB.get(" ")] = 1

# make mixed version that can get fixed with LM
TEST_PROBS = np.vstack(
    [
        np.vstack([BUGS_PROBS, BLANK_PROBS, BLANK_PROBS]) * 0.49 + BUNNY_PROBS * 0.51,
        SPACE_PROBS,
        BUNNY_PROBS,
    ]
)
# convert to log probs without hitting overflow
TEST_LOGITS = np.log(np.clip(TEST_PROBS, 1e-15, 1))

TEST_UNIGRAMS = ["bugs", "bunny"]


class TestDecoder(unittest.TestCase):
    def test_decoder(self):
        alphabet = Alphabet.build_alphabet(SAMPLE_LABELS)

        # test empty
        decoder = BeamSearchDecoderCTC(alphabet)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bunny bunny")

        # with LM but alpha = 0
        language_model = LanguageModel(TEST_KENLM_MODEL, alpha=0.0)
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bunny bunny")

        # alpha = 1
        language_model = LanguageModel(TEST_KENLM_MODEL, alpha=1.0)
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

        # add empty unigrams
        language_model = LanguageModel(TEST_KENLM_MODEL, [], alpha=1.0)
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

        # add restricted unigrams but unk weight 0
        unigrams = ["bunny"]
        language_model = LanguageModel(TEST_KENLM_MODEL, unigrams, alpha=1.0, unk_score_offset=0.0)
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

        # add unk weight
        unigrams = ["bunny"]
        language_model = LanguageModel(
            TEST_KENLM_MODEL, unigrams, alpha=1.0, unk_score_offset=-10.0
        )
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bunny bunny")

        # verify class variables and cleanup
        n_models = len(BeamSearchDecoderCTC.model_container)
        self.assertGreaterEqual(n_models, 2)
        # delete a specific model from container
        decoder.cleanup()
        self.assertLess(len(BeamSearchDecoderCTC.model_container), n_models)
        # delete all models
        BeamSearchDecoderCTC.clear_class_models()
        self.assertEqual(len(BeamSearchDecoderCTC.model_container), 0)
