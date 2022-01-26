# Copyright 2021-present Kensho Technologies, LLC.
import json
import math
import multiprocessing
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
    build_ctcdecoder,
)
from ..language_model import LanguageModel
from .helpers import TempfileTestCase


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

    def test_build_ctcdecoder(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

    def test_decode_batch(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH, TEST_UNIGRAMS)
        with multiprocessing.get_context("fork").Pool() as pool:
            text_list = decoder.decode_batch(pool, [TEST_LOGITS] * 5)
        expected_text_list = ["bugs bunny"] * 5
        self.assertListEqual(expected_text_list, text_list)

    def test_logit_shape_mismatch(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS)
        wrong_shape_logits = np.hstack([TEST_LOGITS] * 2)
        with self.assertRaises(ValueError):
            _ = decoder.decode(wrong_shape_logits)
        with multiprocessing.Pool() as pool:
            with self.assertRaises(ValueError):
                _ = decoder.decode_batch(pool, [wrong_shape_logits] * 5)

    def test_decode_beams_batch(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH, TEST_UNIGRAMS)
        with multiprocessing.get_context("fork").Pool() as pool:
            text_list = decoder.decode_beams_batch(pool, [TEST_LOGITS] * 5)
        expected_text_list = [
            [
                (
                    "bugs bunny",
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
            [
                (
                    "bugs bunny",
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
            [
                (
                    "bugs bunny",
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
            [
                (
                    "bugs bunny",
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
            [
                (
                    "bugs bunny",
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
        ]
        self.assertListEqual(expected_text_list, text_list)


class TestSerialization(TempfileTestCase):
    def _count_num_language_models(self):
        return sum(
            [1 for model in BeamSearchDecoderCTC.model_container.values() if model is not None]
        )

    def test_parse_directory(self):
        good_filenames = [
            ("alphabet.json", "language_model"),
            ("alphabet.json",),
            ("README.md", "alphabet.json", "language_model"),  # additional file in dir
        ]
        bad_filenames = [
            ("language_model"),  # missing alphabet
            ("alphabet.wrong-ext", "language_model"),
        ]

        for filenames in good_filenames:
            self.clear_dir()
            for fn in filenames:
                with open(os.path.join(self.temp_dir, fn), "w") as fi:
                    fi.write("meaningless data")
            BeamSearchDecoderCTC.parse_directory_contents(self.temp_dir)  # should not error out

        for filenames in bad_filenames:
            self.clear_dir()
            for fn in filenames:
                with open(os.path.join(self.temp_dir, fn), "w") as fi:
                    fi.write("meaningless data")
            with self.assertRaises(ValueError):
                LanguageModel.parse_directory_contents(self.temp_dir)

    def test_serialization(self):
        self.clear_dir()
        decoder = build_ctcdecoder(LIBRI_LABELS)
        text = decoder.decode(LIBRI_LOGITS)
        old_num_models = self._count_num_language_models()

        # saving shouldn't alter state of model_container
        decoder.save_to_dir(self.temp_dir)
        self.assertEqual(self._count_num_language_models(), old_num_models)

        new_decoder = BeamSearchDecoderCTC.load_from_dir(self.temp_dir)
        new_text = new_decoder.decode(LIBRI_LOGITS)
        self.assertEqual(text, new_text)

        # this decoder has no LM so we should not increment the number of models
        self.assertEqual(old_num_models, self._count_num_language_models())

        self.clear_dir()
        BeamSearchDecoderCTC.clear_class_models()

        # repeat with a decoder with a language model
        alphabet = Alphabet.build_alphabet(SAMPLE_LABELS)
        language_model = LanguageModel(TEST_KENLM_MODEL, alpha=1.0)
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

        old_num_models = self._count_num_language_models()
        decoder.save_to_dir(self.temp_dir)
        # here too, saving should not alter number of lang models
        self.assertEqual(self._count_num_language_models(), old_num_models)

        new_decoder = BeamSearchDecoderCTC.load_from_dir(self.temp_dir)
        new_text = new_decoder.decode(TEST_LOGITS)
        self.assertEqual(text, new_text)

        # this decoder has a language model so we expect one more key
        self.assertEqual(old_num_models + 1, self._count_num_language_models())

    def test_load_from_hub_offline(self):
        from huggingface_hub.snapshot_download import REPO_ID_SEPARATOR

        # create language model and decode text
        alphabet = Alphabet.build_alphabet(SAMPLE_LABELS)
        language_model = LanguageModel(TEST_KENLM_MODEL, alpha=1.0)
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

        # We are now pretending to have downloaded a HF repository
        # called `kesho/dummy_test` into the cache. The format of
        # cached HF repositories is <flattened-hub-name>.<branch>.<sha256> which
        # is created under the hood in `.load_from_hf_hub`. To mock a cached
        # download we have to do it manually here.
        dummy_hub_name = "kensho/dummy_test"
        dummy_cached_subdir = (
            dummy_hub_name.replace("/", REPO_ID_SEPARATOR) + ".main.123456aoeusnth"
        )
        dummy_cached_dir = os.path.join(self.temp_dir, dummy_cached_subdir)
        os.makedirs(dummy_cached_dir)

        # save decoder
        decoder.save_to_dir(os.path.join(self.temp_dir, dummy_cached_dir))

        # load from cache in offline mode
        new_decoder = BeamSearchDecoderCTC.load_from_hf_hub(
            dummy_hub_name, cache_dir=self.temp_dir, local_files_only=True
        )

        new_text = new_decoder.decode(TEST_LOGITS)
        self.assertEqual(text, new_text)
