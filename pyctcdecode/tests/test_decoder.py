# Copyright 2021-present Kensho Technologies, LLC.
import json
import math
from multiprocessing.context import SpawnContext
import os
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st
import kenlm  # type: ignore
import numpy as np

from ..alphabet import BPE_TOKEN, UNK_BPE_TOKEN, Alphabet
from ..decoder import (
    Beam,
    BeamSearchDecoderCTC,
    LMBeam,
    OutputBeam,
    _merge_beams,
    _normalize_whitespace,
    _prune_history,
    _sort_and_trim_beams,
    _sum_log_scores,
    build_ctcdecoder,
)
from ..language_model import HotwordScorer, LanguageModel, MultiLanguageModel
from .helpers import TempfileTestCase


def _random_matrix(rows: int, cols: int) -> np.ndarray:
    return np.random.normal(size=(rows, cols))


def _random_logits(rows: int, cols: int) -> np.ndarray:
    """Sample random logit matrix of given dimension."""
    xs = np.exp(_random_matrix(rows, cols))
    ps = (xs.T / np.sum(xs, axis=1)).T
    logits = np.log(ps)
    return logits


def _random_libri_logits(n: int) -> np.ndarray:
    return _random_logits(n, len(LIBRI_LABELS) + 1)


def _approx_beams(beams, precis=5):
    """Return beams with scores rounded."""
    return [
        [
            b.text,
            b.next_word,
            b.partial_word,
            b.last_char,
            b.text_frames,
            b.partial_frames,
            round(b.logit_score, precis),
        ]
        for b in beams
    ]


def _approx_output_beams(beams, precis=5):
    """Return beams with scores rounded."""
    simple_beams = []
    for beam in beams:
        text = beam.text
        frames = beam.text_frames
        s1 = beam.logit_score
        s2 = beam.lm_score
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
        input_beams = [
            LMBeam("Batman and", "", "Robi", "i", [], (-1, -1), -1.0, -3.0),
            LMBeam("Batman and", "Robin", "", "", [], (-1, -1), -1.0, -9.0),
            LMBeam("Batman and", "", "Robi", "", [], (-1, -1), -1.0, -5.0),
        ]
        beams = _sort_and_trim_beams(input_beams, 2)
        expected_beams = [
            LMBeam("Batman and", "", "Robi", "i", [], (-1, -1), -1.0, -3.0),
            LMBeam("Batman and", "", "Robi", "", [], (-1, -1), -1.0, -5.0),
        ]
        self.assertListEqual(beams, expected_beams)

    def test_merge_beams(self):
        beams = [
            Beam("Batman and", "", "Robi", "i", [], (-1, -1), -1.0),
            Beam("Batman and", "Robin", "", "", [], (-1, -1), -1.0),
            Beam("Batman and", "", "Robi", "", [], (-1, -1), -1.0),
            Beam("Batman and", "", "Robi", "", [], (-1, -1), -1.0),
            Beam("Batman &", "", "Robi", "", [], (-1, -1), -1.0),
        ]
        merged_beams = _merge_beams(beams)
        expected_merged_beams = [
            Beam("Batman and", "", "Robi", "i", [], (-1, -1), -1.0),
            Beam("Batman and", "Robin", "", "", [], (-1, -1), -1.0),
            Beam("Batman and", "", "Robi", "", [], (-1, -1), math.log(2 * math.exp(-1))),
            Beam("Batman &", "", "Robi", "", [], (-1, -1), -1.0),
        ]
        self.assertListEqual(_approx_beams(merged_beams), _approx_beams(expected_merged_beams))

    def test_prune_history(self):
        beams = [
            LMBeam("A Gandalf owns", "", "potatoes", "s", [], (-1, -1), -1.0, -1.0),
            LMBeam("B Gandalf owns", "", "potatoes", "", [], (-1, -1), -1.0, -1.0),
            LMBeam("C Gandalf owns", "", "potatoes", "s", [], (-1, -1), -1.0, -1.0),
            LMBeam("D Gandalf sells", "", "yeast", "", [], (-1, -1), -1.0, -1.0),
            LMBeam("E Gandalf owns", "", "yeast", "", [], (-1, -1), -1.0, -1.0),
        ]
        pruned_beams = _prune_history(beams, 3)
        expected_pruned_beams = [
            Beam("A Gandalf owns", "", "potatoes", "s", [], (-1, -1), -1.0),
            Beam("B Gandalf owns", "", "potatoes", "", [], (-1, -1), -1.0),
            Beam("D Gandalf sells", "", "yeast", "", [], (-1, -1), -1.0),
            Beam("E Gandalf owns", "", "yeast", "", [], (-1, -1), -1.0),
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


# Replacement for `multiprocessing.Pool` to get reliable tests that can't crash.
class MockPool:
    def __init__(self, ctx):
        """Fake pool to be able to run map."""
        self._ctx = ctx
        self.map_has_run = False

    def map(self, func, list_items):
        """Map."""
        self.map_has_run = True
        return [func(e) for e in list_items]


# A fake context to use with the mock pool
class MockContext:
    """Does nothing."""


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
        pool = MockPool(MockContext())
        text_list = decoder.decode_batch(pool, [TEST_LOGITS] * 5)
        expected_text_list = ["bugs bunny"] * 5
        self.assertListEqual(expected_text_list, text_list)
        self.assertTrue(pool.map_has_run)

        text_list = decoder.decode_batch(None, [TEST_LOGITS] * 5)
        self.assertListEqual(expected_text_list, text_list)

        spawn_pool = MockPool(SpawnContext())
        text_list = decoder.decode_batch(spawn_pool, [TEST_LOGITS] * 5)
        self.assertListEqual(expected_text_list, text_list)
        self.assertFalse(spawn_pool.map_has_run)

    def test_logit_shape_mismatch(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS)
        wrong_shape_logits = np.hstack([TEST_LOGITS] * 2)
        with self.assertRaises(ValueError):
            _ = decoder.decode(wrong_shape_logits)

    def test_decode_beams_batch(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH, TEST_UNIGRAMS)
        pool = MockPool(MockContext())
        text_list = decoder.decode_beams_batch(pool, [TEST_LOGITS] * 5)
        expected_text_list = [
            [
                OutputBeam(
                    "bugs bunny",
                    None,
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
            [
                OutputBeam(
                    "bugs bunny",
                    None,
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
            [
                OutputBeam(
                    "bugs bunny",
                    None,
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
            [
                OutputBeam(
                    "bugs bunny",
                    None,
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
            [
                OutputBeam(
                    "bugs bunny",
                    None,
                    [("bugs", (0, 4)), ("bunny", (7, 13))],
                    -2.853399551509947,
                    0.14660044849005294,
                )
            ],
        ]
        self.assertListEqual(expected_text_list, text_list)
        self.assertTrue(pool.map_has_run)

        text_list = decoder.decode_beams_batch(None, [TEST_LOGITS] * 5)
        self.assertListEqual(expected_text_list, text_list)

        spawn_pool = MockPool(SpawnContext())
        text_list = decoder.decode_beams_batch(spawn_pool, [TEST_LOGITS] * 5)
        self.assertListEqual(expected_text_list, text_list)
        self.assertFalse(spawn_pool.map_has_run)

    def test_multi_lm(self):
        alphabet = Alphabet.build_alphabet(SAMPLE_LABELS)
        lm = LanguageModel(TEST_KENLM_MODEL)
        decoder = BeamSearchDecoderCTC(alphabet, lm)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

        # constructing a multi lm with twice the same should average to the same result
        multi_lm = MultiLanguageModel([lm, lm])
        decoder_multi_lm = BeamSearchDecoderCTC(alphabet, multi_lm)
        text = decoder_multi_lm.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

        beams_1 = decoder.decode_beams(TEST_LOGITS)
        beams_2 = decoder_multi_lm.decode_beams(TEST_LOGITS)
        self.assertListEqual(_approx_output_beams(beams_1), _approx_output_beams(beams_2))

    def test_pruning(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH)
        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")
        text = _greedy_decode(TEST_LOGITS, decoder._alphabet)  # pylint: disable=W0212
        self.assertEqual(text, "bunny bunny")
        # setting a token threshold higher than one will force only argmax characters
        text = decoder.decode(TEST_LOGITS, token_min_logp=0.0)
        self.assertEqual(text, "bunny bunny")

    def test_history_pruning(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS)
        # make test logits where first token is ambiguous and all following tokens are clear
        add_probs = np.vstack([SPACE_PROBS, BUNNY_PROBS])
        logits = np.log(np.clip(np.vstack([TEST_PROBS] + [add_probs] * 5), 1e-15, 1))
        beams = decoder.decode_beams(logits, prune_history=False)
        beams_pruned = decoder.decode_beams(logits, prune_history=True)
        # make sure top result is the same
        self.assertEqual(beams[0].text, beams_pruned[0].text)
        # history pruning will have a strong effect on diversity here
        self.assertEqual(len(beams), 16)
        self.assertEqual(len(beams_pruned), 1)

    def test_stateful(self):
        bunny_bunny_probs = np.vstack(
            [
                BUGS_PROBS,
                SPACE_PROBS,
                np.vstack([BUGS_PROBS, BLANK_PROBS, BLANK_PROBS]) * 0.51 + BUNNY_PROBS * 0.49,
            ]
        )

        # without a LM we get bugs bugs as most likely
        no_lm_decoder = build_ctcdecoder(SAMPLE_LABELS)
        text = no_lm_decoder.decode(bunny_bunny_probs)
        self.assertEqual(text, "bugs bugs")

        # now let's add a LM
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH, TEST_UNIGRAMS)

        # LM correctly picks up the higher bigram probability for 'bugs bunny' over 'bugs bugs'
        text = decoder.decode(bunny_bunny_probs)
        self.assertEqual(text, "bugs bunny")

        # when splitting the LM can only see unigrams
        text = decoder.decode(bunny_bunny_probs[:4]) + " " + decoder.decode(bunny_bunny_probs[4:])
        self.assertEqual(text, "bugs bugs")

        # if we keep state from the first unigram then the second can be scored correctly
        top_result = decoder.decode_beams(bunny_bunny_probs[:4])[0]
        text = top_result.text
        lm_state = top_result.last_lm_state
        text += " " + decoder.decode_beams(bunny_bunny_probs[4:], lm_start_state=lm_state)[0].text
        self.assertEqual(text, "bugs bunny")

    def test_hotwords(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH)

        text = decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bugs bunny")

        text = decoder.decode(TEST_LOGITS, hotwords=["bunny"], hotword_weight=20)
        self.assertEqual(text, "bunny bunny")

        text = decoder.decode(TEST_LOGITS, hotwords=["bugs", "bunny"], hotword_weight=20)
        self.assertEqual(text, "bugs bunny")

        text = decoder.decode(TEST_LOGITS, hotwords=["bugs bunny"], hotword_weight=20)
        self.assertEqual(text, "bugs bunny")

        # check that this also works without language model
        no_lm_decoder = build_ctcdecoder(SAMPLE_LABELS)

        text = no_lm_decoder.decode(TEST_LOGITS)
        self.assertEqual(text, "bunny bunny")

        text = no_lm_decoder.decode(TEST_LOGITS, hotwords=["bugs"])
        self.assertEqual(text, "bugs bunny")

    def test_beam_results(self):
        # build a basic decoder with LM to get all possible combinations
        decoder = build_ctcdecoder(SAMPLE_LABELS)

        beams = decoder.decode_beams(TEST_LOGITS)
        self.assertEqual(len(beams), 16)

        # the best beam should be bunny bunny
        top_beam = beams[0]
        self.assertEqual(top_beam.text, "bunny bunny")

        # the worst beam should be bugs bunny
        worst_beam = beams[-1]
        self.assertEqual(worst_beam.text, "bugs bunny")

        # if we add the language model, that should push bugs bunny to the top, far enough to
        # remove all other beams from the output
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH)
        beams = decoder.decode_beams(TEST_LOGITS)
        self.assertEqual(len(beams), 1)
        top_beam = beams[0]
        self.assertEqual(top_beam.text, "bugs bunny")

        # if we don't punish <unk> and don't prune beams by score we recover all but sorted
        # correctly with 'bugs bunny' and the top (bigram LM) and 'bunny bunny' second (unigram LM)
        alphabet = Alphabet.build_alphabet(SAMPLE_LABELS)
        language_model = LanguageModel(TEST_KENLM_MODEL, unk_score_offset=0.0)
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        beams = decoder.decode_beams(TEST_LOGITS, beam_prune_logp=-20.0)
        self.assertEqual(len(beams), 16)
        self.assertEqual(beams[0].text, "bugs bunny")
        self.assertEqual(beams[1].text, "bunny bunny")

    def test_partial_decode(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS)

        # First we want to confirm that partial decoding does not change if you
        # process the whole logits versus splitting into several chunks
        beams, cached_lm_scores, cached_p_lm_scores = decoder.get_starting_state()
        final_beams = decoder.partial_decode_beams(
            TEST_LOGITS, cached_lm_scores, cached_p_lm_scores, beams, 0, is_end=True
        )

        logits1 = TEST_LOGITS[:3]
        logits2 = TEST_LOGITS[3:8]
        logits3 = TEST_LOGITS[8:]
        beams, cached_lm_scores, cached_p_lm_scores = decoder.get_starting_state()
        beams = decoder.partial_decode_beams(
            logits1, cached_lm_scores, cached_p_lm_scores, beams, 0, is_end=False
        )
        beams = decoder.partial_decode_beams(
            logits2, cached_lm_scores, cached_p_lm_scores, beams, 3, is_end=False
        )
        partial_final_beams = decoder.partial_decode_beams(
            logits3, cached_lm_scores, cached_p_lm_scores, beams, 8, is_end=True
        )

        self.assertEqual(len(final_beams), len(partial_final_beams))
        self.assertEqual("bunny bunny", partial_final_beams[0].text)
        self.assertEqual([(0, 6), (7, 13)], partial_final_beams[0].text_frames)
        self.assertAlmostEqual(-2.6933782130551505, partial_final_beams[0].logit_score)

        for final_beam, partial_final_beam in zip(final_beams, partial_final_beams):
            self.assertEqual(final_beam.text, partial_final_beam.text)
            self.assertEqual(final_beam.text_frames, partial_final_beam.text_frames)
            self.assertAlmostEqual(final_beam.logit_score, partial_final_beam.logit_score)

        # Now we want to confirm that partial decoding matches full decoding
        decoded_beams = decoder.decode_beams(TEST_LOGITS)
        self.assertEqual(len(decoded_beams), len(partial_final_beams))
        for decoded_beam, partial_final_beam in zip(decoded_beams, partial_final_beams):
            self.assertEqual(decoded_beam.text, partial_final_beam.text)
            self.assertEqual(
                [text_frame[1] for text_frame in decoded_beam.text_frames],
                partial_final_beam.text_frames,
            )
            self.assertAlmostEqual(decoded_beam.logit_score, partial_final_beam.logit_score)

    def test_partial_decode_with_lm(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH)
        logits1 = TEST_LOGITS[:3]
        logits2 = TEST_LOGITS[3:8]
        logits3 = TEST_LOGITS[8:]
        beams, cached_lm_scores, cached_p_lm_scores = decoder.get_starting_state()
        beams = decoder.partial_decode_beams(
            logits1, cached_lm_scores, cached_p_lm_scores, beams, 0, is_end=False
        )
        beams = decoder.partial_decode_beams(
            logits2, cached_lm_scores, cached_p_lm_scores, beams, 3, is_end=False
        )
        partial_final_beams = decoder.partial_decode_beams(
            logits3, cached_lm_scores, cached_p_lm_scores, beams, 8, is_end=True
        )
        decoded_beams = decoder.decode_beams(TEST_LOGITS)
        self.assertEqual("bugs bunny", partial_final_beams[0].text)
        self.assertEqual(len(decoded_beams), len(partial_final_beams))
        for decoded_beam, partial_final_beam in zip(decoded_beams, partial_final_beams):
            self.assertEqual(decoded_beam.text, partial_final_beam.text)
            self.assertEqual(
                [text_frame[1] for text_frame in decoded_beam.text_frames],
                partial_final_beam.text_frames,
            )
            self.assertAlmostEqual(decoded_beam.logit_score, partial_final_beam.logit_score)

    def test_partial_decode_with_hotwords(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS)
        hotword_scorer = HotwordScorer.build_scorer(["bugs"], weight=25.0)
        logits1 = TEST_LOGITS[:3]
        logits2 = TEST_LOGITS[3:8]
        logits3 = TEST_LOGITS[8:]
        beams, cached_lm_scores, cached_p_lm_scores = decoder.get_starting_state()
        beams = decoder.partial_decode_beams(
            logits1,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            0,
            hotword_scorer=hotword_scorer,
            is_end=False,
        )
        beams = decoder.partial_decode_beams(
            logits2,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            3,
            hotword_scorer=hotword_scorer,
            is_end=False,
        )
        partial_final_beams = decoder.partial_decode_beams(
            logits3,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            8,
            hotword_scorer=hotword_scorer,
            is_end=True,
        )
        decoded_beams = decoder.decode_beams(TEST_LOGITS, hotwords=["bugs"], hotword_weight=25.0)
        self.assertEqual("bugs bunny", partial_final_beams[0].text)
        self.assertEqual(len(decoded_beams), len(partial_final_beams))
        for decoded_beam, partial_final_beam in zip(decoded_beams, partial_final_beams):
            self.assertEqual(decoded_beam.text, partial_final_beam.text)
            self.assertEqual(
                [text_frame[1] for text_frame in decoded_beam.text_frames],
                partial_final_beam.text_frames,
            )
            self.assertAlmostEqual(decoded_beam.logit_score, partial_final_beam.logit_score)

    def test_partial_decode_with_multiple_hotword_scorers(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS)
        hotword_scorer = HotwordScorer.build_scorer(["bugs"], weight=15.0)
        hotword_scorer2 = HotwordScorer.build_scorer(["bunny"], weight=15.0)
        logits1 = TEST_LOGITS[:3]
        logits2 = TEST_LOGITS[3:8]
        logits3 = TEST_LOGITS[8:]
        beams, cached_lm_scores, cached_p_lm_scores = decoder.get_starting_state()
        beams = decoder.partial_decode_beams(
            logits1,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            0,
            hotword_scorer=hotword_scorer,
            is_end=False,
        )
        beams = decoder.partial_decode_beams(
            logits2,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            3,
            hotword_scorer=hotword_scorer2,
            is_end=False,
        )
        partial_final_beams = decoder.partial_decode_beams(
            logits3,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            8,
            hotword_scorer=None,
            is_end=True,
        )
        # With no LM, the first 3 characters become "bug" from the prefix of "bugs"
        # and then that hotword is dropped, so it's not completed
        self.assertEqual("bugny bunny", partial_final_beams[0].text)

        beams, cached_lm_scores, cached_p_lm_scores = decoder.get_starting_state()
        beams = decoder.partial_decode_beams(
            logits1,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            0,
            hotword_scorer=hotword_scorer,
            is_end=False,
        )
        beams = decoder.partial_decode_beams(
            logits2,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            3,
            hotword_scorer=hotword_scorer,
            is_end=False,
        )
        partial_final_beams = decoder.partial_decode_beams(
            logits3,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            8,
            hotword_scorer=hotword_scorer2,
            is_end=True,
        )
        self.assertEqual("bugs bunny", partial_final_beams[0].text)

    def test_frame_annotation(self):
        # build a basic decoder with LM to get all possible combinations
        decoder = build_ctcdecoder(SAMPLE_LABELS)

        beams = decoder.decode_beams(TEST_LOGITS)
        top_beam = beams[0]
        self.assertEqual(top_beam.text, "bunny bunny")
        # the frame annotations returned should correspond to the position of the two words
        # within the logit matrix
        expected_frames = [("bunny", (0, 6)), ("bunny", (7, 13))]
        self.assertEqual(len(top_beam.text_frames), len(expected_frames))
        self.assertListEqual(top_beam.text_frames, expected_frames)

        worst_beam = beams[-1]
        self.assertEqual(worst_beam.text, "bugs bunny")
        # the first word is of length 4 now, so check for correct frame annotations (ctc blank
        # character won't be included in the frame count if it's at the end of a word)
        expected_frames = [("bugs", (0, 4)), ("bunny", (7, 13))]
        self.assertEqual(len(worst_beam.text_frames), len(expected_frames))
        self.assertListEqual(worst_beam.text_frames, expected_frames)

        # check if frame annotation works in ctc stretched word
        stretched_chars = [" ", "", "b", "u", "n", "", "n", "n", "y", "", " ", " "]
        test_frame_logits = np.zeros((len(stretched_chars), len(SAMPLE_VOCAB)))
        for n, c in enumerate(stretched_chars):
            test_frame_logits[n][SAMPLE_VOCAB.get(c)] = 1
        top_beam = decoder.decode_beams(test_frame_logits)[0]
        self.assertEqual(top_beam.text, "bunny")
        expected_frames = [("bunny", (2, 9))]
        self.assertEqual(len(top_beam.text_frames), len(expected_frames))
        self.assertListEqual(top_beam.text_frames, expected_frames)

        # test for bpe vocabulary
        bpe_labels = ["▁bugs", "▁bun", "ny", ""]
        bpe_vocab = {c: n for n, c in enumerate(bpe_labels)}
        bpe_decoder = build_ctcdecoder(bpe_labels)
        bpe_ctc_out = ["", "▁bugs", "▁bun", "ny", "ny", ""]
        test_frame_logits = np.zeros((len(bpe_ctc_out), len(bpe_vocab)))
        for n, c in enumerate(bpe_ctc_out):
            test_frame_logits[n][bpe_vocab.get(c)] = 1
        top_beam = bpe_decoder.decode_beams(test_frame_logits)[0]
        self.assertEqual(top_beam.text, "bugs bunny")
        expected_frames = [("bugs", (1, 2)), ("bunny", (2, 5))]
        self.assertEqual(len(top_beam.text_frames), len(expected_frames))
        self.assertListEqual(top_beam.text_frames, expected_frames)

    def test_realistic_alphabet(self):
        decoder = build_ctcdecoder(LIBRI_LABELS)
        text = decoder.decode(LIBRI_LOGITS)
        expected_text = (
            "i have a good deal of will you remember and what i have set my mind upon no doubt "
            "i shall some day achieve"
        )
        self.assertEqual(text, expected_text)
        beams = decoder.decode_beams(LIBRI_LOGITS)
        # check that every word received frame annotations
        self.assertEqual(len(beams[0].text.split()), len(beams[0].text_frames))

        # test with fake BPE vocab, spoof space with with ▁▁
        libri_labels_bpe = [UNK_BPE_TOKEN, BPE_TOKEN] + ["##" + c for c in LIBRI_LABELS[1:]]
        zero_row = np.array([[-100.0] * LIBRI_LOGITS.shape[0]]).T
        libri_logits_bpe = np.hstack([zero_row, LIBRI_LOGITS])
        decoder = build_ctcdecoder(libri_labels_bpe)
        text = decoder.decode(libri_logits_bpe)
        expected_text = (
            "i have a good deal of will you remember and what i have set my mind upon no doubt "
            "i shall some day achieve"
        )
        self.assertEqual(text, expected_text)
        # check that every word received frame annotations
        self.assertEqual(len(beams[0].text.split()), len(beams[0].text_frames))

    @settings(deadline=1000)
    @given(st.builds(_random_libri_logits, st.integers(min_value=0, max_value=20)))
    def test_fuzz_decode(self, logits: np.ndarray):
        """Ensure decoder is robust to random logit inputs."""
        decoder = build_ctcdecoder(LIBRI_LABELS)
        decoder.decode(logits)

    @settings(deadline=1000)
    @given(
        st.builds(
            _random_matrix,
            st.integers(min_value=0, max_value=20),
            st.integers(min_value=len(LIBRI_LABELS) + 1, max_value=len(LIBRI_LABELS) + 1),
        )
    )
    def test_invalid_logit_inputs(self, logits: np.ndarray):
        decoder = build_ctcdecoder(LIBRI_LABELS)
        decoder.decode(logits)

    @given(
        alpha=st.one_of(st.none(), st.floats()),
        beta=st.one_of(st.none(), st.floats()),
        unk_score_offset=st.one_of(st.none(), st.floats()),
        lm_score_boundary=st.one_of(st.none(), st.booleans()),
    )
    def test_fuzz_reset_params(self, alpha, beta, unk_score_offset, lm_score_boundary):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH, alpha=0.0)
        decoder.reset_params(
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            lm_score_boundary=lm_score_boundary,
        )


class TestSerialization(TempfileTestCase):
    @classmethod
    def _count_num_language_models(cls):
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
            ("language_model",),  # missing alphabet
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
        import huggingface_hub

        if tuple(int(x) for x in huggingface_hub.__version__.split(".")[:3]) >= (0, 8, 0):
            from huggingface_hub.constants import REPO_ID_SEPARATOR

            new_hub_structure = True
        else:
            from huggingface_hub.snapshot_download import (  # pylint: disable=import-error
                REPO_ID_SEPARATOR,
            )

            new_hub_structure = False

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
        dummy_hub_name_with_separator = dummy_hub_name.replace("/", REPO_ID_SEPARATOR)
        hash_value = "123456aoeusnth"
        if new_hub_structure:
            dummy_cached_subdir = (
                f"models{REPO_ID_SEPARATOR}{dummy_hub_name_with_separator}/snapshots/{hash_value}"
            )
        else:
            dummy_cached_subdir = f"{dummy_hub_name_with_separator}.main.{hash_value}"
        dummy_cached_dir = os.path.join(self.temp_dir, dummy_cached_subdir)
        os.makedirs(dummy_cached_dir)
        if new_hub_structure:
            models_dir = f"models{REPO_ID_SEPARATOR}{dummy_hub_name_with_separator}"
            os.makedirs(f"{self.temp_dir}/{models_dir}/refs")
            with open(f"{self.temp_dir}/{models_dir}/refs/main", "w") as ref_file:
                ref_file.write(hash_value)

        # save decoder
        decoder.save_to_dir(os.path.join(self.temp_dir, dummy_cached_dir))

        # load from cache in offline mode
        new_decoder = BeamSearchDecoderCTC.load_from_hf_hub(
            dummy_hub_name, cache_dir=self.temp_dir, local_files_only=True
        )

        new_text = new_decoder.decode(TEST_LOGITS)
        self.assertEqual(text, new_text)
