# Copyright 2021-present Kensho Technologies, LLC.
import json
import math
import multiprocessing
import os
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st
import kenlm  # type: ignore
import numpy as np

from ..alphabet import BPE_TOKEN, UNK_BPE_TOKEN, Alphabet
from ..decoder import (
    BeamSearchDecoderCTC,
    _merge_beams,
    _normalize_whitespace,
    _prune_history,
    _sort_and_trim_beams,
    _sum_log_scores,
    build_ctcdecoder,
)
from ..language_model import LanguageModel, MultiLanguageModel


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
        with multiprocessing.Pool() as pool:
            text_list = decoder.decode_batch(pool, [TEST_LOGITS] * 5)
        expected_text_list = ["bugs bunny"] * 5
        self.assertListEqual(expected_text_list, text_list)

    def test_decode_beams_batch(self):
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH, TEST_UNIGRAMS)
        with multiprocessing.Pool() as pool:
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
        self.assertListEqual(_approx_lm_beams(beams_1), _approx_lm_beams(beams_2))

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
        self.assertEqual(beams[0][0], beams_pruned[0][0])
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
        text, lm_state, _, _, _ = decoder.decode_beams(bunny_bunny_probs[:4])[0]
        text += " " + decoder.decode_beams(bunny_bunny_probs[4:], lm_start_state=lm_state)[0][0]
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
        self.assertEqual(top_beam[0], "bunny bunny")

        # the worst beam should be bugs bunny
        worst_beam = beams[-1]
        self.assertEqual(worst_beam[0], "bugs bunny")

        # if we add the language model, that should push bugs bunny to the top, far enough to
        # remove all other beams from the output
        decoder = build_ctcdecoder(SAMPLE_LABELS, KENLM_MODEL_PATH)
        beams = decoder.decode_beams(TEST_LOGITS)
        self.assertEqual(len(beams), 1)
        top_beam = beams[0]
        self.assertEqual(top_beam[0], "bugs bunny")

        # if we don't punish <unk> and don't prune beams by score we recover all but sorted
        # correctly with 'bugs bunny' and the top (bigram LM) and 'bunny bunny' second (unigram LM)
        alphabet = Alphabet.build_alphabet(SAMPLE_LABELS)
        language_model = LanguageModel(TEST_KENLM_MODEL, unk_score_offset=0.0)
        decoder = BeamSearchDecoderCTC(alphabet, language_model)
        beams = decoder.decode_beams(TEST_LOGITS, beam_prune_logp=-20.0)
        self.assertEqual(len(beams), 16)
        self.assertEqual(beams[0][0], "bugs bunny")
        self.assertEqual(beams[1][0], "bunny bunny")

    def test_frame_annotation(self):
        # build a basic decoder with LM to get all possible combinations
        decoder = build_ctcdecoder(SAMPLE_LABELS)

        beams = decoder.decode_beams(TEST_LOGITS)
        top_beam = beams[0]
        self.assertEqual(top_beam[0], "bunny bunny")
        # the frame annotations returned should correspond to the position of the the two words
        # within the logit matrix
        expected_frames = [("bunny", (0, 6)), ("bunny", (7, 13))]
        self.assertEqual(len(top_beam[2]), len(expected_frames))
        self.assertListEqual(top_beam[2], expected_frames)

        worst_beam = beams[-1]
        self.assertEqual(worst_beam[0], "bugs bunny")
        # the first word is of length 4 now, so check for correct frame annotations (ctc blank
        # character won't be included in the frame count if it's at the end of a word)
        expected_frames = [("bugs", (0, 4)), ("bunny", (7, 13))]
        self.assertEqual(len(worst_beam[2]), len(expected_frames))
        self.assertListEqual(worst_beam[2], expected_frames)

        # check if frame annotation works in ctc stretched word
        stretched_chars = [" ", "", "b", "u", "n", "", "n", "n", "y", "", " ", " "]
        test_frame_logits = np.zeros((len(stretched_chars), len(SAMPLE_VOCAB)))
        for n, c in enumerate(stretched_chars):
            test_frame_logits[n][SAMPLE_VOCAB.get(c)] = 1
        top_beam = decoder.decode_beams(test_frame_logits)[0]
        self.assertEqual(top_beam[0], "bunny")
        expected_frames = [("bunny", (2, 9))]
        self.assertEqual(len(top_beam[2]), len(expected_frames))
        self.assertListEqual(top_beam[2], expected_frames)

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
        self.assertEqual(len(beams[0][0].split()), len(beams[0][2]))

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
        self.assertEqual(len(beams[0][0].split()), len(beams[0][2]))

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
