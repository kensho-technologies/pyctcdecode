# Copyright 2021-present Kensho Technologies, LLC.
import os
import re
import unittest

from hypothesis import given
from hypothesis import strategies as st
import kenlm
from pygtrie import CharTrie

from ..language_model import HotwordScorer, LanguageModel, MultiLanguageModel
from .helpers import TempfileTestCase


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
KENLM_BINARY_PATH = os.path.join(CUR_PATH, "sample_data", "bugs_bunny_kenlm.arpa")


class TestLanguageModel(unittest.TestCase):
    def test_match_ptn(self):
        hotwords = ["tyrion lannister", "hodor"]
        match_ptn = HotwordScorer.build_scorer(hotwords)._match_ptn  # pylint: disable=W0212

        matched_tokens = match_ptn.findall("i work with hodor and friends")
        expected_tokens = ["hodor"]
        self.assertListEqual(matched_tokens, expected_tokens)

        # sub-ngramming
        matched_tokens = match_ptn.findall("we can match tyrion only")
        expected_tokens = ["tyrion"]
        self.assertListEqual(matched_tokens, expected_tokens)

        # bos/eos
        matched_tokens = match_ptn.findall("hodor is friends with hodor")
        expected_tokens = ["hodor", "hodor"]
        self.assertListEqual(matched_tokens, expected_tokens)

        # word boundary only defined by space and bos/eos
        matched_tokens = match_ptn.findall("do not match hodor, or anything else here")
        expected_tokens = []
        self.assertListEqual(matched_tokens, expected_tokens)

        # punctuation compatibility
        hotwords = ["hodor,"]
        match_ptn = HotwordScorer.build_scorer(hotwords)._match_ptn  # pylint: disable=W0212
        matched_tokens = match_ptn.findall("please match hodor, but not hodor")
        expected_tokens = ["hodor,"]
        self.assertListEqual(matched_tokens, expected_tokens)

    def test_trie(self):
        hotwords = ["tyrion lannister", "hodor"]
        char_trie = HotwordScorer.build_scorer(hotwords)._char_trie  # pylint: disable=W0212
        has_token = char_trie.has_node("hod") > 0
        self.assertTrue(has_token)
        has_token = char_trie.has_node("dor") > 0
        self.assertFalse(has_token)

        # works for full tokens as well
        has_token = char_trie.has_node("hodor") > 0
        self.assertTrue(has_token)

        # sub-ngramming
        has_token = char_trie.has_node("lann") > 0
        self.assertTrue(has_token)

        # punctuation compatibility
        hotwords = ["U.S.A."]
        char_trie = HotwordScorer.build_scorer(hotwords)._char_trie  # pylint: disable=W0212
        has_token = char_trie.has_node("U.S") > 0
        self.assertTrue(has_token)


# fuzz tests below generated with `hypothesis write language_model.py` and edited for concision.


class TestFuzzMultiLanguageModel(unittest.TestCase):
    @given(
        language_models=st.lists(
            st.builds(
                LanguageModel,
                kenlm_model=st.just(kenlm.Model(KENLM_BINARY_PATH)),
                alpha=st.one_of(st.just(0.5), st.floats()),
                beta=st.one_of(st.just(1.5), st.floats()),
                score_boundary=st.one_of(st.just(True), st.booleans()),
                unigrams=st.one_of(
                    st.none(),
                    st.lists(st.text()),
                ),
                unk_score_offset=st.one_of(st.just(-10.0), st.floats()),
            ),
        ),
    )
    def test_fuzz_MultiLanguageModel(self, language_models):
        if len(language_models) >= 2:
            MultiLanguageModel(language_models=language_models)
        else:
            with self.assertRaises(ValueError):
                MultiLanguageModel(language_models=language_models)


class TestHotwordScorer(unittest.TestCase):
    @given(match_ptn=st.just(re.compile("")), char_trie=st.builds(CharTrie), weight=st.floats())
    def test_fuzz_HotwordScorer(self, match_ptn, char_trie, weight):
        HotwordScorer(match_ptn=match_ptn, char_trie=char_trie, weight=weight)

    @given(
        unigrams=st.one_of(
            st.none(),
            st.lists(st.text()),
        ),
        alpha=st.floats(),
        beta=st.floats(),
        unk_score_offset=st.floats(),
        score_boundary=st.booleans(),
        partial_token=st.text(),
    )
    def test_fuzz_LanguageModel(
        self, unigrams, alpha, beta, unk_score_offset, score_boundary, partial_token
    ):
        kenlm_model = kenlm.Model(KENLM_BINARY_PATH)
        lm = LanguageModel(
            kenlm_model=kenlm_model,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            score_boundary=score_boundary,
        )
        lm.score_partial_token(partial_token)


class TestLanguageModelSerialization(TempfileTestCase):
    def test_parse_directory(self):
        good_filenames = [
            ("unigrams.txt", "something.arpa", "attrs.json"),
            ("unigrams.txt", "something.bin", "attrs.json"),
            ("unigrams.txt", "something.binary", "attrs.json"),
            ("unigrams.txt", "something.binary", "attrs.json", ".meaningless", "__pycache__"),
        ]

        bad_filenames = [
            ("something.arpa", "attrs.json"),  # missing unigrams
            ("unigrams.txt", "something.bin", "attrs.json", "extra-file.ext"),  # extra file
            ("unigrams.txt", "something.binary", "attributes.json"),  # wrong filename
        ]

        for filenames in good_filenames:
            self.clear_dir()
            for fn in filenames:
                with open(os.path.join(self.temp_dir, fn), "w") as fi:
                    fi.write("meaningless data")
            LanguageModel.parse_directory_contents(self.temp_dir)  # should not error out

        for filenames in bad_filenames:
            self.clear_dir()
            for fn in filenames:
                with open(os.path.join(self.temp_dir, fn), "w") as fi:
                    fi.write("meaningless data")
            with self.assertRaises(ValueError):
                LanguageModel.parse_directory_contents(self.temp_dir)

    def test_save_and_load_lm(self):
        kenlm_model = kenlm.Model(KENLM_BINARY_PATH)
        lm = LanguageModel(
            kenlm_model=kenlm_model,
            unigrams=["bugs", "bunny"],
            alpha=0.1,
        )
        partial_token = "bu"  # nosec
        score = lm.score_partial_token(partial_token)

        lm.save_to_dir(self.temp_dir)
        dir_contents = lm.parse_directory_contents(self.temp_dir)
        self.assertEqual(len(dir_contents), 3)

        new_lm = LanguageModel.load_from_dir(self.temp_dir)
        self.assertEqual(lm._unigram_set, new_lm._unigram_set)  # pylint: disable=protected-access
        self.assertEqual(lm.alpha, new_lm.alpha)
        self.assertEqual(lm.beta, new_lm.beta)

        new_score = new_lm.score_partial_token(partial_token)
        self.assertEqual(new_score, score)

        # do it again with different params
        # this makes sure things get overwritten properly
        # ie that unigrams are properly set to None  (and lm._unigram_set to empty set)
        lm = LanguageModel(kenlm_model=kenlm_model, unigrams=None, alpha=0.3, beta=0.2)
        lm.save_to_dir(self.temp_dir)
        dir_contents = lm.parse_directory_contents(self.temp_dir)
        self.assertEqual(len(dir_contents), 3)
        new_lm = LanguageModel.load_from_dir(self.temp_dir)
        self.assertEqual(lm._unigram_set, new_lm._unigram_set)  # pylint: disable=protected-access
        self.assertEqual(lm.alpha, new_lm.alpha)
        self.assertEqual(lm.beta, new_lm.beta)

        # do it again, make sure we can load the same thing twice without corrupting the item
        new_lm2 = LanguageModel.load_from_dir(self.temp_dir)
        self.assertEqual(lm._unigram_set, new_lm2._unigram_set)  # pylint: disable=protected-access
        self.assertEqual(lm.alpha, new_lm2.alpha)
        self.assertEqual(lm.beta, new_lm2.beta)
