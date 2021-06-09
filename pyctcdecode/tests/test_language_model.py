# Copyright 2021-present Kensho Technologies, LLC.
import unittest

from ..language_model import HotwordScorer


class TestLanugageModel(unittest.TestCase):
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
