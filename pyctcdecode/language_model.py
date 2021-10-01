# Copyright 2021-present Kensho Technologies, LLC.
from __future__ import division

import abc
import logging
import re
from typing import Collection, Iterable, List, Optional, Pattern, Set, Tuple, cast

import numpy as np
from pygtrie import CharTrie  # type: ignore

from .constants import (
    AVG_TOKEN_LEN,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_HOTWORD_WEIGHT,
    DEFAULT_SCORE_LM_BOUNDARY,
    DEFAULT_UNK_LOGP_OFFSET,
    LOG_BASE_CHANGE_FACTOR,
)


logger = logging.getLogger(__name__)


try:
    import kenlm  # type: ignore
except ImportError:
    logger.warning(
        "kenlm python bindings are not installed. Most likely you want to install it using: "
        "pip install https://github.com/kpu/kenlm/archive/master.zip"
    )


def load_unigram_set_from_arpa(arpa_path: str) -> Set[str]:
    """Read unigrams from arpa file."""
    unigrams = set()
    with open(arpa_path) as f:
        start_1_gram = False
        for line in f:
            line = line.strip()
            if line == "\\1-grams:":
                start_1_gram = True
            elif line == "\\2-grams:":
                break
            if start_1_gram and len(line) > 0:
                parts = line.split("\t")
                if len(parts) == 3:
                    unigrams.add(parts[1])
    if len(unigrams) == 0:
        raise ValueError("No unigrams found in arpa file. Something is wrong with the file.")
    return unigrams


def _prepare_unigram_set(unigrams: Collection[str], kenlm_model: kenlm.Model) -> Set[str]:
    """Filter unigrams down to vocabulary that exists in kenlm_model."""
    if len(unigrams) < 1000:
        logger.warning(
            "Only %s unigrams passed as vocabulary. Is this small or artificial data?",
            len(unigrams),
        )
    unigram_set = set(unigrams)
    unigram_set = set([t for t in unigram_set if t in kenlm_model])
    retained_fraction = 1.0 if len(unigrams) == 0 else len(unigram_set) / len(unigrams)
    if retained_fraction < 0.1:
        logger.warning(
            "Only %s%% of unigrams in vocabulary found in kenlm model-- this might mean that your "
            "vocabulary and language model are incompatible. Is this intentional?",
            round(retained_fraction * 100, 1),
        )
    return unigram_set


def _get_empty_lm_state() -> kenlm.State:
    """Get unintialized kenlm state."""
    try:
        kenlm_state = kenlm.State()
    except ImportError:
        raise ValueError("To use a language model, you need to install kenlm.")
    return kenlm_state


class HotwordScorer:
    def __init__(
        self,
        match_ptn: Pattern[str],
        char_trie: CharTrie,
        weight: float = DEFAULT_HOTWORD_WEIGHT,
    ) -> None:
        """Scorer for hotwords if provided.

        Args:
            match_ptn: match pattern for hotword unigrams
            char_trie: trie for all hotwords to do partial matching
            weight: weight for score increase
        """
        self._match_ptn = match_ptn
        self._char_trie = char_trie
        self._weight = weight

    def __contains__(self, item: str) -> bool:
        """Contains."""
        return cast(bool, self._char_trie.has_node(item) > 0)

    def score(self, text: str) -> float:
        """Get total hotword score for input text."""
        return self._weight * len(self._match_ptn.findall(text))

    def score_partial_token(self, token: str) -> float:
        """Get total hotword score for input text."""
        if token in self:
            # find shortest unigram starting with the given partial token
            min_len = len(next(self._char_trie.iterkeys(token, shallow=True)))
            # scale score by length of unigram matched so far
            score = self._weight * len(token) / min_len
        else:
            score = 0.0
        return score

    @classmethod
    def build_scorer(
        cls, hotwords: Optional[Iterable[str]] = None, weight: float = DEFAULT_HOTWORD_WEIGHT
    ) -> "HotwordScorer":
        """Use hotword list to create regex pattern and character trie for scoring."""
        # make sure we get an iterable
        hotwords = hotwords or []
        # remove whitespace
        hotwords = [s.strip() for s in hotwords if len(s.strip()) > 0]
        if len(hotwords) > 0:
            hotword_unigrams = []
            for ngram in hotwords:
                # split ngrams to get words
                for unigram in ngram.split():
                    hotword_unigrams.append(unigram)

            # create pattern to match full words
            # sort by length to get longest possible match
            # use lookahead and lookbehind to match on word boundary instead of '\b' to only match
            # on space or bos/eos
            match_ptn = re.compile(
                r"|".join(
                    [
                        r"(?<!\S)" + re.escape(s) + r"(?!\S)"
                        for s in sorted(hotword_unigrams, key=len, reverse=True)
                    ]
                )
            )

            # create trie for partial word matches
            char_trie = CharTrie.fromkeys(hotword_unigrams)
        else:
            # make an unmatchable pattern
            match_ptn = re.compile(r"^\b$")
            # empty trie
            char_trie = CharTrie()
        return cls(match_ptn, char_trie, weight)


class AbstractLanguageModel(abc.ABC):
    @property
    @abc.abstractmethod
    def order(self) -> int:
        """Get the order of the n-gram language model."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_start_state(self) -> List[kenlm.State]:
        """Get initial lm state."""
        raise NotImplementedError()

    @abc.abstractmethod
    def score_partial_token(self, partial_token: str) -> float:
        """Get partial token score."""
        raise NotImplementedError()

    @abc.abstractmethod
    def score(
        self, prev_state: kenlm.State, word: str, is_last_word: bool = False
    ) -> Tuple[float, kenlm.State]:
        """Score word conditional on previous lm state."""
        raise NotImplementedError()


class LanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        kenlm_model: kenlm.Model,
        unigrams: Optional[Collection[str]] = None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        unk_score_offset: float = DEFAULT_UNK_LOGP_OFFSET,
        score_boundary: bool = DEFAULT_SCORE_LM_BOUNDARY,
    ) -> None:
        """Language model container class to consolidate functionality.

        Args:
            kenlm_model: instance of kenlm n-gram language model `kenlm.Model`
            unigrams: list of known word unigrams
            alpha: weight for language model during shallow fusion
            beta: weight for length score adjustment of during scoring
            unk_score_offset: amount of log score offset for unknown tokens
            score_boundary: whether to have kenlm respect boundaries when scoring
        """
        self._kenlm_model = kenlm_model
        if unigrams is None:
            logger.warning("No known unigrams provided, decoding results might be a lot worse.")
            unigram_set = set()
            char_trie = None
        else:
            unigram_set = _prepare_unigram_set(unigrams, self._kenlm_model)
            char_trie = CharTrie.fromkeys(unigram_set)
        self._unigram_set = unigram_set
        self._char_trie = char_trie
        self.alpha = alpha
        self.beta = beta
        self.unk_score_offset = unk_score_offset
        self.score_boundary = score_boundary

    @property
    def order(self) -> int:
        """Get the order of the n-gram language model."""
        return cast(int, (self._kenlm_model.order))

    def get_start_state(self) -> kenlm.State:
        """Get initial lm state."""
        start_state = _get_empty_lm_state()
        if self.score_boundary:
            self._kenlm_model.BeginSentenceWrite(start_state)
        else:
            self._kenlm_model.NullContextWrite(start_state)
        return start_state

    def _get_raw_end_score(self, start_state: kenlm.State) -> float:
        """Calculate final lm score."""
        if self.score_boundary:
            end_state = _get_empty_lm_state()
            score: float = self._kenlm_model.BaseScore(start_state, "</s>", end_state)
        else:
            score = 0.0
        return score

    def score_partial_token(self, partial_token: str) -> float:
        """Get partial token score."""
        if self._char_trie is None:
            is_oov = 1.0
        else:
            is_oov = int(self._char_trie.has_node(partial_token) == 0)
        unk_score = self.unk_score_offset * is_oov
        # if unk token length exceeds expected length then additionally decrease score
        if len(partial_token) > AVG_TOKEN_LEN:
            unk_score = unk_score * len(partial_token) / AVG_TOKEN_LEN
        return unk_score

    def score(
        self, prev_state: kenlm.State, word: str, is_last_word: bool = False
    ) -> Tuple[float, kenlm.State]:
        """Score word conditional on start state."""
        end_state = _get_empty_lm_state()
        lm_score = self._kenlm_model.BaseScore(prev_state, word, end_state)
        # override UNK prob. use unigram set if we have because it's faster
        if (
            len(self._unigram_set) > 0
            and word not in self._unigram_set
            or word not in self._kenlm_model
        ):
            lm_score += self.unk_score_offset
        # add end of sentence context if needed
        if is_last_word:
            # note that we want to return the unmodified end_state to keep extension capabilities
            lm_score = lm_score + self._get_raw_end_score(end_state)
        lm_score = self.alpha * lm_score * LOG_BASE_CHANGE_FACTOR + self.beta
        return lm_score, end_state


class MultiLanguageModel(AbstractLanguageModel):
    def __init__(self, language_models: List[LanguageModel]) -> None:
        """Container for multiple language models.

        Args:
            language_models: list of language models
        """
        if len(language_models) < 2:
            raise ValueError("This class is meant to contain at least 2 language models.")
        self._language_models = language_models

    @property
    def order(self) -> int:
        """Get the maximum order of the contained n-gram language model."""
        return max([lm.order for lm in self._language_models])

    def get_start_state(self) -> List[kenlm.State]:
        """Get initial lm state."""
        return [lm.get_start_state() for lm in self._language_models]

    def score_partial_token(self, partial_token: str) -> float:
        """Get partial token score."""
        return float(
            np.mean([lm.score_partial_token(partial_token) for lm in self._language_models])
        )

    def score(
        self, prev_state: List[kenlm.State], word: str, is_last_word: bool = False
    ) -> Tuple[float, List[kenlm.State]]:
        """Score word conditional on previous lm state."""
        score = 0.0
        end_state = []
        for lm_prev_state, lm in zip(prev_state, self._language_models):
            lm_score, lm_end_state = lm.score(lm_prev_state, word, is_last_word=is_last_word)
            score += lm_score
            end_state.append(lm_end_state)
        score = score / len(self._language_models)
        return score, end_state
