# Copyright 2021-present Kensho Technologies, LLC.
from __future__ import annotations, division

import dataclasses
import functools
import heapq
import logging
import math
import multiprocessing as mp
from multiprocessing.pool import Pool
import os
from pathlib import Path
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NBitBase, NDArray

from .alphabet import BPE_TOKEN, Alphabet, verify_alphabet_coverage
from .constants import (
    DEFAULT_ALPHA,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_BETA,
    DEFAULT_HOTWORD_WEIGHT,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_BEAMS,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_SCORE_LM_BOUNDARY,
    DEFAULT_UNK_LOGP_OFFSET,
    MIN_TOKEN_CLIP_P,
)
from .language_model import (
    AbstractLanguageModel,
    AbstractLMState,
    HotwordScorer,
    LanguageModel,
    load_unigram_set_from_arpa,
)


logger = logging.getLogger(__name__)

try:
    import kenlm  # type: ignore
except ImportError:
    logger.warning(
        "kenlm python bindings are not installed. Most likely you want to install it using: "
        "pip install https://github.com/kpu/kenlm/archive/master.zip"
    )

# type hints
# store frame information for each word, where frame is the logit index of (start_frame, end_frame)
Frames = Tuple[int, int]
WordFrames = Tuple[str, Frames]


@dataclasses.dataclass(frozen=True)
class Beam:
    """Contains all the info needed for decoding a beam."""

    text: str
    next_word: str
    partial_word: str
    last_char: Optional[str]
    text_frames: List[Frames]
    partial_frames: Frames
    logit_score: float

    @classmethod
    def from_lm_beam(cls, lm_beam: LMBeam) -> "Beam":
        """Create a Beam from an LMBeam by stripping out the LM score."""
        return Beam(
            text=lm_beam.text,
            next_word=lm_beam.next_word,
            partial_word=lm_beam.partial_word,
            last_char=lm_beam.last_char,
            text_frames=lm_beam.text_frames,
            partial_frames=lm_beam.partial_frames,
            logit_score=lm_beam.logit_score,
        )


@dataclasses.dataclass(frozen=True)
class LMBeam(Beam):
    """Contains everything in a beam plus an additional language model score."""

    lm_score: float


@dataclasses.dataclass(frozen=True)
class OutputBeam:
    """Output information."""

    text: str
    last_lm_state: Optional[AbstractLMState]
    text_frames: List[WordFrames]
    logit_score: float  # Cumulative logit score
    lm_score: float  # Cumulative language model + logit score

    def get_mp_safe_beam(self) -> "OutputBeam":
        """Get a multiprocessing-safe version of the beam."""
        if self.last_lm_state is None:
            last_lm_state = None
        else:
            last_lm_state = self.last_lm_state.get_mp_safe_state()
        return dataclasses.replace(self, last_lm_state=last_lm_state)


# Key for the language model score cache
# text, is_eos
LMScoreCacheKey = Tuple[str, bool]
# LM score with hotword score, raw LM score, AbstracatLMState
LMScoreCacheValue = Tuple[float, float, AbstractLMState]
LMScoreCache = Dict[LMScoreCacheKey, LMScoreCacheValue]

# constants
NULL_FRAMES: Frames = (-1, -1)  # placeholder that gets replaced with positive integer frame indices
EMPTY_START_BEAM = Beam("", "", "", None, [], NULL_FRAMES, 0.0)


# Generic float type
if sys.version_info < (3, 8):
    NpFloat = Any
else:
    if sys.version_info < (3, 9) and not TYPE_CHECKING:
        NpFloat = Any
    else:
        NpFloat = np.floating[NBitBase]

FloatVar = TypeVar("FloatVar", bound=NpFloat)
Shape = TypeVar("Shape")


def _get_valid_pool(pool: Optional[Pool]) -> Optional[Pool]:
    """Return the pool if the pool is appropriate for multiprocessing."""
    if pool is not None and isinstance(
        pool._ctx, mp.context.SpawnContext  # type: ignore [attr-defined] # pylint: disable=W0212
    ):
        logger.warning(
            "Specified pool object has a spawn context, which is not currently supported. "
            "See https://github.com/kensho-technologies/pyctcdecode/issues/65."
            "\nFalling back to sequential decoding."
        )
        return None
    return pool


def _normalize_whitespace(text: str) -> str:
    """Efficiently normalize whitespace."""
    return " ".join(text.split())


def _sort_and_trim_beams(beams: List[LMBeam], beam_width: int) -> List[LMBeam]:
    """Take top N beams by score."""
    return heapq.nlargest(beam_width, beams, key=lambda x: x.lm_score)


def _sum_log_scores(s1: float, s2: float) -> float:
    """Sum log odds in a numerically stable way."""
    # this is slightly faster than using max
    if s1 >= s2:
        log_sum = s1 + math.log(1 + math.exp(s2 - s1))
    else:
        log_sum = s2 + math.log(1 + math.exp(s1 - s2))
    return log_sum


def _log_softmax(
    x: np.ndarray[Shape, np.dtype[FloatVar]],
    axis: Optional[int] = None,
) -> np.ndarray[Shape, np.dtype[FloatVar]]:
    """Logarithm of softmax function, following implementation of scipy.special."""
    x_max = np.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0  # pylint: disable=R0204
    tmp = x - x_max
    exp_tmp: np.ndarray[Shape, np.dtype[FloatVar]] = np.exp(tmp)
    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out: np.ndarray[Shape, np.dtype[FloatVar]] = np.log(s)
    out = tmp - out
    return out


def _merge_tokens(token_1: str, token_2: str) -> str:
    """Fast, whitespace safe merging of tokens."""
    if len(token_2) == 0:
        text = token_1
    elif len(token_1) == 0:
        text = token_2
    else:
        text = token_1 + " " + token_2
    return text


def _merge_beams(beams: List[Beam]) -> List[Beam]:
    """Merge beams with same prefix together."""
    beam_dict = {}
    for beam in beams:
        new_text = _merge_tokens(beam.text, beam.next_word)
        hash_idx = (new_text, beam.partial_word, beam.last_char)
        if hash_idx not in beam_dict:
            beam_dict[hash_idx] = beam
        else:
            # We've already seen this text - we want to combine the scores
            beam_dict[hash_idx] = dataclasses.replace(
                beam, logit_score=_sum_log_scores(beam_dict[hash_idx].logit_score, beam.logit_score)
            )
    return list(beam_dict.values())


def _prune_history(beams: List[LMBeam], lm_order: int) -> List[Beam]:
    """Filter out beams that are the same over max_ngram history.

    Since n-gram language models have a finite history when scoring a new token, we can use that
    fact to prune beams that only differ early on (more than n tokens in the past) and keep only the
    higher scoring ones. Note that this helps speed up the decoding process but comes at the cost of
    some amount of beam diversity. If more than the top beam is used in the output it should
    potentially be disabled.

    Args:
        beams: list of LMBeam
        lm_order: int, the order of the n-gram model

    Returns:
        list of Beam
    """
    # let's keep at least 1 word of history
    min_n_history = max(1, lm_order - 1)
    seen_hashes = set()
    filtered_beams = []
    # for each beam after this, check if we need to add it
    for lm_beam in beams:
        # hash based on history that can still affect lm scoring going forward
        hash_idx = (
            tuple(lm_beam.text.split()[-min_n_history:]),
            lm_beam.partial_word,
            lm_beam.last_char,
        )
        if hash_idx not in seen_hashes:
            filtered_beams.append(Beam.from_lm_beam(lm_beam))
            seen_hashes.add(hash_idx)
    return filtered_beams


class BeamSearchDecoderCTC:
    # Note that we store the language model (large object) as a class variable.
    # The advantage of this is that during multiprocessing they won't cause and overhead in time.
    # This somewhat breaks conventional garbage collection which is why there are
    # specific functions for cleaning up the class variables manually if space needs to be freed up.
    # Specifically we create a random dictionary key during object instantiation which becomes the
    # storage key for the class variable model_container. This allows for multiple model instances
    # to be loaded at the same time.
    model_container: Dict[bytes, Optional[AbstractLanguageModel]] = {}

    # serialization filenames
    _ALPHABET_SERIALIZED_FILENAME = "alphabet.json"
    _LANGUAGE_MODEL_SERIALIZED_DIRECTORY = "language_model"

    def __init__(
        self,
        alphabet: Alphabet,
        language_model: Optional[AbstractLanguageModel] = None,
    ) -> None:
        """CTC beam search decoder for token logit matrix.

        Args:
            alphabet: class containing the labels for input logit matrices
            language_model: convenience class to store language model functionality
        """
        self._alphabet = alphabet
        self._idx2vocab = {n: c for n, c in enumerate(self._alphabet.labels)}
        self._is_bpe = alphabet.is_bpe
        self._model_key = os.urandom(16)
        BeamSearchDecoderCTC.model_container[self._model_key] = language_model

    def reset_params(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        unk_score_offset: Optional[float] = None,
        lm_score_boundary: Optional[bool] = None,
    ) -> None:
        """Reset parameters that don't require re-instantiating the model."""
        # todo: make more generic to accomodate other language models
        language_model = self._language_model
        if language_model is None:
            return
        params: Dict[str, Any] = {}
        if alpha is not None:
            params["alpha"] = alpha
        if beta is not None:
            params["beta"] = beta
        if unk_score_offset is not None:
            params["unk_score_offset"] = unk_score_offset
        if lm_score_boundary is not None:
            params["score_boundary"] = lm_score_boundary
        language_model.reset_params(**params)

    @classmethod
    def clear_class_models(cls) -> None:
        """Clear all models from class variable."""
        cls.model_container = {}

    def cleanup(self) -> None:
        """Manual cleanup of models in class variable."""
        if self._model_key in BeamSearchDecoderCTC.model_container:
            del BeamSearchDecoderCTC.model_container[self._model_key]

    @property
    def _language_model(self) -> Optional[AbstractLanguageModel]:
        """Retrieve the language model."""
        return BeamSearchDecoderCTC.model_container[self._model_key]

    def _check_logits_dimension(
        self,
        logits: NDArray[NpFloat],
    ) -> None:
        """Verify correct shape and dimensions for input logits."""
        if len(logits.shape) != 2:
            raise ValueError(
                "Input logits have %s dimensions, but need 2: (time, vocabulary)"
                % len(logits.shape)
            )
        if logits.shape[-1] != len(self._idx2vocab):
            raise ValueError(
                "Input logits shape is %s, but vocabulary is size %s. "
                "Need logits of shape: (time, vocabulary)" % (logits.shape, len(self._idx2vocab))
            )

    def _get_lm_beams(
        self,
        beams: List[Beam],
        hotword_scorer: HotwordScorer,
        cached_lm_scores: Dict[LMScoreCacheKey, LMScoreCacheValue],
        cached_partial_token_scores: Dict[str, float],
        is_eos: bool = False,
    ) -> List[LMBeam]:
        """Update score by averaging logit_score and lm_score."""
        # get language model and see if exists
        language_model = self._language_model
        # if no language model available then return raw score + hotwords as lm score
        if language_model is None:
            new_beams = []
            for beam in beams:
                new_text = _merge_tokens(beam.text, beam.next_word)
                # note that usually this gets scaled with alpha
                lm_hw_score = (
                    beam.logit_score
                    + hotword_scorer.score(new_text)
                    + hotword_scorer.score_partial_token(beam.partial_word)
                )

                new_beams.append(
                    LMBeam(
                        text=new_text,
                        next_word="",
                        partial_word=beam.partial_word,
                        last_char=beam.last_char,
                        text_frames=beam.text_frames,
                        partial_frames=beam.partial_frames,
                        logit_score=beam.logit_score,
                        lm_score=lm_hw_score,
                    )
                )
            return new_beams

        new_beams = []
        for beam in beams:
            # fast token merge
            new_text = _merge_tokens(beam.text, beam.next_word)
            cache_key = (new_text, is_eos)
            if cache_key not in cached_lm_scores:
                _, prev_raw_lm_score, start_state = cached_lm_scores[(beam.text, False)]
                score, end_state = language_model.score(
                    start_state, beam.next_word, is_last_word=is_eos
                )
                raw_lm_score = prev_raw_lm_score + score
                lm_hw_score = raw_lm_score + hotword_scorer.score(new_text)
                cached_lm_scores[cache_key] = (lm_hw_score, raw_lm_score, end_state)
            lm_score, _, _ = cached_lm_scores[cache_key]
            word_part = beam.partial_word
            if len(word_part) > 0:
                if word_part not in cached_partial_token_scores:
                    # if prefix available in hotword trie use that, otherwise default to char trie
                    if word_part in hotword_scorer:
                        cached_partial_token_scores[word_part] = hotword_scorer.score_partial_token(
                            word_part
                        )
                    else:
                        cached_partial_token_scores[word_part] = language_model.score_partial_token(
                            word_part
                        )
                lm_score += cached_partial_token_scores[word_part]

            new_beams.append(
                LMBeam(
                    text=new_text,
                    next_word="",
                    partial_word=word_part,
                    last_char=beam.last_char,
                    text_frames=beam.text_frames,
                    partial_frames=beam.partial_frames,
                    logit_score=beam.logit_score,
                    lm_score=beam.logit_score + lm_score,
                )
            )

        return new_beams

    def _partial_decode_logits(
        self,
        logits: NDArray[NpFloat],
        beams: List[Beam],
        beam_width: int,
        beam_prune_logp: float,
        token_min_logp: float,
        prune_history: bool,
        hotword_scorer: HotwordScorer,
        cached_lm_scores: LMScoreCache,
        cached_p_lm_scores: Dict[str, float],
        processed_frames: int = 0,
    ) -> List[Beam]:
        """Decode logits for a set of beams with warmed score caches."""
        language_model = self._language_model
        # bpe we can also have trailing word boundaries ▁⁇▁ so we may need to remember breaks
        force_next_break = False
        for frame_idx, logit_col in enumerate(logits, start=processed_frames):
            max_idx = logit_col.argmax()
            idx_list = set(np.where(logit_col >= token_min_logp)[0]) | {max_idx}
            new_beams: List[Beam] = []
            for idx_char in idx_list:
                p_char = logit_col[idx_char]
                char = self._idx2vocab[idx_char]
                for beam in beams:
                    # if only blank token or same token
                    if char == "" or beam.last_char == char:
                        if char == "":
                            new_end_frame = beam.partial_frames[0]
                        else:
                            new_end_frame = frame_idx + 1
                        new_part_frames = (
                            beam.partial_frames
                            if char == ""
                            else (beam.partial_frames[0], new_end_frame)
                        )
                        new_beams.append(
                            Beam(
                                text=beam.text,
                                next_word=beam.next_word,
                                partial_word=beam.partial_word,
                                last_char=char,
                                text_frames=beam.text_frames,
                                partial_frames=new_part_frames,
                                logit_score=beam.logit_score + p_char,
                            )
                        )
                    # if bpe and leading space char
                    elif self._is_bpe and (char[:1] == BPE_TOKEN or force_next_break):
                        force_next_break = False
                        # some tokens are bounded on both sides like ▁⁇▁
                        clean_char = char
                        if char[:1] == BPE_TOKEN:
                            clean_char = clean_char[1:]
                        if char[-1:] == BPE_TOKEN:
                            clean_char = clean_char[:-1]
                            force_next_break = True
                        new_frame_list = (
                            beam.text_frames
                            if beam.partial_word == ""
                            else beam.text_frames + [beam.partial_frames]
                        )
                        new_beams.append(
                            Beam(
                                text=beam.text,
                                next_word=beam.partial_word,
                                partial_word=clean_char,
                                last_char=char,
                                text_frames=new_frame_list,
                                partial_frames=(frame_idx, frame_idx + 1),
                                logit_score=beam.logit_score + p_char,
                            )
                        )
                    # if not bpe and space char
                    elif not self._is_bpe and char == " ":
                        new_frame_list = (
                            beam.text_frames
                            if beam.partial_word == ""
                            else beam.text_frames + [beam.partial_frames]
                        )
                        new_beams.append(
                            Beam(
                                text=beam.text,
                                next_word=beam.partial_word,
                                partial_word="",
                                last_char=char,
                                text_frames=new_frame_list,
                                partial_frames=NULL_FRAMES,
                                logit_score=beam.logit_score + p_char,
                            )
                        )
                    # general update of continuing token without space
                    else:
                        new_part_frames = (
                            (frame_idx, frame_idx + 1)
                            if beam.partial_frames[0] < 0
                            else (beam.partial_frames[0], frame_idx + 1)
                        )
                        new_beams.append(
                            Beam(
                                beam.text,
                                beam.next_word,
                                beam.partial_word + char,
                                char,
                                beam.text_frames,
                                new_part_frames,
                                beam.logit_score + p_char,
                            )
                        )

            # lm scoring and beam pruning
            new_beams = _merge_beams(new_beams)
            scored_beams = self._get_lm_beams(
                new_beams,
                hotword_scorer,
                cached_lm_scores,
                cached_p_lm_scores,
            )
            # remove beam outliers
            max_score = max([b.lm_score for b in scored_beams])
            scored_beams = [b for b in scored_beams if b.lm_score >= max_score + beam_prune_logp]
            # beam pruning by taking highest N prefixes and then filtering down
            trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
            # prune history and remove lm score from beams
            if prune_history:
                lm_order = 1 if language_model is None else language_model.order
                beams = _prune_history(trimmed_beams, lm_order=lm_order)
            else:
                beams = [Beam.from_lm_beam(b) for b in trimmed_beams]

        return beams

    def _finalize_beams(
        self,
        beams: Sequence[Beam],
        beam_width: int,
        beam_prune_logp: float,
        hotword_scorer: HotwordScorer,
        cached_lm_scores: LMScoreCache,
        cached_p_lm_scores: Dict[str, float],
        force_next_word: bool = False,
        is_end: bool = False,
    ) -> List[LMBeam]:
        """Perform final language model scoring and sorting."""
        if force_next_word or is_end:
            new_beams = []
            for beam in beams:
                new_token_times = (
                    beam.text_frames
                    if beam.partial_word == ""
                    else beam.text_frames + [beam.partial_frames]
                )
                new_beams.append(
                    Beam(
                        text=beam.text,
                        next_word=beam.partial_word,
                        partial_word="",
                        last_char=None,
                        text_frames=new_token_times,
                        partial_frames=(-1, -1),
                        logit_score=beam.logit_score,
                    ),
                )
            new_beams = _merge_beams(new_beams)
        else:
            new_beams = list(beams)
        scored_beams = self._get_lm_beams(
            new_beams,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
            is_eos=is_end,
        )
        # remove beam outliers
        max_score = max([b.lm_score for b in scored_beams])
        scored_beams = [b for b in scored_beams if b.lm_score >= max_score + beam_prune_logp]
        return _sort_and_trim_beams(scored_beams, beam_width)

    def _decode_logits(
        self,
        logits: NDArray[NpFloat],
        beam_width: int,
        beam_prune_logp: float,
        token_min_logp: float,
        prune_history: bool,
        hotword_scorer: HotwordScorer,
        lm_start_state: Optional[AbstractLMState] = None,
    ) -> List[OutputBeam]:
        """Perform beam search decoding."""
        # local dictionaries to cache scores during decoding
        # we can pass in an input start state to keep the decoder stateful and working on realtime
        language_model = self._language_model
        if language_model is None:
            cached_lm_scores: LMScoreCache = {}
        else:
            if lm_start_state is None:
                start_state = language_model.get_start_state()
            else:
                start_state = lm_start_state
            cached_lm_scores = {("", False): (0.0, 0.0, start_state)}
        cached_p_lm_scores: Dict[str, float] = {}
        # start with single beam to expand on
        beams = [EMPTY_START_BEAM]

        beams = self._partial_decode_logits(
            logits,
            beams,
            beam_width,
            beam_prune_logp,
            token_min_logp,
            prune_history,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
        )
        trimmed_beams = self._finalize_beams(
            beams,
            beam_width,
            beam_prune_logp,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
            force_next_word=True,
            is_end=True,
        )

        # remove unnecessary information from beams
        output_beams = [
            OutputBeam(
                text=_normalize_whitespace(lm_beam.text),
                last_lm_state=(
                    cached_lm_scores[(lm_beam.text, True)][-1]
                    if (lm_beam.text, True) in cached_lm_scores
                    else None
                ),
                text_frames=list(zip(lm_beam.text.split(), lm_beam.text_frames)),
                logit_score=lm_beam.logit_score,
                lm_score=lm_beam.lm_score,  # same as logit_score if lm is missing
            )
            for lm_beam in trimmed_beams
        ]
        return output_beams

    def get_starting_state(self) -> Tuple[List[Beam], LMScoreCache, Dict[str, float]]:
        """Get the starting beams and initial caches."""
        start_beam = [EMPTY_START_BEAM]
        language_model = self._language_model
        if language_model is None:
            cached_lm_scores: LMScoreCache = {}
        else:
            start_state = language_model.get_start_state()
            cached_lm_scores = {("", False): (0.0, 0.0, start_state)}
        cached_p_lm_scores: Dict[str, float] = {}
        return start_beam, cached_lm_scores, cached_p_lm_scores

    def partial_decode_beams(
        self,
        logits: NDArray[NpFloat],
        cached_lm_scores: LMScoreCache,
        cached_p_lm_scores: Dict[str, float],
        beams: List[Beam],
        processed_frames: int,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        prune_history: bool = DEFAULT_PRUNE_BEAMS,
        hotword_scorer: Optional[HotwordScorer] = None,
        force_next_word: bool = False,
        is_end: bool = False,
    ) -> List[LMBeam]:
        """Decode beams for the given logits, allowing for additional decoding steps."""
        self._check_logits_dimension(logits)
        hotword_scorer = hotword_scorer or HotwordScorer.build_scorer([], weight=0.0)
        # make sure we have log probs as input
        if math.isclose(logits.sum(axis=1).mean(), 1):
            # input looks like probabilities, so take log
            logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
        else:
            # convert logits into log probs
            logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)
        beams = self._partial_decode_logits(
            logits,
            beams,
            beam_width,
            beam_prune_logp,
            token_min_logp,
            prune_history,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
            processed_frames=processed_frames,
        )
        trimmed_beams = self._finalize_beams(
            beams,
            beam_width,
            beam_prune_logp,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
            force_next_word=force_next_word,
            is_end=is_end,
        )
        return trimmed_beams

    def decode_beams(
        self,
        logits: NDArray[NpFloat],
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        prune_history: bool = DEFAULT_PRUNE_BEAMS,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
        lm_start_state: Optional[AbstractLMState] = None,
    ) -> List[OutputBeam]:
        """Convert input token logit matrix to decoded beams including meta information.

        Args:
            logits: logit matrix of token log probabilities
            beam_width: maximum number of beams at each step in decoding
            beam_prune_logp: beams that are much worse than best beam will be pruned
            token_min_logp: tokens below this logp are skipped unless they are argmax of frame
            prune_history: prune beams based on shared recent history at the cost of beam diversity
            hotwords: list of words with extra importance, can be OOV for LM
            hotword_weight: weight factor for hotword importance
            lm_start_state: language model start state for stateful predictions

        Returns:
            List of beams of type OutputBeam with various meta information
        """
        self._check_logits_dimension(logits)
        # prepare hotword input
        hotword_scorer = HotwordScorer.build_scorer(hotwords, weight=hotword_weight)
        # make sure we have log probs as input
        if math.isclose(logits.sum(axis=1).mean(), 1):
            # input looks like probabilities, so take log
            logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
        else:
            # convert logits into log probs
            logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)
        decoded_beams = self._decode_logits(
            logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            prune_history=prune_history,
            hotword_scorer=hotword_scorer,
            lm_start_state=lm_start_state,
        )
        return decoded_beams

    def _decode_beams_mp_safe(
        self,
        logits: NDArray[NpFloat],
        beam_width: int,
        beam_prune_logp: float,
        token_min_logp: float,
        prune_history: bool,
        hotwords: Optional[Iterable[str]],
        hotword_weight: float,
    ) -> List[OutputBeam]:
        """Thing wrapper around self.decode_beams to allow for multiprocessing."""
        decoded_beams = self.decode_beams(
            logits=logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            prune_history=prune_history,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )
        # remove state to allow multiprocessing
        decoded_beams_mp_safe = [output_beam.get_mp_safe_beam() for output_beam in decoded_beams]
        return decoded_beams_mp_safe

    def decode_beams_batch(
        self,
        pool: Optional[Pool],
        logits_list: NDArray[NpFloat],
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        prune_history: bool = DEFAULT_PRUNE_BEAMS,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
    ) -> List[List[OutputBeam]]:
        """Use multiprocessing pool to batch decode input logits.

        Note that multiprocessing here does not work for a spawn context, so in that case
        or with no pool, just runs a loop in a single process.

        Args:
            pool: multiprocessing pool for parallel execution
            logits_list: list of logit matrices of token log probabilities
            beam_width: maximum number of beams at each step in decoding
            beam_prune_logp: beams that are much worse than best beam will be pruned
            token_min_logp: tokens below this logp are skipped unless they are argmax of frame
            prune_history: prune beams based on shared recent history at the cost of beam diversity
            hotwords: list of words with extra importance, can be OOV for LM
            hotword_weight: weight factor for hotword importance

        Returns:
            List of list of multiprocessing-safe OutputBeams with various meta information
        """
        valid_pool = _get_valid_pool(pool)
        if valid_pool is None:
            return [
                self._decode_beams_mp_safe(
                    logits,
                    beam_width=beam_width,
                    beam_prune_logp=beam_prune_logp,
                    token_min_logp=token_min_logp,
                    hotwords=hotwords,
                    prune_history=prune_history,
                    hotword_weight=hotword_weight,
                )
                for logits in logits_list
            ]

        for logits in logits_list:
            self._check_logits_dimension(logits)
        p_decode = functools.partial(
            self._decode_beams_mp_safe,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            hotwords=hotwords,
            prune_history=prune_history,
            hotword_weight=hotword_weight,
        )
        decoded_beams_list: List[List[OutputBeam]] = valid_pool.map(p_decode, logits_list)
        return decoded_beams_list

    def decode(
        self,
        logits: NDArray[NpFloat],
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
        lm_start_state: Optional[AbstractLMState] = None,
    ) -> str:
        """Convert input token logit matrix to decoded text.

        Args:
            logits: logit matrix of token log probabilities
            beam_width: maximum number of beams at each step in decoding
            beam_prune_logp: beams that are much worse than best beam will be pruned
            token_min_logp: tokens below this logp are skipped unless they are argmax of frame
            hotwords: list of words with extra importance, can be OOV for LM
            hotword_weight: weight factor for hotword importance
            lm_start_state: language model start state for stateful predictions

        Returns:
            The decoded text (str)
        """
        decoded_beams = self.decode_beams(
            logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            prune_history=True,  # we can set this to True since we only care about top 1 beam
            hotwords=hotwords,
            hotword_weight=hotword_weight,
            lm_start_state=lm_start_state,
        )
        return decoded_beams[0].text

    def decode_batch(
        self,
        pool: Optional[Pool],
        logits_list: NDArray[NpFloat],
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
    ) -> List[str]:
        """Use multiprocessing pool to batch decode input logits.

        Note that multiprocessing here does not work for a spawn context, so in that case
        or with no pool, just runs a loop in a single process.

        Args:
            pool: multiprocessing pool for parallel execution
            logits_list: list of logit matrices of token log probabilities
            beam_width: maximum number of beams at each step in decoding
            beam_prune_logp: beams that are much worse than best beam will be pruned
            token_min_logp: tokens below this logp are skipped unless they are argmax of frame
            hotwords: list of words with extra importance, can be OOV for LM
            hotword_weight: weight factor for hotword importance

        Returns:
            The decoded texts (list of str)
        """
        valid_pool = _get_valid_pool(pool)
        if valid_pool is None:
            return [
                self.decode(
                    logits,
                    beam_width=beam_width,
                    beam_prune_logp=beam_prune_logp,
                    token_min_logp=token_min_logp,
                    hotwords=hotwords,
                    hotword_weight=hotword_weight,
                )
                for logits in logits_list
            ]

        p_decode = functools.partial(
            self.decode,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )
        decoded_text_list: List[str] = valid_pool.map(p_decode, logits_list)
        return decoded_text_list

    def save_to_dir(self, filepath: str) -> None:
        """Save a decoder to a directory."""
        alphabet_path = os.path.join(filepath, self._ALPHABET_SERIALIZED_FILENAME)
        with open(alphabet_path, "w") as fi:
            fi.write(self._alphabet.dumps())

        lm = self._language_model
        if lm is None:
            logger.info("decoder has no language model.")
        else:
            lm_path = os.path.join(filepath, self._LANGUAGE_MODEL_SERIALIZED_DIRECTORY)
            os.makedirs(lm_path)
            logger.info("Saving language model to %s", lm_path)
            lm.save_to_dir(lm_path)

    @staticmethod
    def parse_directory_contents(filepath: str) -> Dict[str, Union[str, None]]:
        """Check contents of a directory for correct BeamSearchDecoderCTC files."""
        contents = os.listdir(filepath)
        # filter out hidden files
        contents = [c for c in contents if not c.startswith(".") and not c.startswith("__")]
        if BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME not in contents:
            raise ValueError(
                f"Could not find alphabet file "
                f"{BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME}. Found {contents}"
            )
        alphabet_filepath = os.path.join(
            filepath, BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
        )
        contents.remove(BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME)
        lm_directory: Optional[str]
        if contents:
            if BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY not in contents:
                raise ValueError(
                    f"Count not find language model directory. Looking for "
                    f"{BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY}, found {contents}"
                )
            lm_directory = os.path.join(
                filepath, BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY
            )
        else:
            lm_directory = None
        return {"alphabet": alphabet_filepath, "language_model": lm_directory}

    @classmethod
    def load_from_dir(
        cls, filepath: str, unigram_encoding: Optional[str] = None
    ) -> "BeamSearchDecoderCTC":
        """Load a decoder from a directory."""
        filenames = cls.parse_directory_contents(filepath)
        with open(filenames["alphabet"], "r") as fi:  # type: ignore
            alphabet = Alphabet.loads(fi.read())
        if filenames["language_model"] is None:
            language_model = None
        else:
            language_model = LanguageModel.load_from_dir(
                filenames["language_model"], unigram_encoding=unigram_encoding
            )
        return cls(alphabet, language_model=language_model)

    @classmethod
    def load_from_hf_hub(  # type: ignore
        cls, model_id: str, cache_dir: Optional[str] = None, **kwargs: Any
    ) -> "BeamSearchDecoderCTC":
        """Class method to load model from https://huggingface.co/ .

        Args:
            model_id: string, the `model id` of a pretrained model hosted inside a model
                repo on https://huggingface.co/. Valid model ids can be namespaced under a user or
                organization name, like ``kensho/5gram-spanish-kenLM``. For more information, please
                take a look at https://huggingface.co/docs/hub/main.
            cache_dir: path to where the language model should be downloaded and cached.

        Returns:
            instance of BeamSearchDecoderCTC
        """
        if sys.version_info >= (3, 8):
            from importlib.metadata import metadata
        else:
            from importlib_metadata import metadata

        library_name = metadata("pyctcdecode")["Name"]
        cache_dir = cache_dir or os.path.join(Path.home(), ".cache", library_name)

        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError:
            raise ImportError(
                "You need to install huggingface_hub to use `load_from_hf_hub`. "
                "See https://pypi.org/project/huggingface-hub/ for installation."
            )

        cached_directory = snapshot_download(  # pylint: disable=not-callable
            model_id, cache_dir=cache_dir, **kwargs
        )

        return cls.load_from_dir(cached_directory)


##########################################################################################
# Main entry point and convenience function to create BeamSearchDecoderCTC object ########
##########################################################################################


def build_ctcdecoder(
    labels: List[str],
    kenlm_model_path: Optional[str] = None,
    unigrams: Optional[Collection[str]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    unk_score_offset: float = DEFAULT_UNK_LOGP_OFFSET,
    lm_score_boundary: bool = DEFAULT_SCORE_LM_BOUNDARY,
) -> BeamSearchDecoderCTC:
    """Build a BeamSearchDecoderCTC instance with main functionality.

    Args:
        labels: class containing the labels for input logit matrices
        kenlm_model_path: path to kenlm n-gram language model
        unigrams: list of known word unigrams
        alpha: weight for language model during shallow fusion
        beta: weight for length score adjustment of during scoring
        unk_score_offset: amount of log score offset for unknown tokens
        lm_score_boundary: whether to have kenlm respect boundaries when scoring

    Returns:
        instance of BeamSearchDecoderCTC
    """
    kenlm_model = None if kenlm_model_path is None else kenlm.Model(kenlm_model_path)
    if kenlm_model_path is not None and kenlm_model_path.endswith(".arpa"):
        logger.info("Using arpa instead of binary LM file, decoder instantiation might be slow.")
    if unigrams is None and kenlm_model_path is not None:
        if kenlm_model_path.endswith(".arpa"):
            unigrams = load_unigram_set_from_arpa(kenlm_model_path)
        else:
            logger.warning(
                "Unigrams not provided and cannot be automatically determined from LM file (only "
                "arpa format). Decoding accuracy might be reduced."
            )
    alphabet = Alphabet.build_alphabet(labels)
    if unigrams is not None:
        verify_alphabet_coverage(alphabet, unigrams)
    if kenlm_model is not None:
        language_model: Optional[AbstractLanguageModel] = LanguageModel(
            kenlm_model,
            unigrams,
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            score_boundary=lm_score_boundary,
        )
    else:
        language_model = None
    return BeamSearchDecoderCTC(alphabet, language_model)
