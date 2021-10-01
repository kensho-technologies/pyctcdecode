# Copyright 2021-present Kensho Technologies, LLC.
from __future__ import division

import logging
import re
from typing import Collection, List


BPE_TOKEN = "▁"  # nosec # representation of token boundary in BPE alphabet
UNK_TOKEN = "⁇"  # nosec # representation of special UNK token in regular alphabet
UNK_BPE_TOKEN = "▁⁇▁"  # nosec # representation of special UNK token in BPE alphabet

# special tokens are usually encode with things like `[]` or `<>`
SPECIAL_TOKEN_PTN = re.compile(r"^[<\[].+[>\]]$")
BLANK_TOKEN_PTN = re.compile(r"^[<\[]pad[>\]]$", flags=re.IGNORECASE)
UNK_TOKEN_PTN = re.compile(r"^[<\[]unk[>\]]$", flags=re.IGNORECASE)

logger = logging.getLogger(__name__)


def _check_if_bpe(labels: List[str]) -> bool:
    """Check if input alphabet is BPE or not."""
    is_bpe = any([s.startswith("##") for s in labels]) or any(
        [s.startswith(BPE_TOKEN) for s in labels]
    )
    if is_bpe:
        logger.info("Alphabet determined to be of BPE style.")
    else:
        logger.info("Alphabet determined to be of regular style.")
    return is_bpe


def _normalize_regular_alphabet(labels: List[str]) -> List[str]:
    """Normalize non-bpe labels to alphabet for decoder."""
    normalized_labels = labels[:]
    # substitute space characters
    if "|" in normalized_labels and " " not in normalized_labels:
        logger.info("Found '|' in vocabulary but not ' ', doing substitution.")
        normalized_labels[normalized_labels.index("|")] = " "
    # substituted ctc blank char
    for n, label in enumerate(normalized_labels):
        if BLANK_TOKEN_PTN.match(label):
            logger.info(
                "Found %s in vocabulary, interpreted as a CTC blank token, substituting with %s.",
                label,
                "",
            )
            normalized_labels[n] = ""
    if "_" in normalized_labels and "" not in normalized_labels:
        logger.info("Found '_' in vocabulary but not '', doing substitution.")
        normalized_labels[normalized_labels.index("_")] = ""
    if "" not in normalized_labels:
        logger.info("CTC blank char '' not found, appending to end.")
        normalized_labels.append("")
    # substitute unk
    for n, label in enumerate(normalized_labels):
        if UNK_TOKEN_PTN.match(label):
            logger.info(
                "Found %s in vocabulary, interpreting as unknown token, substituting with %s.",
                label,
                UNK_TOKEN,
            )
            normalized_labels[n] = UNK_TOKEN
    # additional checks
    if any([len(c) > 1 for c in normalized_labels]):
        logger.warning(
            "Found entries of length > 1 in alphabet. This is unusual unless style is BPE, but the "
            "alphabet was not recognized as BPE type. Is this correct?"
        )
    if " " not in normalized_labels:
        logger.warning("Space token ' ' missing from vocabulary.")
    return normalized_labels


def _convert_bpe_token_style(token: str) -> str:
    """Convert token from ## style bpe format to ▁ style."""
    if token.startswith("##"):
        return token[2:]
    elif SPECIAL_TOKEN_PTN.match(token) or token in ("", BPE_TOKEN, UNK_BPE_TOKEN):
        return token
    elif token in ("<unk>", UNK_BPE_TOKEN):
        return token
    else:
        return BPE_TOKEN + token


def _normalize_bpe_alphabet(labels: List[str]) -> List[str]:
    """Normalize alphabet for bpe decoder."""
    normalized_labels = labels[:]
    # if BPE is of style '##' then convert it
    if any([s.startswith("##") for s in labels]):
        normalized_labels = [_convert_bpe_token_style(c) for c in normalized_labels]
    # substituted ctc blank char
    for n, label in enumerate(normalized_labels):
        if BLANK_TOKEN_PTN.match(label):
            logger.info("Found %s in vocabulary, substituting with %s.", label, "")
            normalized_labels[n] = ""
    if "" not in normalized_labels:
        logger.info("CTC blank char '' not found, appending to end.")
        normalized_labels.append("")
    # substitute unk
    for n, label in enumerate(normalized_labels):
        if UNK_TOKEN_PTN.match(label):
            logger.info("Found %s in vocabulary, substituting with %s.", label, UNK_BPE_TOKEN)
            normalized_labels[n] = UNK_BPE_TOKEN
    # additional checks
    if UNK_BPE_TOKEN not in normalized_labels:
        logger.warning("UNK token %s not found, is this a mistake?", UNK_BPE_TOKEN)
    return normalized_labels


def _verify_alphabet(labels: List[str], is_bpe: bool) -> None:
    """Verify basic alphabet labels."""
    # check if duplicates exist
    if len(labels) != len(set(labels)):
        raise ValueError("Alphabet contains duplicate entries, this is not allowed.")
    # check if space character is absent in bpe alphabet
    if is_bpe and any([" " in s for s in labels]):
        raise ValueError("Space token ' ' found in vocabulary even though it looks like BPE.")


class Alphabet:
    def __init__(self, labels: List[str], is_bpe: bool) -> None:
        """Init."""
        self._labels = labels
        self._is_bpe = is_bpe

    @property
    def is_bpe(self) -> bool:
        """Whether the alphabet is bpe style."""
        return self._is_bpe

    @property
    def labels(self) -> List[str]:
        """Deep copy of the labels."""
        return self._labels[:]  # this is a copy

    @classmethod
    def build_alphabet(cls, labels: List[str]) -> "Alphabet":
        """Make an alphabet from labels in standardized format for decoder."""
        is_bpe = _check_if_bpe(labels)
        _verify_alphabet(labels, is_bpe)
        if is_bpe:
            normalized_labels = _normalize_bpe_alphabet(labels)
        else:
            normalized_labels = _normalize_regular_alphabet(labels)
        return cls(normalized_labels, is_bpe)


def verify_alphabet_coverage(alphabet: Alphabet, unigrams: Collection[str]) -> None:
    """Verify if alphabet covers a given unigrams."""
    label_chars = set(alphabet.labels)
    unigram_sample_chars = set("".join(unigrams))
    if len(unigram_sample_chars - label_chars) / len(unigram_sample_chars) > 0.2:
        logger.warning("Unigrams and labels don't seem to agree.")
