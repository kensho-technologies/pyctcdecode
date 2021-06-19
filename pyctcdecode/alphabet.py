# Copyright 2021-present Kensho Technologies, LLC.
import logging
from typing import Collection, List


UNK_CHAR = "⁇"  # representation of unknown character in regular alphabet
BPE_CHAR = "▁"  # character used for token boundary if BPE is used
UNK_BPE_CHAR = "▁⁇▁"  # representation of unknown character in BPE

logger = logging.getLogger(__name__)


def _check_if_bpe(labels: List[str]):
    """Check if input alphabet is BPE or not."""
    is_bpe = any([s.startswith("##") for s in labels]) or any(
        [s.startswith(BPE_CHAR) for s in labels]
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
    elif "|" in normalized_labels:
        logger.warning(
            "Found '|' in vocabulary. If this denotes something special, please construct the "
            "alphabet manually."
        )
    # substituted ctc blank char
    if "_" in normalized_labels and "" not in normalized_labels:
        logger.info("Found '_' in vocabulary but not '', doing substitution.")
        normalized_labels[normalized_labels.index("_")] = ""
    elif "_" in normalized_labels:
        logger.warning(
            "Found '_' in vocabulary. If this denotes something special, please construct the "
            "alphabet manually."
        )
    if "" not in normalized_labels:
        logger.info("CTC blank char '' not found, appending to end.")
        normalized_labels.append("")
    # substitute unk
    for n, label in enumerate(normalized_labels):
        if label.lower() in ("unk", "<unk>"):
            logger.info("Found %s in vocabulary, substituting with %s.", label, UNK_CHAR)
            normalized_labels[n] = UNK_CHAR
    # substitute other special characters
    if any([label[:1] == "<" and label[-1:] == ">" for label in normalized_labels]):
        logger.warning("Special characters found. Substituting with %s.", UNK_CHAR)
        for n, label in enumerate(normalized_labels):
            if label.startswith("<") and label.endswith(">"):
                normalized_labels[n] = UNK_CHAR
    # additional checks
    if any([len(c) > 1 for c in normalized_labels]):
        logger.warning(
            "Found entries of length > 1 in alphabet. This is unusual unless style is BPE. "
            "Is this correct?"
        )
    if " " not in normalized_labels:
        logger.warning("Space token ' ' missing from vocabulary.")
    return normalized_labels


def _convert_bpe_token_style(token: str) -> str:
    """Convert token from ## style bpe format to ▁ style."""
    if token.startswith("##"):
        return token[2:]
    elif token in ("", BPE_CHAR, UNK_BPE_CHAR):
        return token
    else:
        return BPE_CHAR + token


def _normalize_bpe_alphabet(labels: List[str]) -> List[str]:
    """Normalize alphabet for bpe decoder."""
    normalized_labels = labels[:]
    # substitute unk
    for n, label in enumerate(normalized_labels):
        if label.lower() in ("unk", "<unk>", "⁇"):
            logger.info("Found %s in vocabulary, substituting with %s.", label, UNK_BPE_CHAR)
            normalized_labels[n] = UNK_BPE_CHAR
    if UNK_BPE_CHAR not in normalized_labels:
        logger.info("UNK not found in labels, prepending to beginning.")
        normalized_labels = [UNK_BPE_CHAR] + normalized_labels
    # substituted ctc blank char
    if "" not in normalized_labels:
        logger.info("CTC blank char '' not found, appending to end.")
        normalized_labels.append("")
    # substitute other special characters
    if any(["<" in label and ">" in label for label in normalized_labels]):
        logger.warning("Special characters found. Substituting with %s.", UNK_BPE_CHAR)
        for n, label in enumerate(normalized_labels):
            if "<" in label and ">" in label:
                normalized_labels[n] = UNK_BPE_CHAR
    # if BPE is of style '##' then convert it
    if any([s.startswith("##") for s in labels]):
        normalized_labels = [_convert_bpe_token_style(c) for c in normalized_labels]
    return normalized_labels


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
        if is_bpe:
            normalized_labels = _normalize_bpe_alphabet(labels)
        else:
            normalized_labels = _normalize_regular_alphabet(labels)
        return cls(normalized_labels, is_bpe)


def verify_alphabet_coverage(alphabet: Alphabet, unigrams: Collection[str]) -> None:
    """Verify if alphabet covers a given unigrams."""
    label_chars = set(alphabet.labels)
    unigram_sample_chars = set("".join(unigrams[:100]))
    if len(unigram_sample_chars - label_chars) / len(unigram_sample_chars) > 0.2:
        logger.warning("Unigrams and labels don't seem to agree.")
