# Copyright 2021-present Kensho Technologies, LLC.
import logging
from typing import List, Optional


BPE_CHAR = "▁"  # character used for token boundary if BPE is used
UNK_BPE_CHAR = "▁⁇▁"  # representation of unknown character in BPE

logger = logging.getLogger(__name__)


def _get_ctc_index(label_list: List[str]) -> int:
    """Get index of ctc blank character in alphabet."""
    return len(label_list) - 1 if label_list[-1] == "" else -1


def _normalize_alphabet(label_list: List[str], ctc_token_idx: Optional[int] = None) -> List[str]:
    """Normalize alphabet for non-bpe decoder."""
    if any([len(c) > 1 for c in label_list]):
        raise ValueError("For non-bpe alphabet only length 1 entries and blank token are allowed.")
    if ctc_token_idx is None:
        ctc_token_idx = _get_ctc_index(label_list)
    clean_labels = label_list[:]
    # check for space token
    if " " not in clean_labels:
        raise ValueError("Space token ' ' missing from vocabulary.")
    # specify ctc blank token
    if ctc_token_idx == -1:
        clean_labels.append("")
    else:
        clean_labels[ctc_token_idx] = ""
    return clean_labels


def _convert_bpe_format(token: str) -> str:
    """Convert token from ## type bpe format to ▁ type."""
    if token[:2] == "##":
        return token[2:]
    elif token == BPE_CHAR:
        return token
    elif token == "":  # nosec
        return token
    else:
        return BPE_CHAR + token


def _normalize_bpe_alphabet(
    label_list: List[str],
    unk_token_idx: Optional[int] = None,
    ctc_token_idx: Optional[int] = None,
) -> List[str]:
    """Normalize alphabet for bpe decoder."""
    if ctc_token_idx is None:
        ctc_token_idx = _get_ctc_index(label_list)
    # create copy
    clean_labels = label_list[:]
    # there are two common formats for BPE vocabulary
    # 1) where ▁ indicates a space (this is the main format we use)
    if any([s[:1] == BPE_CHAR and len(s) > 1 for s in clean_labels]):
        # verify unk token and make sure it is consistently represented as ▁⁇▁
        if unk_token_idx is None and clean_labels[0] in ("<unk>", UNK_BPE_CHAR):
            unk_token_idx = 0
        else:
            raise ValueError(
                "First token in vocab for BPE should be '▁⁇▁' or specify unk_token_idx."
            )
        clean_labels[unk_token_idx] = UNK_BPE_CHAR
    # 2) where ## indicates continuation of a token (note: also contains the single token: ▁)
    elif any([s[:2] == "##" for s in clean_labels]):
        # convert to standard format 1)
        clean_labels = [_convert_bpe_format(c) for c in clean_labels]
        # add unk token if needed
        if clean_labels[0] in ("<unk>", UNK_BPE_CHAR):
            clean_labels[0] = UNK_BPE_CHAR
        else:
            clean_labels = [UNK_BPE_CHAR] + clean_labels
            ctc_token_idx += 1
    else:
        raise ValueError(
            "Unknown BPE format for vocabulary. Supported formats are 1) ▁ for indicating a"
            " space and 2) ## for continuation of a word."
        )
    # specify ctc blank token
    if ctc_token_idx == -1:
        clean_labels.append("")
    else:
        clean_labels[ctc_token_idx] = ""
    return clean_labels


class Alphabet:
    def __init__(self, labels: List[str], is_bpe: bool) -> None:
        """Init."""
        self._labels = labels
        self._is_bpe = is_bpe

    @property
    def is_bpe(self) -> bool:
        """Whether the alphabet is a bytepair encoded one."""
        return self._is_bpe

    @property
    def labels(self) -> List[str]:
        """Deep copy of the labels."""
        return self._labels[:]  # this is a copy

    @classmethod
    def build_alphabet(
        cls, label_list: List[str], ctc_token_idx: Optional[int] = None
    ) -> "Alphabet":
        """Make a non-BPE alphabet."""
        formatted_alphabet_list = _normalize_alphabet(label_list, ctc_token_idx)
        return cls(formatted_alphabet_list, False)

    @classmethod
    def build_bpe_alphabet(
        cls,
        label_list: List[str],
        unk_token_idx: Optional[int] = None,
        ctc_token_idx: Optional[int] = None,
    ) -> "Alphabet":
        """Make a BPE alphabet."""
        formatted_label_list = _normalize_bpe_alphabet(label_list, unk_token_idx, ctc_token_idx)
        return cls(formatted_label_list, True)
