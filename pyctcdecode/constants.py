import math


# default parameters for decoding (can be modified)
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 1.5
DEFAULT_UNK_LOGP_OFFSET = -10.0
DEFAULT_BEAM_WIDTH = 100
DEFAULT_HOTWORD_WEIGHT = 10.0
DEFAULT_PRUNE_LOGP = -10.0
DEFAULT_PRUNE_BEAMS = False
DEFAULT_MIN_TOKEN_LOGP = -5.0
DEFAULT_SCORE_LM_BOUNDARY = True

# other constants for decoding
AVG_TOKEN_LEN = 6  # average number of characters expected per token (used for UNK scoring)
MIN_TOKEN_CLIP_P = 1e-15  # clipping to avoid underflow in case of malformed logit input
LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)  # kenlm returns base10 but we like natural
