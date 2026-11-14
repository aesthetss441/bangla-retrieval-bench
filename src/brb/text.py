"""Tokenizers for BM25.

Bangla vowel signs and the hasanta are combining marks (Unicode categories Mc/Mn),
not letters. Python's ``\\w`` does not match them, so the tokenizer most tutorials
use, ``re.findall(r"\\w+", text)``, cuts a Bangla word at every vowel sign:
"বাংলাদেশ" becomes ["ব", "ল", "দ", "শ"]. ``bangla`` keeps the whole Bengali block
together instead.
"""
from __future__ import annotations

import re
import unicodedata

_NAIVE = re.compile(r"\w+")
# Bengali block plus zero-width (non-)joiners used inside conjuncts; the danda (U+0964) is outside the block.
_BANGLA = re.compile(r"[ঀ-৿‌‍]+|[A-Za-z0-9]+")


def naive(text: str) -> list[str]:
    return _NAIVE.findall(text.lower())


def bangla(text: str) -> list[str]:
    return _BANGLA.findall(unicodedata.normalize("NFC", text).lower())


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Character n-grams inside word boundaries. Bangla is heavily inflected
    (দেশ, দেশের, দেশে, দেশগুলো), and n-grams let different forms of a word share terms
    without a stemmer."""
    grams: list[str] = []
    for token in bangla(text):
        padded = f"<{token}>"
        grams.extend(padded[i : i + n] for i in range(max(1, len(padded) - n + 1)))
    return grams


TOKENIZERS = {"naive": naive, "bangla": bangla, "char3": char_ngrams}
