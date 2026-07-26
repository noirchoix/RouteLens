from __future__ import annotations

from sklearn.feature_extraction.text import HashingVectorizer


def reaction_text_vectorizer(n_features: int = 2**18) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype="float32",
    )
