"""
Tests for vocabulary analysis and the exclusivity property of the controlled
reference corpus. These pin the design claim: TF-IDF top-N is near-fully
mutually exclusive across categories, and meaningfully more exclusive than
the frequency-only N-gram top-N.
"""

from sentineldesk.data_gen.controlled import generate_controlled_corpus
from sentineldesk.vocabulary.analysis import (
    exclusivity_report,
    ngram_counts_by_category,
    tfidf_weights_by_category,
)


def test_ngram_counts_shape_and_abundance():
    corpus = generate_controlled_corpus(per_category=60)
    ng = ngram_counts_by_category(corpus, top_n=20)
    assert len(ng) == 7
    for terms in ng.values():
        assert terms and all(isinstance(t, str) and isinstance(c, int) for t, c in terms)
        # "abundant": the top term recurs across many tickets
        assert terms[0][1] > 5


def test_tfidf_weights_shape():
    corpus = generate_controlled_corpus(per_category=60)
    tf = tfidf_weights_by_category(corpus, top_n=20)
    assert len(tf) == 7
    for terms in tf.values():
        assert terms and all(isinstance(t, str) and isinstance(w, float) for t, w in terms)


def test_tfidf_more_exclusive_than_ngram():
    corpus = generate_controlled_corpus(per_category=120)
    ng = ngram_counts_by_category(corpus, top_n=20)
    tf = tfidf_weights_by_category(corpus, top_n=20)
    ex_ng = exclusivity_report(ng)["_overall"]
    ex_tf = exclusivity_report(tf)["_overall"]
    # TF-IDF strips shared filler -> strictly more exclusive, and near-perfect
    assert ex_tf >= ex_ng
    assert ex_tf >= 0.90
    assert ex_ng >= 0.65
