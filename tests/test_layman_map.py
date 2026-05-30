"""The layman map must cover every signature-lexicon term, or the enrichment
seed and the scorer's recall dictionary silently drift out of sync."""

from sentineldesk.data_gen.controlled import SIGNATURE_LEXICON
from sentineldesk.data_gen.layman_map import LAYMAN_MAP


def test_layman_map_covers_signature_lexicon():
    missing = []
    for category, terms in SIGNATURE_LEXICON.items():
        for term in terms:
            if term not in LAYMAN_MAP.get(category, {}):
                missing.append((category, term))
    assert not missing, f"layman map missing entries for: {missing}"


def test_every_term_has_phrasings():
    for category, terms in LAYMAN_MAP.items():
        for term, phrasings in terms.items():
            assert phrasings, f"{category}/{term} has no layman phrasings"
            assert all(isinstance(p, str) and p for p in phrasings)
