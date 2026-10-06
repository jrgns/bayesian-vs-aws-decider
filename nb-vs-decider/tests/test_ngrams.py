from scripts.ngrams import NGRAM_N, NgramIndex, dedup_key, tokens

WORDS = [f"w{i}" for i in range(40)]


def test_tokens_lowercase_and_strip_punctuation():
    assert tokens("Hello, World!  It's 5-4_3.") == ["hello", "world", "it", "s", "5", "4", "3"]


def test_dedup_key_collapses_whitespace_and_case():
    assert dedup_key("  Foo\n\tBAR  baz ") == "foo bar baz"


def test_planted_13gram_found_and_12gram_ignored():
    wos = tokens(" ".join(WORDS[:30]))
    index = NgramIndex.build([tokens("completely different words here " * 5), wos])
    planted = ["x", "y"] + WORDS[5 : 5 + NGRAM_N] + ["z"]
    hits = index.lookup(planted)
    assert hits == [(2, 1)]
    assert index.lookup(["x"] + WORDS[5 : 5 + NGRAM_N - 1] + ["z"]) == []


def test_ngram_order_matters():
    index = NgramIndex.build([WORDS[:NGRAM_N]])
    assert index.lookup(list(reversed(WORDS[:NGRAM_N]))) == []


def test_short_text_has_no_ngrams():
    assert NgramIndex.build([WORDS[: NGRAM_N - 1]]).keys.size == 0


def test_exact_lookup_by_hash():
    from scripts.ngrams import hash64

    index = NgramIndex.from_hashes([hash64("a b"), hash64("c d")])
    assert index.items_for(hash64("c d")) == [1]
    assert index.items_for(hash64("zzz")) == []
