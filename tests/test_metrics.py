import pytest

from htr.metrics import cer, levenshtein, wer


def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("flaw", "lawn") == 2


def test_cer_perfect_and_total():
    assert cer(["hello"], ["hello"]) == 0.0
    assert cer(["ab"], ["cd"]) == 1.0


def test_cer_partial():
    # 1 substitution over 5 ref chars
    assert cer(["hello"], ["hallo"]) == pytest.approx(0.2)


def test_cer_corpus_weighting():
    # total edits 1, total ref chars 6
    assert cer(["hello", "a"], ["hello", "b"]) == pytest.approx(1 / 6)


def test_wer_single_words():
    assert wer(["cat", "dog"], ["cat", "dot"]) == pytest.approx(0.5)


def test_wer_multiword():
    assert wer(["the quick fox"], ["the slow fox"]) == pytest.approx(1 / 3)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cer(["a"], [])
    with pytest.raises(ValueError):
        wer(["a"], [])
