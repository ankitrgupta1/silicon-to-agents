import pytest

from s2a.bpe import BPETokenizer

CORPUS = "the cat sat on the mat. the cat ate the rat. " * 20


@pytest.mark.parametrize("pretokenize", [True, False])
def test_round_trip(pretokenize):
    tok = BPETokenizer(pretokenize).train(CORPUS, vocab_size=300)
    for text in [CORPUS, "a brand new sentence!", "naïve café 東京 🙂", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_compresses_training_text():
    tok = BPETokenizer().train(CORPUS, vocab_size=300)
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode("utf-8")) / 3


def test_untrained_tokenizer_is_bytes():
    tok = BPETokenizer()
    assert tok.encode("hi") == [104, 105]
    assert tok.vocab_size == 256


def test_learns_frequent_words_as_single_tokens():
    tok = BPETokenizer().train(CORPUS, vocab_size=300)
    assert tok.tokens(" cat") == [" cat"]
    assert tok.tokens("the") == ["the"]


def test_pretokenize_keeps_merges_inside_words():
    tok = BPETokenizer(pretokenize=True).train(CORPUS, vocab_size=300)
    for piece in tok.vocab.values():
        text = piece.decode("utf-8", errors="replace")
        # A learned token never contains a space after its first character.
        assert " " not in text[1:]


def test_vocab_size_and_errors():
    tok = BPETokenizer().train(CORPUS, vocab_size=265)
    assert tok.vocab_size == 265
    with pytest.raises(ValueError):
        BPETokenizer().train(CORPUS, vocab_size=100)


def test_stops_when_nothing_left_to_merge():
    tok = BPETokenizer().train("ab", vocab_size=1000)
    assert tok.vocab_size == 257
