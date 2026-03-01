"""Tokenization and lemmatization module.

Uses the shared NLP processor from nlp_processor.py.
"""

from src.pipeline.nlp_processor import analyze_speech, load_model


def tokenize(text: str) -> list[dict]:
    """Tokenize text and return token info including lemmas."""
    nlp = load_model()
    doc = nlp(text)
    return [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "is_stop": token.is_stop,
            "is_punct": token.is_punct,
        }
        for token in doc
    ]
