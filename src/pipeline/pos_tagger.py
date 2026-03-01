"""POS tagging module.

Uses the shared NLP processor from nlp_processor.py.
"""

from src.pipeline.nlp_processor import load_model


def pos_tag(text: str) -> list[tuple[str, str]]:
    """Return (token, POS) pairs for the text."""
    nlp = load_model()
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]
