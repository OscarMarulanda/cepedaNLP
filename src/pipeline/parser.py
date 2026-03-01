"""Dependency parsing module.

Uses the shared NLP processor from nlp_processor.py.
"""

from src.pipeline.nlp_processor import load_model


def parse_dependencies(text: str) -> list[dict]:
    """Parse dependency structure for each token."""
    nlp = load_model()
    doc = nlp(text)
    return [
        {
            "text": token.text,
            "dep": token.dep_,
            "head": token.head.text,
            "children": [child.text for child in token.children],
        }
        for token in doc
    ]
