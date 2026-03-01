"""Named Entity Recognition module.

Uses BETO NER (mrm8488/bert-spanish-cased-finetuned-ner) via the shared
NLP processor's cached pipeline.
"""

from src.pipeline.nlp_processor import _get_ner_pipeline


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text using BETO NER."""
    ner = _get_ner_pipeline()
    results = ner(text)
    entities = []
    for r in results:
        entity_text = r["word"].strip().rstrip(".,;:!?\"')")
        if not entity_text:
            continue
        entities.append({
            "text": entity_text,
            "label": r["entity_group"],
            "start_char": r["start"],
            "end_char": r["end"],
        })
    return entities
