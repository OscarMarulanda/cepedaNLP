"""spaCy-based NLP pipeline for processed speech transcripts.

Combines tokenization, lemmatization, POS tagging, NER, and dependency parsing
into a single pipeline processor. Individual module files (tokenizer.py, etc.)
import from this shared processor.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import spacy
from spacy.tokens import Doc
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

BETO_NER_MODEL = "mrm8488/bert-spanish-cased-finetuned-ner"
GAZETTEER_PATH = Path("data/gazetteer/colombian_locations.txt")

# Entities that should be discarded entirely (noise from BETO).
_ENTITY_BLACKLIST: set[str] = {"Pela"}


def _load_gazetteer() -> set[str]:
    """Load Colombian locations gazetteer from file.

    The gazetteer contains ~1,100 locations from DANE DIVIPOLA
    (municipalities, departments) plus manual additions (countries,
    regions). Used for post-processing BETO NER to correct
    PER/ORG → LOC misclassifications.
    """
    locations: set[str] = set()
    if not GAZETTEER_PATH.exists():
        logger.warning("Gazetteer file not found: %s", GAZETTEER_PATH)
        return locations
    with open(GAZETTEER_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                locations.add(line)
    logger.info("Loaded %d locations from gazetteer", len(locations))
    return locations


_LOCATION_GAZETTEER: set[str] = _load_gazetteer()

# Global model caches
_nlp = None
_ner_pipeline = None


def load_model(model_name: str = "es_core_news_lg") -> spacy.Language:
    """Load spaCy model (cached after first call)."""
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model '%s'...", model_name)
        _nlp = spacy.load(model_name)
        logger.info("spaCy model loaded")
    return _nlp


def _get_ner_pipeline():
    """Load BETO NER pipeline (cached after first call)."""
    global _ner_pipeline
    if _ner_pipeline is None:
        logger.info("Loading BETO NER model '%s'...", BETO_NER_MODEL)
        _ner_pipeline = hf_pipeline(
            "ner",
            model=BETO_NER_MODEL,
            aggregation_strategy="first",
        )
        logger.info("BETO NER model loaded")
    return _ner_pipeline


@dataclass
class TokenInfo:
    """Token-level NLP information."""

    text: str
    lemma: str
    pos: str
    dep: str
    head_text: str
    is_stop: bool
    is_punct: bool


@dataclass
class EntityInfo:
    """Named entity information."""

    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class SentenceAnalysis:
    """Full NLP analysis for a single sentence."""

    sentence_index: int
    text: str
    tokens: list[TokenInfo] = field(default_factory=list)
    entities: list[EntityInfo] = field(default_factory=list)
    pos_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence_index": self.sentence_index,
            "sentence_text": self.text,
            "tokens": [
                {
                    "text": t.text,
                    "lemma": t.lemma,
                    "pos": t.pos,
                    "dep": t.dep,
                    "head": t.head_text,
                }
                for t in self.tokens
            ],
            "pos_tags": self.pos_tags,
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "start_char": e.start_char,
                    "end_char": e.end_char,
                }
                for e in self.entities
            ],
        }


@dataclass
class SpeechAnalysis:
    """Full NLP analysis for an entire speech."""

    speech_id: str
    sentences: list[SentenceAnalysis] = field(default_factory=list)
    all_entities: list[EntityInfo] = field(default_factory=list)

    @property
    def entity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entity in self.all_entities:
            key = f"{entity.label}:{entity.text}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def to_dict(self) -> dict:
        return {
            "speech_id": self.speech_id,
            "num_sentences": len(self.sentences),
            "num_entities": len(self.all_entities),
            "sentences": [s.to_dict() for s in self.sentences],
            "entity_summary": self.entity_counts,
        }


def analyze_sentence(sent: spacy.tokens.Span, index: int) -> SentenceAnalysis:
    """Analyze a single spaCy sentence span."""
    analysis = SentenceAnalysis(
        sentence_index=index,
        text=sent.text.strip(),
    )

    for token in sent:
        analysis.tokens.append(TokenInfo(
            text=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            dep=token.dep_,
            head_text=token.head.text,
            is_stop=token.is_stop,
            is_punct=token.is_punct,
        ))
        analysis.pos_tags.append(token.pos_)

    # NER via BETO transformer (replaces spaCy NER)
    ner = _get_ner_pipeline()
    sent_text = sent.text.strip()
    if sent_text:
        ner_results = ner(sent_text)
        for r in ner_results:
            entity_text = r["word"].strip()
            # Strip trailing punctuation leaked by tokenizer
            entity_text = entity_text.rstrip(".,;:!?\"')")
            if not entity_text:
                continue
            # Discard known noise
            if entity_text in _ENTITY_BLACKLIST:
                continue
            # Gazetteer correction: override label for known locations
            label = r["entity_group"]
            if entity_text in _LOCATION_GAZETTEER and label != "LOC":
                label = "LOC"
            # Offsets from BETO are relative to sent_text;
            # convert to speech-level offsets
            analysis.entities.append(EntityInfo(
                text=entity_text,
                label=label,
                start_char=sent.start_char + r["start"],
                end_char=sent.start_char + r["end"],
            ))

    return analysis


def analyze_speech(speech_id: str, text: str) -> SpeechAnalysis:
    """Run full NLP pipeline on a speech text.

    Processes the text sentence by sentence to manage memory.
    """
    nlp = load_model()
    speech_analysis = SpeechAnalysis(speech_id=speech_id)

    doc = nlp(text)

    for i, sent in enumerate(doc.sents):
        sent_analysis = analyze_sentence(sent, i)
        speech_analysis.sentences.append(sent_analysis)
        speech_analysis.all_entities.extend(sent_analysis.entities)

    logger.info(
        "Analyzed %s: %d sentences, %d entities",
        speech_id,
        len(speech_analysis.sentences),
        len(speech_analysis.all_entities),
    )
    return speech_analysis


def analyze_from_file(
    speech_id: str,
    input_dir: Path = PROCESSED_DIR,
) -> SpeechAnalysis:
    """Load a cleaned transcript and run NLP analysis."""
    input_path = input_dir / f"{speech_id}.json"

    with open(input_path) as f:
        data = json.load(f)

    return analyze_speech(speech_id, data["full_text"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_id = "bGeWx5YWoro"
    test_path = PROCESSED_DIR / f"{test_id}.json"

    if test_path.exists():
        analysis = analyze_from_file(test_id)
        result = analysis.to_dict()

        print(f"\nSentences: {result['num_sentences']}")
        print(f"Entities: {result['num_entities']}")

        print("\nTop 15 entities:")
        for key, count in list(result["entity_summary"].items())[:15]:
            print(f"  {key}: {count}")

        print("\nSample sentence analysis (sentence 5):")
        if len(result["sentences"]) > 5:
            sent = result["sentences"][5]
            print(f"  Text: {sent['sentence_text']}")
            print(f"  POS: {sent['pos_tags']}")
            if sent["entities"]:
                print(f"  Entities: {[(e['text'], e['label']) for e in sent['entities']]}")
    else:
        logger.error("Processed transcript not found: %s", test_path)
