"""Stream-processing orchestrator for the full corpus pipeline.

Processes one speech at a time: download → diarize → transcribe → delete audio →
clean → NLP analysis → load to DB. Minimizes disk usage (~60MB peak).
"""

import json
import logging
import sys
from pathlib import Path

from src.corpus.downloader import (
    build_manifest,
    download_audio,
    scrape_channel_metadata,
)
from src.corpus.transcriber import transcribe_audio, load_text_speech
from src.corpus.cleaner import clean_transcript
from src.corpus.db_loader import get_connection, get_corpus_stats, load_speech, speech_exists
from src.corpus.diarizer import (
    load_reference_embedding,
    remap_timestamps,
    run_diarization,
    save_diarization_result,
    REFERENCE_EMBEDDING_PATH,
)
from src.pipeline.nlp_processor import analyze_speech

try:
    from src.rag.chunker import chunk_speech_from_db
    from src.rag.embedder import embed_texts
    from src.corpus.db_loader import chunks_exist, load_chunks
    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/speech_manifest.json")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
AUDIO_DIR = Path("data/audio")


def process_single_speech(
    manifest_entry: dict,
    conn,
    keep_audio: bool = False,
    skip_diarization: bool = False,
) -> bool:
    """Process a single speech through the full pipeline.

    Returns True if successfully processed, False if skipped or failed.
    """
    speech_id = manifest_entry["id"]
    title = manifest_entry["title"]
    source = manifest_entry.get("source", "youtube")

    logger.info("=" * 60)
    logger.info("Processing: %s", title)
    logger.info("ID: %s | Source: %s", speech_id, source)

    diarization_result = None

    try:
        # Step 1: Get raw transcript
        raw_path = RAW_DIR / f"{speech_id}.json"

        if source == "website_text":
            # Text-only speech — load from file (no diarization possible)
            text_file = Path(manifest_entry.get("text_file", ""))
            if not text_file.exists():
                logger.error("Text file not found: %s", text_file)
                return False
            raw = load_text_speech(speech_id, text_file)
        elif raw_path.exists():
            # Already transcribed
            logger.info("Transcript already exists, skipping download/transcribe")
            with open(raw_path) as f:
                raw = json.load(f)
        else:
            # Download audio
            logger.info("Downloading audio...")
            audio_path = download_audio(manifest_entry["url"], AUDIO_DIR)

            # Diarize FIRST (while audio still exists)
            if not skip_diarization:
                logger.info("Running speaker diarization...")
                speaker_audio, offsets, diarization_result = run_diarization(
                    audio_path, speech_id, REFERENCE_EMBEDDING_PATH,
                )

                if diarization_result.target_speaker is not None:
                    pct = (
                        diarization_result.target_duration_seconds
                        / diarization_result.duration_seconds
                        * 100
                    )
                    logger.info(
                        "Diarization: identified Cepeda as %s "
                        "(confidence=%.3f), %.1f%% of audio (%.0fs / %.0fs)",
                        diarization_result.target_speaker,
                        diarization_result.confidence,
                        pct,
                        diarization_result.target_duration_seconds,
                        diarization_result.duration_seconds,
                    )

                    # Transcribe only Cepeda's extracted audio
                    logger.info("Transcribing Cepeda's audio only...")
                    raw = transcribe_audio(speaker_audio, speech_id)

                    # Remap timestamps to original video time
                    if offsets:
                        raw = remap_timestamps(raw, offsets)
                        with open(RAW_DIR / f"{speech_id}.json", "w") as f:
                            json.dump(raw, f, ensure_ascii=False, indent=2)

                    # Clean up extracted speaker audio
                    if speaker_audio != audio_path and speaker_audio.exists():
                        speaker_audio.unlink()
                        logger.info("Deleted extracted audio: %s", speaker_audio.name)
                else:
                    logger.warning(
                        "Diarization: could not identify Cepeda "
                        "(best confidence=%.3f). Transcribing full audio.",
                        diarization_result.confidence,
                    )
                    logger.info("Transcribing full audio with Whisper...")
                    raw = transcribe_audio(audio_path, speech_id)

                # Save diarization result for debugging/auditing
                save_diarization_result(speech_id, diarization_result)
            else:
                # Diarization skipped — transcribe full audio
                logger.info("Transcribing with Whisper...")
                raw = transcribe_audio(audio_path, speech_id)

            # Delete original audio to save disk space
            if not keep_audio and audio_path.exists():
                audio_path.unlink()
                logger.info("Deleted audio: %s", audio_path.name)

        # Step 2: Clean transcript
        processed_path = PROCESSED_DIR / f"{speech_id}.json"
        if processed_path.exists():
            logger.info("Cleaned transcript already exists")
            with open(processed_path) as f:
                cleaned = json.load(f)
        else:
            logger.info("Cleaning transcript...")
            cleaned, report = clean_transcript(speech_id)
            logger.info("Cleaning: %s", report.summary)

        # Step 3: NLP analysis
        logger.info("Running NLP analysis...")
        analysis = analyze_speech(speech_id, cleaned["full_text"])

        # Step 4: Load to DB
        logger.info("Loading to database...")
        speech_db_id = load_speech(
            conn, manifest_entry, raw, cleaned, analysis,
            diarization_result=diarization_result,
        )

        # Step 5: Chunk + embed for RAG
        if _HAS_RAG and not chunks_exist(conn, speech_db_id):
            logger.info("Chunking and embedding for RAG...")
            chunks = chunk_speech_from_db(conn, speech_db_id)
            if chunks:
                chunk_texts = [c.text for c in chunks]
                embeddings = embed_texts(chunk_texts)
                load_chunks(conn, speech_db_id, chunks, embeddings)
                logger.info(
                    "Created %d chunks with embeddings for speech_id=%d",
                    len(chunks), speech_db_id,
                )

        logger.info("Done: %s (db_id=%d)", title, speech_db_id)
        return True

    except Exception:
        logger.exception("Failed to process: %s", title)
        return False


def run_pipeline(
    max_new: int | None = None,
    max_scrape: int | None = None,
    keep_audio: bool = False,
    skip_diarization: bool = False,
) -> dict:
    """Run the full pipeline on the channel.

    Args:
        max_new: Process at most N *new* speeches (skips already-loaded ones).
        max_scrape: Limit number of videos to scrape from channel.
            Defaults to all campaign speeches (~45).
        keep_audio: If True, don't delete audio files after transcription.
        skip_diarization: If True, skip speaker diarization entirely.

    Returns summary statistics.
    """
    # Validate reference embedding exists before processing
    if not skip_diarization:
        try:
            load_reference_embedding(REFERENCE_EMBEDDING_PATH)
            logger.info("Reference embedding loaded from %s", REFERENCE_EMBEDDING_PATH)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    # Step 1: Scrape channel and build manifest
    logger.info("Scraping channel metadata...")
    entries = scrape_channel_metadata(max_videos=max_scrape)
    logger.info("Found %d videos", len(entries))

    logger.info("Building manifest with full metadata...")
    manifest = build_manifest(entries)
    logger.info("Manifest has %d entries", len(manifest))

    # Step 2: Process each speech, counting only new ones toward the limit
    conn = get_connection()
    new_processed = 0
    skipped = 0
    failures = 0

    try:
        for i, entry in enumerate(manifest):
            # Stop once we've processed enough new speeches
            if max_new is not None and new_processed >= max_new:
                logger.info("Reached --new=%d limit, stopping", max_new)
                break

            logger.info("\n[%d/%d] Processing speech...", i + 1, len(manifest))

            # Check if already in DB before doing any work
            youtube_url = entry.get("url")
            existing_id = speech_exists(conn, youtube_url, entry["title"])
            if existing_id:
                logger.info(
                    "Already in DB (id=%d), skipping: %s",
                    existing_id, entry["title"],
                )
                skipped += 1
                continue

            ok = process_single_speech(
                entry, conn,
                keep_audio=keep_audio,
                skip_diarization=skip_diarization,
            )
            if ok:
                new_processed += 1
            else:
                failures += 1

        stats = get_corpus_stats(conn)
    finally:
        conn.close()

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(
        "New: %d, Skipped: %d, Failed: %d",
        new_processed, skipped, failures,
    )
    logger.info("Corpus stats: %s", stats)

    return {
        "new_processed": new_processed,
        "skipped": skipped,
        "failures": failures,
        "corpus_stats": stats,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse simple CLI args
    max_new = None
    max_scrape = None
    keep_audio = False
    skip_diarization = False

    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("--new="):
            max_new = int(arg.split("=")[1])
        elif arg.startswith("--scrape="):
            max_scrape = int(arg.split("=")[1])
        elif arg == "--keep-audio":
            keep_audio = True
        elif arg == "--skip-diarization":
            skip_diarization = True
        elif arg == "--all":
            max_scrape = 45

    if not args:
        print("Usage: python -m src.corpus.pipeline_runner [OPTIONS]")
        print("\nOptions:")
        print("  --new=N              Process N new speeches (skips already loaded)")
        print("  --all                Process all campaign speeches (~45)")
        print("  --scrape=N           Limit channel scraping to N videos")
        print("  --keep-audio         Don't delete audio files after processing")
        print("  --skip-diarization   Skip speaker diarization (transcribe full audio)")
        print("\nExamples:")
        print("  python -m src.corpus.pipeline_runner --new=5")
        print("  python -m src.corpus.pipeline_runner --all")
        print("  python -m src.corpus.pipeline_runner --new=3 --skip-diarization")
        sys.exit(0)

    # Default: scrape enough to find new speeches
    if max_scrape is None:
        if max_new is not None:
            # Scrape more than needed to account for already-loaded ones
            max_scrape = max_new + 20
        else:
            max_scrape = 45

    result = run_pipeline(
        max_new=max_new,
        max_scrape=max_scrape,
        keep_audio=keep_audio,
        skip_diarization=skip_diarization,
    )
    print(f"\nFinal result: {json.dumps(result, indent=2)}")
