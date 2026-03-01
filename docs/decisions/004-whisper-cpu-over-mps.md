# ADR 004: Whisper CPU Over MPS on Apple Silicon

**Date:** 2026-02-24
**Status:** Accepted

## Context
Tested Whisper large-v3 transcription on M4 Mac Mini with both CPU and MPS (Metal Performance Shaders) backends.

## Decision
Use CPU for all Whisper transcription.

## Rationale
- CPU is ~1.9x faster than MPS on M4 Mac Mini for Whisper large-v3.
- MPS support for Whisper is immature — causes slowdowns, not speedups.
- ~7.5 min transcription per 20-min speech on CPU is acceptable.

## Consequences
- No GPU acceleration for transcription, but faster anyway.
- Full corpus (~44 speeches) takes ~5-6 hours on CPU.
