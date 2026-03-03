# Future Ideas

## 1. Interaction Memory — User Analytics Table

Store user interactions to understand how people use the system: what they ask, what topics are popular, which chunks get retrieved most, how conversations flow.

**Open questions:**
- What exactly to track? Candidates: raw question text, detected intent, retrieved chunk IDs, similarity scores, response length, session ID, timestamp, whether the user asked a follow-up, tool calls triggered
- Should we store the full conversation turn or just the query?
- Privacy considerations — is anonymous session-level tracking enough?
- Can we derive insights like "most asked topics", "questions with no good matches", "average conversation depth"?

**Potential uses:**
- Identify gaps in the corpus (questions where retrieval scores are consistently low)
- Understand which speeches/topics generate the most interest
- Improve the system prompt based on real usage patterns
- Build a feedback loop: low-similarity retrievals flag topics that need more coverage

## 2. Living Corpus — Automated Speech Ingestion

Turn the system from a static snapshot into a self-updating knowledge base. When new speeches appear on the YouTube channel, automatically process and incorporate them.

**Implementation sketch:**
- Cron job (or scheduled task) that periodically checks the YouTube channel for new videos
- Compares against `speeches` table to identify unprocessed uploads
- Runs the full pipeline: download → diarize → transcribe → clean → NLP → DB → chunk → embed
- Pipeline already supports this (`pipeline_runner.py --new=N`), just needs automation
- Could run on the Mac Mini (where the heavy models live) or trigger remotely

**Open questions:**
- Frequency? Daily, hourly, on-demand?
- Notification when new speeches are processed?
- Error handling — what if a video fails mid-pipeline?
- Should it point at Supabase directly or sync after local processing?
- Campaign season may produce bursts of content — need to handle batches gracefully
