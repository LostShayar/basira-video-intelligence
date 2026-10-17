# Architecture

## Design rule

Computer vision produces the facts. The AI layer only verifies, searches and explains facts that are already stored. A slow model can never stall the video pipeline, and the language model never invents events.

## Data flow

1. **Ingest** — one reader thread per camera pushes frames into a bounded queue. When the queue is full, the oldest frame is dropped, so latency stays flat instead of growing.
2. **Detect + track** — YOLO detects people and vehicles; ByteTrack gives each object a stable ID across frames.
3. **Analytics** — per-camera zones and lines from `configs/cameras.yaml`: line crossings, restricted-zone entry, dwell time (loitering), crowd density.
4. **Event engine** — turns rule hits into events with a cooldown, saves a face-blurred snapshot and a short clip, and writes a row to PostgreSQL.
5. **Intelligence (async workers)**
   - *VLM verifier* — reads new events, sends snapshot + context to a vision-language model, stores `{confirmed, reason, confidence}`.
   - *Embeddings* — CLIP embeddings of keyframes and crops, stored in pgvector for text search.
   - *Copilot* — a LangGraph agent with read-only tools over events and footage, also exposed as an MCP server.
6. **Product** — FastAPI serves events, search and chat; Streamlit renders tiles, the event feed, search and the copilot panel.

## Key decisions

| Decision | Why |
| --- | --- |
| Bounded queues that drop stale frames | Real-time systems prefer fresh frames over complete ones |
| PostgreSQL + pgvector instead of separate vector DB | Events and embeddings in one place, one less service |
| Verification runs async, after the event is stored | Video pipeline latency is independent of model latency |
| Face blur before anything is written to disk | Privacy by default, not as an add-on |
| Agent uses a read-only DB role | The copilot cannot modify or delete data |

## Event schema (planned)

| Column | Type |
| --- | --- |
| id | uuid |
| camera_id | text |
| track_id | int |
| event_type | text (line_cross, zone_entry, loitering, crowding) |
| zone_id | text |
| started_at | timestamptz |
| confidence | float |
| snapshot_path | text |
| clip_path | text |
| verified | bool, null until checked |
| verify_reason | text |
