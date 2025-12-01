# Know My Rights — Python + AI MVP

A minimal, local-first prototype for an Indian legal rights companion with:
- **FastAPI backend** providing:
  - `/api/guide?scenario=...` → step-by-step actions for situations (arrest, accident, traffic stop, harassment).
  - `/api/ask` → AI Q&A over your local markdown content using embeddings + semantic search (RAG).
  - `/api/classify` → lightweight text classification to auto-detect scenario from free text.
  - `/api/sos` → (optional) SMS/WhatsApp SOS via Twilio (stubbed by default).
- **Streamlit UI** with emergency buttons + chat.
- **Local content store** in `content/*.md`, fully editable by you.
- **Local vector DB** using ChromaDB + `sentence-transformers` (no external APIs required).

> ⚠️ **Disclaimer:** This tool provides general information, not legal advice. For legal decisions, consult a qualified lawyer. Laws change and vary by state; verify all guidance before use.

## Quick Start

1) **Create a virtual env** (recommended) and install packages:
```bash
pip install -r requirements.txt
```

2) **(One-time)** Build the local index (embeddings) from `content/`:
```bash
python ai/build_index.py
```

3) **Run the backend**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4) **Run the UI** (in another terminal):
```bash
streamlit run streamlit_app.py
```

Visit the UI at `http://localhost:8501`. Backend at `http://localhost:8000`.

## Configure SOS (Optional)
- Copy `.env.example` to `.env` and fill:
  - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`
- Add up to three comma-separated phone numbers in the UI.
- The `/api/sos` endpoint will send messages when configured; otherwise it will no-op.

## Extend / Customize
- Add or edit files in `content/` with your jurisdiction-specific rights.
- Re-run `python ai/build_index.py` after you edit content.
- Add more scenarios in `app/scenarios.py` and corresponding markdowns in `content/`.
- Add languages by creating `content_hi` etc. and tweaking `ai/build_index.py` to index them.

## Tech Notes
- Embeddings model: `sentence-transformers/all-MiniLM-L6-v2` (small, fast, good quality).
- Vector store: ChromaDB (persisted under `.chroma/`).
- Classifier: a simple zero-shot fallback using the embedding similarity vs scenario titles + synonyms.
- No internet or paid API required. You can swap models later if needed.

## Security & Privacy
- This is a prototype. If used in production, implement user auth, encryption at rest, secure logs, rate limits,
  PII minimization, and audit trails. Do not collect more than necessary.
