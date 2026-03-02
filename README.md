# CLINI-Q — Intelligent SOP Navigator (Streamlit MVP)

This is a minimal, **deployable Streamlit app** that demonstrates the CLINI‑Q concept: an AI‑assisted SOP navigator that turns role‑specific questions into actionable procedural guidance with citations to site SOPs.

> **MVP Guardrails:** No medical advice. No PHI/PII. Not an IRB/regulatory submission tool. Verify locally with PI & SOPs.

## Features
- Role selection (CRC, RN, Admin, Trainee) and scenario entry
- TF‑IDF retrieval over local SOP files (`/data/sops/*.txt|*.pdf`)
- Optional OpenAI drafting of step‑by‑step guidance with citations (falls back to offline mode)
- Clear compliance reminders and disclaimers

## Quickstart (Local)
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# (optional) export OPENAI_API_KEY=sk-...

streamlit run app.py
```

Place your SOP files under `data/sops/`. You can use `python ingest.py <path/to/file.pdf>` to copy files into the repo.

## Deploy to Streamlit Community Cloud
1. Push this folder to a **GitHub repo** (e.g., `cliniq-streamlit`).
2. In Streamlit Cloud, create a new app:
   - **Repository:** `yourname/cliniq-streamlit`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. In *Settings → Secrets*, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Click **Deploy**.

## Configuration
- `OPENAI_MODEL` env var (optional) chooses the model (default `gpt-4o-mini`).
- Evidence snippets shown per query are controlled by the sidebar slider.

## Project Structure
```
.
├── app.py
├── ingest.py
├── requirements.txt
├── .streamlit/
│   └── secrets.toml (example)
└── data/
    └── sops/
        └── sample_sop.txt
```

## Notes
- PDF text extraction uses `pypdf`. Ensure your PDFs are text‑based, not pure scans.
- Retrieval uses scikit‑learn TF‑IDF; replace with vector search later if needed.
- The app outputs procedural guidance and citations to file names; always verify with your SOPs.
- Licensed MIT.
