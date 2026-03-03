#!/usr/bin/env python3
# CLINI-Q • SOP Navigator (consolidated)

import os
import csv
import re
import base64
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
from difflib import SequenceMatcher, get_close_matches

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PATHS & CONFIG ------------------
ROOT_DIR = Path(__file__).parent
ASSETS_DIR = ROOT_DIR / "assets"
ICON_PATH = ASSETS_DIR / "icon.png"           # optional small square icon for chat bubble
LOGO_PATH = ASSETS_DIR / "cliniq_logo.png"    # optional wide header logo

# CSV: repo root preferred; fallback to data/
FAQ_CSV = ROOT_DIR / "cliniq_faq.csv"
if not FAQ_CSV.exists():
    FAQ_CSV = ROOT_DIR / "data" / "cliniq_faq.csv"

DEFAULT_SOP_DIR = ROOT_DIR / "data" / "sops"
DATA_DIR = Path(os.environ.get("SOP_DIR", "").strip() or DEFAULT_SOP_DIR)

APP_TITLE = "CLINI-Q • SOP Navigator"
DISCLAIMER = (
    " 💻This tool provides procedural guidance only. Do not use for clinical decisions or PHI. "
    " 📚Always verify with your site SOPs and Principal Investigator (PI)."
)
FINAL_VERIFICATION_LINE = "Verify with your site SOP and PI before execution."

# ------------------ ROLES & SCENARIOS ------------------
ROLES = {
    "Clinical Research Coordinator (CRC)": "CRC",
    "Registered Nurse (RN)": "RN",
    "Administrator (Admin)": "ADMIN",
    "Trainee": "TRAINEE",
    "Participant": "PARTICIPANT",
}

ROLE_SCENARIOS: Dict[str, List[str]] = {
    "CRC": ["IP shipment", "Missed visit", "Adverse event (AE) reporting", "Protocol deviation", "Monitoring visit preparation"],
    "RN": ["Pre-dose checks for IP", "AE identification and documentation", "Unblinding contingency", "Concomitant medication documentation"],
    "ADMIN": ["Delegation log management", "Regulatory binder maintenance", "Safety report distribution", "IRB submission packet assembly"],
    "TRAINEE": ["SOP basics: GCP overview", "Site initiation: required logs", "Source documentation fundamentals"],
    "PARTICIPANT": [
        "Missed/rescheduled visits — windows, documentation, safety checks",
        "Duration & schedule — calendars, visit frequency, conflicts in windows",
        "Costs & reimbursements — billing, travel/parking/meals",
        "Placebo & randomization — plain-language explanation",
        "Side effects & AEs — who to contact, how handled",
        "Privacy & confidentiality — protections, IRB oversight",
        "Eligibility, alternatives, withdrawal rights, results access, complaints",
        "Participant communication — guidance & complaint pathways",
    ],
}

CLARIFYING_QUESTIONS: Dict[str, List[Dict[str, List[str]]]] = {
    "IP shipment": [
        {"Shipment type?": ["Initial shipment", "Resupply", "Return/destruction"]},
        {"Temperature control?": ["Ambient", "Refrigerated (2–8°C)", "Frozen (≤ -20°C)"]},
        {"Chain of custody ready?": ["Yes", "No"]},
    ],
    "Missed visit": [
        {"Visit window status?": ["Within window", "Outside window"]},
        {"Reason documented?": ["Yes", "No"]},
        {"Make-up allowed by protocol?": ["Yes", "No", "Unclear"]},
    ],
    "Adverse event (AE) reporting": [
        {"AE seriousness?": ["Non-serious", "Serious (SAE)"]},
        {"Related to IP?": ["Related", "Not related", "Unknown"]},
        {"Expectedness (per IB)?": ["Expected", "Unexpected", "Unknown"]},
    ],
}

# ------------------ DATA TYPES ------------------
@dataclass
class Snippet:
    text: str
    source: str
    score: float

# ------------------ HELPERS ------------------
def _img_to_b64(path: Path) -> str:
    try:
        img = Image.open(path)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

def _show_bubble(html: str, avatar_b64: str):
    st.markdown(
        f"""
        <div style='display:flex;align-items:flex-start;margin:10px 0;'>
            {'<img src="data:image/png;base64,'+avatar_b64+'" width="40" style="margin-right:10px;border-radius:8px;"/>' if avatar_b64 else ''}
            <div style='background:#f6f6f6;padding:12px;border-radius:120px;max-width:75%;'>
                {html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
from difflib import get_close_matches

def get_question_pool(faq_df: pd.DataFrame, sel_df: pd.DataFrame) -> List[str]:
    """Prefer selected category questions; if none, fall back to all FAQs."""
    if isinstance(sel_df, pd.DataFrame) and not sel_df.empty:
        pool = sel_df["Question"].dropna().tolist()
    elif isinstance(faq_df, pd.DataFrame) and not faq_df.empty:
        pool = faq_df["Question"].dropna().tolist()
    else:
        pool = []
    # dedupe but keep order
    seen, uniq = set(), []
    for q in pool:
        if q not in seen:
            uniq.append(q); seen.add(q)
    return uniq

def suggest_questions(query: str, pool: List[str], n: int = 5) -> List[str]:
    """Return up to n similar questions (dash/space-insensitive). Fallback to first n."""
    if not pool:
        return []
    norm_map = { _norm(x): x for x in pool }
    candidates = get_close_matches(_norm(query), list(norm_map.keys()), n=n, cutoff=0.35)
    if candidates:
        return [norm_map[c] for c in candidates]
    return pool[:n]

def render_alt_buttons(alts: List[str], df_scope: pd.DataFrame, faq_df: pd.DataFrame):
    """Show clickable alternatives and answer upon click (exact → tolerant fuzzy)."""
    if not alts:
        st.session_state["chat"].append({"role":"assistant","content":"No FAQs available yet. Please add rows to cliniq_faq.csv."})
        return
    st.markdown("Here are similar questions:")
    cols = st.columns(min(3, len(alts)))
    for i, q in enumerate(alts):
        with cols[i % len(cols)]:
            if st.button(q, key=f"alt_{i}", use_container_width=True):
                st.session_state["chat"].append({"role": "user", "content": q})
                scope = df_scope if isinstance(df_scope, pd.DataFrame) and not df_scope.empty else faq_df
                # exact
                if q in scope["Question"].values:
                    ans = scope[scope["Question"] == q].iloc[0]["Answer"]
                    st.session_state["chat"].append({"role":"assistant","content":f"<b>Answer:</b> {ans}"})
                    st.session_state["clear_input"] = True
                    st.rerun()
                # tolerant fuzzy
                best_q, best_score = None, 0.0
                for cand in scope["Question"].tolist():
                    score = SequenceMatcher(None, _norm(q), _norm(cand)).ratio()
                    if score > best_score:
                        best_q, best_score = cand, score
                if best_q and best_score >= 0.75:
                    ans = scope[scope["Question"] == best_q].iloc[0]["Answer"]
                    st.session_state["chat"].append({"role":"assistant","content":f"<b>Answer:</b> {ans}"})
                else:
                    st.session_state["chat"].append({"role":"assistant","content":"Still couldn’t match that wording. Try another suggestion."})
                st.session_state["clear_input"] = True
                st.rerun()

# tolerant string normalizer (dash/space/case-insensitive)
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    # normalize all dash variants to a plain hyphen
    s = re.sub(r"[\u2010-\u2015\u2212\-]+", "-", s)
    # remove punctuation that often varies between CSV and UI
    s = re.sub(r"[,:;/\\()\\[\\]{}\"'·•–—]+", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def load_faq_csv_tolerant(path: Path) -> pd.DataFrame:
    """
    Reads CSV with expected columns: Category, Question, Answer.
    If a row has more than 3 columns (because Answer contains commas),
    extra columns are joined back into Answer.
    """
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["Category", "Question", "Answer"])

    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header row
        for raw in reader:
            if not raw or all(not c.strip() for c in raw):
                continue
            if len(raw) == 1:
                raw = [c.strip() for c in raw[0].split(",")]
            if len(raw) < 3:
                raw += [""] * (3 - len(raw))
            cat = raw[0].strip()
            q   = raw[1].strip()
            ans = ",".join(raw[2:]).strip()
            rows.append([cat, q, ans])

    df = pd.DataFrame(rows, columns=["Category", "Question", "Answer"]).fillna("")
    df["Category"] = df["Category"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["Question"] = df["Question"].str.strip()
    df["Answer"]   = df["Answer"].str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for p in sorted(data_dir.glob("**/*")):
        if p.suffix.lower() == ".txt":
            try:
                docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                pass
        elif p.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(p))
                pages = [page.extract_text() or "" for page in reader.pages]
                docs.append((p.name, "\n".join(pages)))
            except Exception:
                pass
    if not docs:
        docs = [("placeholder.txt", "No SOP files found. Add .txt/.pdf under data/sops.")]
    return docs

@st.cache_data(show_spinner=False)
def build_index(docs: List[Tuple[str, str]]):
    sources = [d[0] for d in docs]
    corpus = [d[1] for d in docs]
    n = len(corpus)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=(0.95 if n > 1 else 1.0))
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, sources, corpus

def retrieve(query: str, vectorizer, matrix, sources, corpus, k: int = 5) -> List[Snippet]:
    if not query.strip():
        return []
    sims = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    idxs = sims.argsort()[::-1][:k]
    return [Snippet(text=corpus[i][:2000], source=sources[i], score=float(sims[i])) for i in idxs]

def compose_guidance(role_label: str, scenario: str, answers: Dict[str, str], snippets: List[Snippet]) -> dict:
    role_short = ROLES.get(role_label, role_label)
    cites = sorted({f"Source: {s.source}" for s in snippets})
    steps = [
        f"Confirm {role_short} responsibilities for '{scenario}' using cited SOP sections.",
        "Identify protocol windows/criteria impacted based on clarifying details provided.",
        "Follow site-required documentation order; complete forms/logs referenced in citations.",
        "Record actions with date/time, signer, and cross-references in source records.",
        "Escalate uncertainties to PI/medical lead and document guidance.",
    ]
    return {
        "steps": steps,
        "citations": cites,
        "compliance": [
            "Adhere to ICH-GCP E6(R2) and site SOPs.",
            "Use site-approved templates; maintain confidentiality (no PHI in this tool).",
        ],
        "disclaimer": FINAL_VERIFICATION_LINE,
    }

# ------------------ MAIN APP ------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")

    # --- Header (RISe-like, left aligned) ---
    st.markdown(
        """
        <style>
          .hero { text-align:left; margin-top:.10rem; }
          .hero h1 { font-size:2.05rem; font-weight:1000; margin:0; }
          .hero p  { font-size:1.5rem; color:#333; max-width:2000px; margin:.35rem 0 0 0; }
          .divider-strong { border-top:4px solid #222; margin:.4rem 0 1.0rem; }
          .card { border:1px solid #e5e7eb; border-radius:12px; padding:.8rem 1rem; background:#fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=420)
    st.markdown(
        """
        <div class="hero">
          <h1>💡Smart Assistant for Clinical Trial SOP Navigation</h1>
          <p> 🛡️I am trained on institutional Standard Operating Procedures (SOPs) and compliance frameworks, helping research teams navigate essential documentation, regulatory requirements, and Good Clinical Practice (GCP) standards with clarity and confidence.🛡️</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider-strong"></div>', unsafe_allow_html=True)
    st.caption(DISCLAIMER)

    # --- Session defaults ---
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("suggested", [])
    st.session_state.setdefault("last_role", None)
    st.session_state.setdefault("last_category", None)
    st.session_state.setdefault("clear_input", False)

    # --- Sidebar ---
    with st.sidebar:
        st.header("User Setup")

        faq_df = load_faq_csv_tolerant(FAQ_CSV)
        categories = ["All Categories"] + sorted(faq_df["Category"].unique().tolist()) if not faq_df.empty else ["All Categories"]
        category = st.selectbox("📂 Knowledge category (optional)", categories, key="category_select")

        role_label = st.selectbox("🎭 Your role", list(ROLES.keys()), key="role_select")
        role_code = ROLES[role_label]
        scenario_list = ROLE_SCENARIOS.get(role_code, [])
        scenario = st.selectbox("📌 Scenario", scenario_list if scenario_list else ["—"], key="scenario_select")

        st.subheader("Clarifying questions")
        answers: Dict[str, str] = {}
        for qdef in CLARIFYING_QUESTIONS.get(scenario, []):
            for q, opts in qdef.items():
                answers[q] = st.selectbox(q, opts, key=f"clar_{q}")

        st.slider("Evidence snippets", min_value=3, max_value=10, value=5, step=1, key="k_slider")

        # Refresh dynamic prompts on change
        role_changed = (st.session_state["last_role"] != role_code)
        cat_changed  = (st.session_state["last_category"] != category)
        if role_changed or cat_changed:
            st.session_state["suggested"] = []
            st.session_state["last_role"] = role_code
            st.session_state["last_category"] = category

        st.divider()
        st.subheader("Document Upload")
        uploaded = st.file_uploader("📎 Upload a reference file (optional)", type=["pdf", "docx", "txt"])
        if uploaded:
            st.success(f"Uploaded file: {uploaded.name}")
            
    # --- Dynamic suggestions ---
    sel_df = faq_df if category == "All Categories" else faq_df[faq_df["Category"] == category]
    if not sel_df.empty and category != "All Categories":
        suggestions = sel_df["Question"].head(4).tolist()
    else:
        role_list = ROLE_SCENARIOS.get(role_code, [])
        suggestions = [f"What are the steps for {s}?" for s in role_list[:4]]

    if suggestions:
        st.markdown("#### Try asking one of these:")
        cols = st.columns(min(4, len(suggestions)))
        icon_b64 = _img_to_b64(ICON_PATH)
        for i, s in enumerate(suggestions):
            with cols[i % len(cols)]:
                if st.button(s, key=f"sugg_{role_code}_{category}_{i}", use_container_width=True):
                    st.session_state["chat"].append({"role": "user", "content": s})
                    answered = False
                    if not sel_df.empty:
                        # 1) exact match
                        if s in sel_df["Question"].values:
                            ans = sel_df[sel_df["Question"] == s].iloc[0]["Answer"]
                            st.session_state["chat"].append({"role": "assistant", "content": f"<b>Answer:</b> {ans}"})
                            answered = True
                        else:
                            # 2) tolerant/fuzzy match
                            q_norm = _norm(s)
                            best_q, best_score = None, 0.0
                            for q in sel_df["Question"].tolist():
                                score = SequenceMatcher(None, q_norm, _norm(q)).ratio()
                                if score > best_score:
                                    best_q, best_score = q, score
                            if best_q and best_score >= 0.75:
                                ans = sel_df[sel_df["Question"] == best_q].iloc[0]["Answer"]
                                st.session_state["chat"].append({"role": "assistant", "content": f"<b>Answer:</b> {ans}"})
                                answered = True
                    if not answered:
                        pool = get_question_pool(faq_df, sel_df)
                        alts = suggest_questions(s, pool, n=5)
                        html = "I couldn't find a close match. Try one of these:"
                        st.session_state["chat"].append({"role":"assistant","content": html})
                        render_alt_buttons(alts, sel_df, faq_df)

                    st.session_state["clear_input"] = True
                    st.rerun()
    # --- Manual question input ---
    question = st.text_input(
        "💬 What would you like me to help you with?",
        value="" if not st.session_state["clear_input"] else "",
        placeholder="Ask about steps, documentation, reporting timelines…",
        key="free_text",
    )
    st.session_state["clear_input"] = False

    if st.button("Submit", key="submit_btn") and question.strip():
        st.session_state["chat"].append({"role": "user", "content": question})
        if not sel_df.empty:
            all_q = sel_df["Question"].tolist()
            best, score = None, 0.0
            for q in all_q:
                s = SequenceMatcher(None, _norm(question), _norm(q)).ratio()
                if s > score:
                    best, score = q, s
            if best and score >= 0.75:
                ans = sel_df[sel_df["Question"] == best].iloc[0]["Answer"]
                st.session_state["chat"].append({"role": "assistant", "content": f"<b>Answer:</b> {ans}"})
            else:
                top = get_close_matches(_norm(question), [_norm(x) for x in all_q], n=3, cutoff=0.45)
                if top:
                    # map normalized back to original for display
                    inv = { _norm(x): x for x in all_q }
                    top_disp = [inv[t] for t in top if t in inv]
                    msg = (
                        "I couldn't find an exact match. Here are similar questions:<br>"
                        + "<br>".join(f"{i}. {t}" for i, t in enumerate(top_disp, start=1))
                        + "<br>Click a suggestion above or refine your query."
                    )
                    st.session_state["chat"].append({"role": "assistant", "content": msg})
                else:
                    st.session_state["chat"].append({"role": "assistant", "content": "No close match found in the selected category."})
        st.rerun()

    # --- Chat render (bubble look with icon) ---
    if st.session_state["chat"]:
        st.divider()
        st.subheader("Conversation")
        icon_b64 = _img_to_b64(ICON_PATH)
        for msg in st.session_state["chat"]:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='text-align:right;margin:10px 0;'>
                        <div style='display:inline-block;background:#e6f7ff;padding:12px;border-radius:12px;max-width:75%;'>
                            <b>You:</b> {msg['content']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                _show_bubble(msg["content"], icon_b64 or "")

        # Download chat history (unchanged, visible under chat)
    if st.session_state["chat"]:
        chat_text = ""
        for m in st.session_state["chat"]:
            who = "You" if m["role"] == "user" else "Assistant"
            chat_text += f"{who}: {m['content']}\n\n"
        b64 = base64.b64encode(chat_text.encode()).decode()
        st.markdown(
            f'<a href="data:file/txt;base64,{b64}" download="cliniq_chat_history.txt">📥 Download Chat History</a>',
            unsafe_allow_html=True,
        )

    # ----- SOP Retrieval & Guidance -----
    st.divider()
    docs = load_documents(DATA_DIR)
    vectorizer, matrix, sources, corpus = build_index(docs)

    sop_query = f"{scenario} {ROLES[role_label]} SOP responsibilities documentation reporting"
    st.subheader("🔎 Search evidence from SOPs")
    st.write("Query:", sop_query)

    k = st.session_state.get("k_slider", 5)
    snippets = retrieve(sop_query, vectorizer, matrix, sources, corpus, k=k)
    if snippets:
        for i, s in enumerate(snippets, 1):
            with st.expander(f"{i}. {s.source}  (relevance {s.score:.2f})", expanded=(i == 1)):
                st.text(s.text if s.text else "(no text)")
    else:
        st.info("No SOP files found. Add .txt or .pdf files under `data/sops`.")

    st.divider()
    if st.button("Generate CLINI-Q Guidance", type="primary", key="guidance_btn"):
        plan = compose_guidance(role_label, scenario, answers, snippets)
        st.success("Draft guidance generated.")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Steps")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for i, step in enumerate(plan.get("steps", []), 1):
                st.markdown(f"**{i}.** {step}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("### SOP Citations")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("; ".join(plan.get("citations", [])) or "-")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Compliance")
        for item in plan.get("compliance", []):
            st.markdown(f"- {item}")
        st.markdown(f"> {plan.get('disclaimer', FINAL_VERIFICATION_LINE)}")

    st.caption("© 2025 CLINIQ ⚖️Disclaimer: Demo tool only. No PHI/PII 📚 For official guidance, refer to your office policies.⚖️")

# -------- entrypoint --------
if __name__ == "__main__":
    main()
