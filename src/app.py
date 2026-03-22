"""
LendSynthetix — AI Credit Decision Engine
Full-stack: RAG + Multi-Agent War Room + Sentiment + Persistent Memory + Agent Chat
"""

import streamlit as st
import sys, os, json, time, tempfile, importlib.util
from pathlib import Path
from datetime import datetime

# Path setup: works regardless of which subfolder app.py is placed in
_BASE = Path(__file__).resolve().parent
for _p in [_BASE, _BASE.parent, _BASE.parent.parent]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
os.chdir(_BASE)

def _ensure_package(pkg_name):
    """Force-load a local sub-package from _BASE into sys.modules."""
    if pkg_name in sys.modules:
        return
    pkg_dir   = _BASE / pkg_name
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()
    spec = importlib.util.spec_from_file_location(
        pkg_name, str(init_file),
        submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)

_ensure_package("memory")
_ensure_package("sample_data")

from war_room       import run_war_room, LoanDecision
from rag_pipeline   import FinancialDocumentParser, parse_text_application
from sentiment      import analyze_sentiment, sentiment_color, tone_color
from agent_chat     import ask_agent, get_suggested_questions, AGENT_PERSONAS
from memory.db      import save_run, save_chat_message, get_chat_history, get_all_runs, get_decision_stats, get_run_by_id
from sample_data.sample_inputs import ALL_SAMPLES


# ══════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════

st.set_page_config(
    page_title="LendSynthetix",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load API keys from st.secrets (Streamlit Cloud) or .env at startup
# This ensures keys are available before any module tries to use them
try:
    if hasattr(st, "secrets"):
        if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        if "GOOGLE_API_KEY" in st.secrets and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# Suppress noisy warnings
import warnings, logging, os as _os
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Silence HuggingFace tokenizers parallelism warning
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import html as _html

def _safe(s: str) -> str:
    """Escape user input before injecting into HTML strings."""
    return _html.escape(str(s))

@st.cache_data
def _load_css(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

css_path = Path(__file__).parent / "static" / "styles.css"
if css_path.exists():
    css_content = _load_css(str(css_path))
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Force sidebar visible — works on all Streamlit versions
st.markdown("""
<style>
/* Force sidebar visible regardless of theme */
[data-testid="stSidebar"] {
    visibility: visible !important;
    opacity: 1 !important;
    width: auto !important;
    min-width: 244px !important;
    transform: none !important;
}
[data-testid="stSidebarNav"] {
    visibility: visible !important;
}
/* Keep collapse button functional */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# Force sidebar visible — hide the header bar cleanly while keeping toggle
st.markdown("""
<style>
/* Hide header bar content but keep the sidebar toggle button */
header[data-testid="stHeader"] {
    height: 0px !important;
    min-height: 0px !important;
    overflow: hidden !important;
}
/* Keep the sidebar toggle button visible and clickable */
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}
/* Remove top padding since header is collapsed */
.block-container {
    padding-top: 0.5rem !important;
}
/* Ensure sidebar is not zero-width */
section[data-testid="stSidebar"] {
    min-width: 244px !important;
}
</style>
""", unsafe_allow_html=True)




# ══════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════

def decision_css(d: str) -> str:
    return {"APPROVED": "approved", "REJECTED": "rejected",
            "ESCALATED": "escalated"}.get(d, "pending")

def decision_icon(d: str) -> str:
    return {"APPROVED": "✓", "REJECTED": "✕", "ESCALATED": "!", "PENDING": "?"}.get(d, "?")

def clean_decision(d) -> str:
    return str(d).replace("LoanDecision.", "")

def raroc_fmt(r, veto: bool = False) -> str:
    if r is None:
        return "N/A"
    if r == 0.0 and veto:
        return "0.0000 (veto)"
    return f"{r:.4f}"


def generate_run_report(run: dict) -> str:
    """Generate a plain-text summary report for a historical run."""
    import json as _json
    flags = []
    try:
        flags = _json.loads(run.get("critical_flags", "[]"))
    except Exception:
        pass

    history = []
    try:
        history = _json.loads(run.get("debate_history", "[]"))
    except Exception:
        pass

    veto   = bool(run.get("compliance_veto"))
    raroc  = run.get("raroc_score")
    dec    = run.get("final_decision", "UNKNOWN")

    lines = [
        "=" * 60,
        "LENDSYNTHETIX — LOAN EVALUATION REPORT",
        "=" * 60,
        f"Company:          {run.get('company_name', 'Unknown')}",
        f"Evaluation Date:  {run.get('timestamp', 'N/A')}",
        f"Run ID:           #{run.get('id', '?')}",
        f"Model Used:       {run.get('model_used', 'N/A')}",
        "",
        "── DECISION SUMMARY ──────────────────────────────────",
        f"Final Decision:   {dec}",
        f"RAROC Score:      {f'{raroc:.4f}' if raroc else 'N/A'}",
        f"Compliance Veto:  {'ISSUED' if veto else 'Not issued'}",
        f"Debate Rounds:    {run.get('debate_rounds', 'N/A')}",
        f"Flags Raised:     {', '.join(flags) if flags else 'None'}",
        "",
        "── SENTIMENT ─────────────────────────────────────────",
        f"Sentiment Score:  {run.get('sentiment_score', 'N/A')}",
        f"Sentiment Label:  {run.get('sentiment_label', 'N/A')}",
        "",
        "── RAROC THRESHOLDS ──────────────────────────────────",
        "  > 0.15  → Approve",
        "  0.08 – 0.15 → Escalate",
        "  < 0.08  → Reject",
        f"  Result: {'ABOVE' if raroc and raroc >= 0.15 else 'IN ESCALATION ZONE' if raroc and raroc >= 0.08 else 'BELOW'} threshold",
        "",
        "── DECISION MEMO ─────────────────────────────────────",
        run.get("decision_memo", "No memo available."),
        "",
        "── DEBATE HISTORY ────────────────────────────────────",
    ]
    for entry in history:
        lines.append(str(entry))
        lines.append("")

    lines.append("=" * 60)
    lines.append("Generated by LendSynthetix v2.0")
    lines.append("=" * 60)
    return "\n".join(lines)


# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════

st.markdown("""
<div class="ls-header">
  <div>
    <div class="ls-wordmark">Lend<span>Synthetix</span></div>
    <div class="ls-sub">Commercial Credit Decision Engine &nbsp;&nbsp;|&nbsp;&nbsp; Internal Use Only</div>
  </div>
  <div class="ls-badges">
    <span class="ls-badge ls-badge-green"><span class="ls-dot"></span>System Online</span>
    <span class="ls-badge ls-badge-blue">v2.0</span>
    <span class="ls-badge">LangGraph &middot; Groq &middot; LlamaIndex</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════

with st.sidebar:
    st.markdown("#### API Config")
    groq_key = st.text_input("Groq API Key", type="password",
                              value=os.environ.get("GROQ_API_KEY", ""),
                              help="Get free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    google_key = st.text_input("Google API Key (for PDF)", type="password",
                               value=os.environ.get("GOOGLE_API_KEY", ""),
                               help="Required for PDF upload. Get at console.cloud.google.com")
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key

    model = st.selectbox("LLM Model", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "mistral-saba-24b",
    ])
    os.environ["LS_MODEL"] = model

    st.markdown("---")
    st.markdown("#### Agent Config")
    # Agent weight sliders — informational display only
    st.slider("Sales Agent Weight",  0.0, 1.0, 0.35, 0.05, disabled=True)
    st.slider("Risk Agent Weight",   0.0, 1.0, 0.45, 0.05, disabled=True)
    veto_on      = st.toggle("Compliance Veto Power", value=True)
    max_rounds   = st.number_input("Max Debate Rounds", 1, 5, 3)
    run_sentiment = st.toggle("Sentiment Analysis", value=True)

    st.markdown("---")
    st.markdown("#### Sample Datasets")
    sample = st.selectbox("Load sample", [
        "— Select —",
        "Tech Startup (Sales vs Risk)",
        "Manufacturing (Compliance Veto)",
        "Retail Chain (Borderline)",
    ])

    st.markdown("---")
    if st.button("Clear Session", use_container_width=True):
        for k in ["war_room_result", "run_war_room", "loan_text",
                  "company_name", "sentiment_result", "current_run_id",
                  "chat_history_ui"]:
            st.session_state.pop(k, None)
        st.rerun()


# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Application",
    "War Room",
    "Analytics",
    "Agent Q&A",
    "Decision Memo",
    "History",
])


# ──────────────────────────────────────────
# TAB 1 — INPUT
# ──────────────────────────────────────────

with tab1:
    st.markdown("### Application Input")

    mode = st.radio("Input method", ["Paste Text", "Upload PDF", "Sample Dataset"], horizontal=True)

    loan_text    = ""
    company_name = ""

    if mode == "Upload PDF":
        groq_key_ok = bool(os.environ.get("GROQ_API_KEY", "").strip())
        if not groq_key_ok:
            st.error("❌ Groq API key required for PDF parsing. Enter it in the sidebar first.")
            pdf = None
        else:
            pdf = st.file_uploader("Upload financial statement PDF", type=["pdf"])

        if pdf:
            # ── Cache PDF bytes in session_state ──────────────────
            # Streamlit reruns on every interaction; pdf.read() returns
            # empty bytes after the first read because the pointer is at EOF.
            # Store raw bytes once, reuse on subsequent reruns.
            pdf_cache_key = f"pdf_bytes_{pdf.name}_{pdf.size}"
            if pdf_cache_key not in st.session_state:
                st.session_state[pdf_cache_key] = pdf.read()

            pdf_bytes = st.session_state[pdf_cache_key]

            # ── Parse only once per file ───────────────────────────
            parse_cache_key = f"pdf_parsed_{pdf.name}_{pdf.size}"
            if parse_cache_key not in st.session_state:
                with st.spinner("Parsing document via RAG pipeline..."):
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(pdf_bytes)
                            tmp_path = tmp.name

                        parser   = FinancialDocumentParser(tmp_path)
                        metrics  = parser.extract_all_metrics()
                        summary  = parser.to_summary_text()

                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                        st.session_state[parse_cache_key] = {
                            "loan_text":    summary,
                            "company_name": metrics.get("company_name", "Unknown"),
                            "n_tables":     metrics.get("_tables_found", 0),
                        }
                        st.success(
                            f"Parsed — {len(summary):,} characters extracted"
                            + (f", {metrics.get('_tables_found', 0)} tables found"
                               if metrics.get("_tables_found") else "")
                        )
                    except Exception as e:
                        st.error(f"PDF parsing error: {e}")

            if parse_cache_key in st.session_state:
                cached = st.session_state[parse_cache_key]
                loan_text    = cached["loan_text"]
                company_name = cached["company_name"]

    elif mode == "Paste Text":
        company_name = st.text_input("Company Name", placeholder="e.g. Acme Pvt Ltd")

        # Voice input component
        st.markdown("""
        <div style="margin-bottom:4px;display:flex;align-items:center;gap:10px">
          <span style="font-size:12px;color:var(--text-muted);">
            Voice input (Chrome only) — or paste text directly below
          </span>
          <span style="font-size:11px;background:rgba(37,99,235,0.08);
                       color:#60a5fa;border:1px solid rgba(37,99,235,0.2);
                       padding:2px 8px;border-radius:4px">Chrome required</span>
        </div>
        """, unsafe_allow_html=True)

        # Voice input via Web Speech API
        st.components.v1.html("""
        <div style="margin-bottom:8px">
          <button id="voiceBtn" onclick="toggleVoice()" style="
            padding:6px 14px; border-radius:20px; font-size:12px; cursor:pointer;
            border:1px solid rgba(37,99,235,0.4); background:rgba(37,99,235,0.08);
            color:#60a5fa; font-family:monospace; transition:all 0.2s;">
            🎙️ Start Voice Input
          </button>
          <span id="voiceStatus" style="font-size:11px;color:#4a6a8a;margin-left:10px;"></span>
          <div id="transcript" style="
            margin-top:8px; padding:8px 12px; border-radius:6px;
            background:rgba(13,27,42,0.8); border:1px solid rgba(30,66,102,0.5);
            font-size:12px; color:#7a9ab8; font-family:monospace;
            min-height:32px; display:none; white-space:pre-wrap;">
          </div>
        </div>
        <script>
        let recognition, isRecording = false, fullTranscript = '';
        function toggleVoice() {
          if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            document.getElementById('voiceStatus').textContent = 'Voice not supported in this browser.';
            return;
          }
          if (isRecording) {
            recognition.stop(); isRecording = false;
            document.getElementById('voiceBtn').textContent = 'Start Voice Input';
            document.getElementById('voiceBtn').style.color = '#60a5fa';
            document.getElementById('voiceStatus').textContent = 'Transcription complete — copy text below.';
          } else {
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SR();
            recognition.continuous = true; recognition.interimResults = true;
            recognition.lang = 'en-IN';
            recognition.onresult = e => {
              let interim = '';
              for (let i = e.resultIndex; i < e.results.length; i++) {
                if (e.results[i].isFinal) fullTranscript += e.results[i][0].transcript + ' ';
                else interim += e.results[i][0].transcript;
              }
              const box = document.getElementById('transcript');
              box.style.display = 'block';
              box.textContent = fullTranscript + interim;
            };
            recognition.start(); isRecording = true;
            document.getElementById('voiceBtn').textContent = 'Stop Recording';
            document.getElementById('voiceBtn').style.color = '#f87171';
            document.getElementById('voiceStatus').textContent = 'Recording...';
          }
        }
        </script>
        """, height=120)

        loan_text = st.text_area("Paste loan application details",
                                  height=260,
                                  placeholder="Paste the full application text here, or use voice input above...")

    elif mode == "Sample Dataset":
        key_map = {
            "Tech Startup (Sales vs Risk)":       "tech_startup",
            "Manufacturing (Compliance Veto)":    "manufacturing",
            "Retail Chain (Borderline)":          "retail",
        }
        key = key_map.get(sample)
        if key:
            data         = ALL_SAMPLES[key]
            company_name = data["company"]
            loan_text    = data["text"]
            st.info(f"{company_name} — {data['expected_outcome']}")

    if loan_text:
        with st.expander("View extracted data"):
            st.code(loan_text, language="text")

        st.markdown("---")

        # ── Show processing banner if war room is running ─────────────
        if st.session_state.get("run_war_room") and not st.session_state.get("war_room_result"):
            _running_company = st.session_state.get("company_name", "...")
            st.markdown(f"""
            <div style="background:rgba(37,99,235,0.10);border:2px solid rgba(37,99,235,0.45);
                         border-radius:12px;padding:18px 20px;margin-bottom:12px">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
                <div style="width:10px;height:10px;border-radius:50%;background:#3b82f6;
                             flex-shrink:0;animation:pulse-dot 1.2s ease-in-out infinite"></div>
                <span style="font-size:14px;font-weight:600;color:#e8f0f7">
                  War Room running — {_safe(_running_company)}</span>
              </div>
              <div style="font-size:12px;color:#7a9ab8;margin-bottom:12px">
                Agents are currently debating this application.
                Click the tab below to watch the live debate.
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
                <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.2);
                             border-radius:8px;padding:8px 12px;font-size:11px;color:#60a5fa">
                  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                                background:#2563eb;margin-right:6px;vertical-align:middle;
                                animation:pulse-dot 1.2s ease-in-out infinite"></span>
                  Sales Agent
                </div>
                <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                             border-radius:8px;padding:8px 12px;font-size:11px;color:#f59e0b">
                  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                                background:#f59e0b;margin-right:6px;vertical-align:middle;
                                animation:pulse-dot 1.4s ease-in-out infinite"></span>
                  Risk Agent
                </div>
                <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                             border-radius:8px;padding:8px 12px;font-size:11px;color:#10b981">
                  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                                background:#10b981;margin-right:6px;vertical-align:middle;
                                animation:pulse-dot 1.6s ease-in-out infinite"></span>
                  Compliance
                </div>
              </div>
              <div style="margin-top:10px;font-size:11px;color:#4a6a8a;text-align:center">
                ↑ Click the <strong style="color:#60a5fa">War Room</strong> tab above
                to watch live · page auto-refreshes on completion
              </div>
            </div>
            <style>
            @keyframes pulse-dot {{
              0%,100% {{ opacity:1; box-shadow:0 0 0 0 rgba(59,130,246,0.4); }}
              50%      {{ opacity:.7; box-shadow:0 0 0 5px rgba(59,130,246,0); }}
            }}
            </style>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption(f"Company: **{company_name}** &nbsp;|&nbsp; {len(loan_text):,} chars")
        with col2:
            _is_running = (
                st.session_state.get("run_war_room")
                and not st.session_state.get("war_room_result")
            )
            if _is_running:
                st.button("Processing...", disabled=True, use_container_width=True)
            elif st.button("Launch Engine", type="primary", use_container_width=True):
                st.session_state["loan_text"]     = loan_text
                st.session_state["company_name"]  = company_name
                st.session_state["run_war_room"]  = True
                st.session_state.pop("war_room_result", None)
                st.session_state.pop("current_run_id", None)
                st.session_state.pop("chat_history_ui", None)
                st.session_state.pop("sentiment_result", None)
                for _agent in ["Sales Agent", "Risk Agent", "Compliance Agent", "Moderator"]:
                    st.session_state.pop(f"chat_ui_{_agent}", None)
                st.rerun()


# ──────────────────────────────────────────
# TAB 2 — DEBATE CHAMBER
# ──────────────────────────────────────────

with tab2:
    if not st.session_state.get("run_war_room"):
        st.info("Submit a loan application in the Input tab to begin the war room debate.")
    else:
        loan_text    = st.session_state.get("loan_text", "")
        company_name = st.session_state.get("company_name", "Unknown")

        # Agent info cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="agent-card sales">
                <div class="agent-label">Sales Agent</div>
                <div class="agent-role">Relationship Manager</div>
                <div class="agent-desc">Champions growth story, recurring revenue & client relationships.</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="agent-card risk">
                <div class="agent-label">Risk Agent</div>
                <div class="agent-role">Credit Underwriter</div>
                <div class="agent-desc">Evaluates DSCR, debt ratios, collateral & repayment capacity.</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="agent-card compliance">
                <div class="agent-label">Compliance Agent</div>
                <div class="agent-role">Regulatory Auditor — Veto Authority</div>
                <div class="agent-desc">Enforces KYC/AML, RBI guidelines. Holds absolute veto power.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Run war room if not done
        if "war_room_result" not in st.session_state:

            # ── Live status display ────────────────────────────────
            status_box = st.empty()
            progress   = st.progress(0)
            timer_box  = st.empty()

            # Show live timer widget immediately using HTML component
            st.components.v1.html("""
            <style>
            body{margin:0;background:transparent}
            #live-box{
                background:#0d1b2a;border:1px solid rgba(37,99,235,0.3);
                border-radius:10px;padding:16px 20px;font-family:monospace;
            }
            .row{display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:10px}
            .badge{background:rgba(37,99,235,0.15);color:#60a5fa;
                    padding:3px 10px;border-radius:4px;font-size:11px}
            #timer{font-size:22px;font-weight:600;color:#2563eb}
            .agents{display:grid;grid-template-columns:1fr 1fr 1fr;
                     gap:8px;margin-top:12px}
            .agent{padding:8px 12px;border-radius:6px;font-size:11px;
                    border:1px solid transparent}
            .agent.sales{background:rgba(37,99,235,0.08);
                          border-color:rgba(37,99,235,0.2);color:#60a5fa}
            .agent.risk{background:rgba(245,158,11,0.08);
                         border-color:rgba(245,158,11,0.2);color:#f59e0b}
            .agent.comp{background:rgba(16,185,129,0.08);
                         border-color:rgba(16,185,129,0.2);color:#10b981}
            .dot{display:inline-block;width:6px;height:6px;border-radius:50%;
                  margin-right:6px;vertical-align:middle}
            @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
            .pulse{animation:pulse 1.4s ease-in-out infinite}
            </style>
            <div id="live-box">
              <div class="row">
                <span style="color:#e8f0f7;font-size:13px;font-weight:600">
                  WAR ROOM PROCESSING</span>
                <span class="badge pulse" id="status-label">Running...</span>
              </div>
              <div style="color:#4a6a8a;font-size:11px;margin-bottom:12px">
                Switch to the <strong style="color:#60a5fa">War Room</strong>
                tab to watch the live agent debate
              </div>
              <div id="timer">0s</div>
              <div style="color:#4a6a8a;font-size:10px;margin-top:2px">
                elapsed
              </div>
              <div class="agents">
                <div class="agent sales">
                  <span class="dot" style="background:#2563eb"></span>Sales Agent
                </div>
                <div class="agent risk">
                  <span class="dot" style="background:#f59e0b"></span>Risk Agent
                </div>
                <div class="agent comp">
                  <span class="dot" style="background:#10b981"></span>Compliance
                </div>
              </div>
            </div>
            <script>
            var start = Date.now();
            setInterval(function(){
                var sec = Math.floor((Date.now()-start)/1000);
                document.getElementById('timer').textContent = sec + 's';
            }, 1000);
            </script>
            """, height=190)

            STEPS = [
                (10, "Initializing",     "Setting up agent state machine"),
                (30, "Agents debating",  "Sales · Risk · Compliance reviewing application..."),
                (75, "Computing RAROC",  "Moderator weighing arguments and scoring..."),
                (90, "Writing memo",     "Drafting official audit decision memo..."),
                (97, "Sentiment",        "Scoring borrower tone..."),
            ]

            def show_status(step_idx, extra=""):
                pct, title, subtitle = STEPS[step_idx]
                elapsed = round(time.time() - _start_ts, 1)
                progress.progress(pct)
                status_box.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid rgba(37,99,235,0.35);
                             border-radius:12px;padding:20px 24px;margin-bottom:4px">

                  <!-- Header row -->
                  <div style="display:flex;align-items:center;
                               justify-content:space-between;margin-bottom:14px">
                    <div style="display:flex;align-items:center;gap:10px">
                      <div style="width:8px;height:8px;border-radius:50%;
                                   background:#3b82f6;flex-shrink:0;
                                   box-shadow:0 0 6px #3b82f6;
                                   animation:glow 1s ease-in-out infinite"></div>
                      <span style="font-size:12px;font-weight:600;color:#e8f0f7;
                                    letter-spacing:.03em">WAR ROOM IN PROGRESS</span>
                    </div>
                    <span style="font-family:monospace;font-size:12px;
                                  color:#4a6a8a;background:#112236;
                                  padding:3px 10px;border-radius:4px">
                      {elapsed}s elapsed
                    </span>
                  </div>

                  <!-- Current step -->
                  <div style="margin-bottom:14px">
                    <div style="font-size:15px;font-weight:600;
                                 color:#e8f0f7;margin-bottom:3px">{title}</div>
                    <div style="font-size:12px;color:#7a9ab8">
                      {subtitle}{" — " + extra if extra else ""}
                    </div>
                  </div>

                  <!-- Progress bar with percentage -->
                  <div style="display:flex;align-items:center;gap:10px">
                    <div style="flex:1;height:6px;background:#112236;
                                 border-radius:3px;overflow:hidden">
                      <div style="height:100%;width:{pct}%;background:#2563eb;
                                   border-radius:3px;transition:width .5s ease"></div>
                    </div>
                    <span style="font-family:monospace;font-size:11px;
                                  color:#2563eb;min-width:32px;text-align:right">
                      {pct}%
                    </span>
                  </div>

                  <!-- Steps indicator -->
                  <div style="display:flex;gap:6px;margin-top:12px">
                    {"".join([
                        '<div style="height:3px;flex:1;border-radius:2px;background:' + ("#2563eb" if i <= step_idx else "#112236") + '"></div>'
                        for i in range(len(STEPS))
                    ])}
                  </div>

                  <!-- Step labels -->
                  <div style="display:flex;justify-content:space-between;
                               margin-top:5px;font-size:9px;color:#4a6a8a">
                    <span>Sales</span>
                    <span>Risk</span>
                    <span>Compliance</span>
                    <span>Debate</span>
                    <span>Moderator</span>
                    <span>Memo</span>
                    <span>Sentiment</span>
                  </div>
                </div>
                <style>
                @keyframes glow {{
                  0%,100% {{ opacity:1; box-shadow:0 0 6px #3b82f6; }}
                  50%      {{ opacity:.6; box-shadow:0 0 2px #3b82f6; }}
                }}
                </style>
                """, unsafe_allow_html=True)

                timer_box.markdown(
                    '<div style="font-size:11px;color:#4a6a8a;'
                    'text-align:center;margin-top:4px">'
                    'Switch to the <strong style="color:#60a5fa">War Room</strong> tab '
                    'to watch live · page auto-refreshes on completion</div>',
                    unsafe_allow_html=True
                )

            _start_ts = time.time()
            show_status(0)
            show_status(1, company_name)

            start = time.time()
            try:
                with st.spinner("War room in progress..."):
                    result = run_war_room(
                        loan_text,
                        company_name,
                        max_rounds=int(max_rounds),
                        veto_enabled=bool(veto_on),
                    )

                st.session_state["war_room_result"] = result
                elapsed = round(time.time() - start, 1)
                st.session_state["last_latency"]    = f"{elapsed}s"
                st.session_state["run_count"]       = st.session_state.get("run_count", 0) + 1

                show_status(2, f"RAROC computed in {elapsed}s")

                # Sentiment — only if API key is set
                sentiment_result = None
                _key_ok = bool(os.environ.get("GROQ_API_KEY", "").strip())
                if run_sentiment and _key_ok:
                    show_status(4)
                    sentiment_result = analyze_sentiment(loan_text)
                    st.session_state["sentiment_result"] = sentiment_result

                # Persist to DB
                run_id = save_run(
                    result,
                    loan_text,
                    model=model,
                    sentiment_score=sentiment_result.get("overall_score") if sentiment_result else None,
                    sentiment_label=sentiment_result.get("label") if sentiment_result else None,
                )
                st.session_state["current_run_id"] = run_id

            except Exception as e:
                st.error(f"War Room Error: {str(e)}")
                st.exception(e)
                st.stop()

            # Done
            progress.progress(100)
            decision_done = str(result.get("final_decision","")).replace("LoanDecision.","")
            status_box.markdown(f"""
            <div style="background:rgba(16,185,129,0.06);
                         border:1px solid rgba(16,185,129,0.3);
                         border-radius:10px;padding:14px 20px;
                         display:flex;align-items:center;gap:12px">
              <div style="width:10px;height:10px;border-radius:50%;background:#10B981"></div>
              <div>
                <div style="font-size:13px;font-weight:600;color:#10B981">
                  Decision reached — {_safe(decision_done)}</div>
                <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
                  Total time: {st.session_state["last_latency"]} &nbsp;·&nbsp;
                  Switching to debate view...
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            status_box.empty()
            progress.empty()
            timer_box.empty()
            st.rerun()

        # Show debate
        result = st.session_state.get("war_room_result", {})
        if result:
            decision = clean_decision(result.get("final_decision", "PENDING"))
            css = decision_css(decision)

            # Quick result banner
            raroc          = result.get("raroc_score") or 0.0
            c_rounds       = result.get("consensus_rounds", result.get("debate_round", 1))
            c_flags        = len(result.get("critical_flags", []))
            c_resolved     = result.get("flags_resolved", 0)
            latency        = st.session_state.get("last_latency", "—")

            st.markdown(f"""
            <div class="verdict {css}" style="margin-bottom:12px">
                <span class="verdict-icon">{decision_icon(decision)}</span>
                <span class="verdict-label">{decision}</span>
                <span class="verdict-raroc">RAROC: {raroc_fmt(raroc)} &nbsp;·&nbsp; {latency}</span>
            </div>""", unsafe_allow_html=True)

            # Success metrics strip — directly from problem statement criteria
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;
                         margin-bottom:16px">
              <div style="background:var(--navy-800);border:1px solid var(--border-subtle);
                           border-radius:8px;padding:10px 14px">
                <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                             letter-spacing:.08em;margin-bottom:4px">Consensus Speed</div>
                <div style="font-size:13px;font-weight:600;color:var(--text-primary)">
                  Decision in Round {c_rounds}</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:2px">
                  {"Immediate consensus" if c_rounds == 1 else "Fast resolution" if c_rounds == 2 else "Full debate required"}
                </div>
              </div>
              <div style="background:var(--navy-800);border:1px solid var(--border-subtle);
                           border-radius:8px;padding:10px 14px">
                <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                             letter-spacing:.08em;margin-bottom:4px">Flag Resolution</div>
                <div style="font-size:13px;font-weight:600;color:var(--text-primary)">
                  {c_resolved}/{c_flags} flags resolved</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:2px">
                  {"All flags cleared" if c_flags > 0 and c_resolved == c_flags
                    else "Unresolved flags remain" if c_flags > c_resolved
                    else "No flags raised"}
                </div>
              </div>
              <div style="background:var(--navy-800);border:1px solid var(--border-subtle);
                           border-radius:8px;padding:10px 14px">
                <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                             letter-spacing:.08em;margin-bottom:4px">Explainability</div>
                <div style="font-size:13px;font-weight:600;color:#10B981">
                  Full audit trail</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:2px">
                  {len(result.get("debate_history", []))} debate entries logged
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Debate History")
            history = result.get("debate_history", [])
            current_round = 0

            for entry in history:
                if not entry.strip():
                    continue
                lines  = entry.strip().split("\n")
                header = lines[0]
                body   = "\n".join(lines[1:]).strip()

                if "[ROUND" in header:
                    try:
                        r = int(header.split("[ROUND ")[1].split("]")[0])
                        if r != current_round:
                            current_round = r
                            st.markdown(f"""<div class="section-title">Round {r}</div>""",
                                        unsafe_allow_html=True)
                    except Exception:
                        pass

                if "SALES AGENT" in header:
                    css_cls, label = "sales", "Sales Agent"
                elif "RISK AGENT" in header:
                    css_cls, label = "risk", "Risk Agent"
                elif "COMPLIANCE AGENT" in header:
                    css_cls, label = "compliance", "Compliance Agent"
                elif "MODERATOR" in header:
                    css_cls, label = "moderator", "Moderator — Final Resolution"
                    st.markdown('<div class="section-title">Final Resolution</div>',
                                unsafe_allow_html=True)
                else:
                    continue

                st.markdown(f"""
                <div class="debate-msg {css_cls}">
                    <div class="debate-sender {css_cls}">{label}</div>
                    <div class="debate-body">{body}</div>
                </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# TAB 3 — ANALYTICS
# ──────────────────────────────────────────

with tab3:
    result = st.session_state.get("war_room_result", {})

    if not result:
        st.info("Run the engine first to see analytics.")
    else:
        import re as _re

        # ── Extract all values first, no inline conditionals ──────
        decision         = clean_decision(result.get("final_decision", "PENDING"))
        raroc            = result.get("raroc_score") or 0.0
        veto             = result.get("compliance_veto", False)
        flags            = result.get("critical_flags", [])
        resolved_flags   = result.get("resolved_flags", [])
        consensus_rounds = result.get("consensus_rounds", result.get("debate_round", 1))
        flags_resolved   = result.get("flags_resolved", 0)
        sentiment        = st.session_state.get("sentiment_result") or {}

        dec_colors  = {"APPROVED": "#10B981", "REJECTED": "#EF4444",
                       "ESCALATED": "#F59E0B", "PENDING": "#6B7280"}
        dec_color   = dec_colors.get(decision, "#6B7280")

        # Pre-compute every conditional — no nested quotes inside f-strings
        flag_color      = "#EF4444" if flags else "#10B981"
        resolved_color  = "#10B981" if flags_resolved > 0 else "#6B7280"
        veto_color      = "#EF4444" if veto else "#10B981"
        veto_text       = "Issued" if veto else "Not issued"
        threshold_text  = "Approve" if raroc >= 0.15 else "Escalate" if raroc >= 0.08 else "Reject"
        delta           = raroc - 0.15
        delta_sign      = "+" if delta >= 0 else ""
        delta_col       = "#10B981" if delta >= 0 else "#EF4444"

        if consensus_rounds <= 1:
            speed_label, speed_color = "Round 1 — Immediate", "#10B981"
        elif consensus_rounds == 2:
            speed_label, speed_color = "Round 2 — Fast", "#10B981"
        else:
            speed_label, speed_color = f"Round {consensus_rounds} — Full debate", "#F59E0B"

        if veto:
            bar_color, bar_label = "#EF4444", "VETO — REJECTED"
        elif raroc >= 0.15:
            bar_color, bar_label = "#10B981", "APPROVED"
        elif raroc >= 0.08:
            bar_color, bar_label = "#F59E0B", "ESCALATED"
        else:
            bar_color, bar_label = "#EF4444", "REJECTED"

        bar_pct    = round(min(raroc / 0.35, 1.0) * 100, 1)
        delta_str  = f"{delta_sign}{delta:.4f}"
        delta_disp = "#10B981" if delta >= 0 else "#EF4444"

        # ── Verdict banner ────────────────────────────────────────
        css = decision_css(decision)
        st.markdown(
            f'<div class="verdict {css}">'
            f'<span class="verdict-icon">{decision_icon(decision)}</span>'
            f'<span class="verdict-label">LOAN {decision}</span>'
            f'<span class="verdict-raroc">RAROC {raroc_fmt(raroc, veto)}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── 5 KPI cards ───────────────────────────────────────────
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:12px 0">'
            f'<div style="background:var(--navy-800);border:1px solid var(--border-subtle);border-radius:12px;padding:14px">'
            f'<div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px">Decision</div>'
            f'<div style="font-size:15px;font-weight:600;color:{dec_color}">{decision}</div></div>'
            f'<div style="background:var(--navy-800);border:1px solid var(--border-subtle);border-radius:12px;padding:14px">'
            f'<div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px">RAROC Score</div>'
            f'<div style="font-size:20px;font-weight:600;color:{dec_color};font-family:monospace">{raroc_fmt(raroc, veto)}</div></div>'
            f'<div style="background:var(--navy-800);border:1px solid var(--border-subtle);border-radius:12px;padding:14px">'
            f'<div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px">Consensus Speed</div>'
            f'<div style="font-size:12px;font-weight:600;color:{speed_color};margin-top:3px">{speed_label}</div></div>'
            f'<div style="background:var(--navy-800);border:1px solid var(--border-subtle);border-radius:12px;padding:14px">'
            f'<div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px">Flags Raised</div>'
            f'<div style="font-size:20px;font-weight:600;color:{flag_color};font-family:monospace">{len(flags)}</div></div>'
            f'<div style="background:var(--navy-800);border:1px solid var(--border-subtle);border-radius:12px;padding:14px">'
            f'<div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px">Flags Resolved</div>'
            f'<div style="font-size:20px;font-weight:600;color:{resolved_color};font-family:monospace">{flags_resolved}</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Critical flags pills ───────────────────────────────────
        if flags:
            st.markdown("**Critical flags:**")
            pills = ""
            for f in flags:
                f_clean = _re.sub(r"[*_`]+", "", str(f)).strip()
                if not f_clean:
                    continue
                is_resolved = f in resolved_flags
                fc   = "#10B981" if is_resolved else "#EF4444"
                fbg  = "rgba(16,185,129,0.08)" if is_resolved else "rgba(239,68,68,0.08)"
                mark = " ✓" if is_resolved else ""
                pills += (
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:4px;'
                    f'font-size:11px;font-family:monospace;background:{fbg};'
                    f'color:{fc};border:1px solid {fc}44;margin:2px 4px 2px 0">'
                    f'{_html.escape(f_clean)}{mark}</span>'
                )
            st.markdown(pills, unsafe_allow_html=True)
            st.markdown("")

        st.markdown("---")

        # ── Two column layout ─────────────────────────────────────
        col_left, col_right = st.columns(2)

        # ── LEFT: RAROC bar + agent bars ──────────────────────────
        with col_left:
            st.markdown("**RAROC Analysis**")

            # Simple bar chart
            st.markdown(
                f'<div style="padding:8px 0 12px">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">'
                f'<span style="font-size:26px;font-weight:700;color:{bar_color};font-family:monospace">{raroc:.4f}</span>'
                f'<span style="font-size:12px;color:{delta_disp};font-family:monospace">{delta_str} vs 0.15</span>'
                f'<span style="font-size:11px;font-weight:600;color:{bar_color};background:{bar_color}18;'
                f'padding:3px 10px;border-radius:20px;border:1px solid {bar_color}44">{bar_label}</span>'
                f'</div>'
                f'<div style="height:16px;background:rgba(255,255,255,0.06);border-radius:6px;position:relative">'
                f'<div style="height:100%;width:{bar_pct}%;background:{bar_color};border-radius:6px;opacity:0.85"></div>'
                f'<div style="position:absolute;left:42.9%;top:-3px;width:2px;height:22px;background:rgba(255,255,255,0.6)"></div>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;margin-top:4px;font-size:10px;font-family:monospace">'
                f'<span style="color:#EF4444">0.00 Reject</span>'
                f'<span style="color:rgba(255,255,255,0.5)">0.15 hurdle</span>'
                f'<span style="color:#10B981">0.35 Approve</span>'
                f'</div></div>',
                unsafe_allow_html=True
            )

            st.markdown("**Agent score breakdown**")

            sales_w = 0 if veto else result.get("_sales_weight", 5)
            risk_w  = 0 if veto else result.get("_risk_weight", 5)
            comp_memo_upper = result.get("compliance_memo", "").upper()
            comp_w  = 10 if veto else (8 if "FLAGGED" in comp_memo_upper else 5)

            for ag_name, ag_score, ag_color in [
                ("Sales Agent", sales_w, "#2563EB"),
                ("Risk Agent",  risk_w,  "#F59E0B"),
                ("Compliance",  comp_w,  "#10B981"),
            ]:
                ag_pct = int(ag_score / 10 * 100)
                st.markdown(
                    f'<div style="margin-bottom:10px">'
                    f'<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">'
                    f'<span style="color:var(--text-secondary)">{ag_name}</span>'
                    f'<span style="color:var(--text-primary);font-weight:600;font-family:monospace">{ag_score}/10</span>'
                    f'</div>'
                    f'<div style="height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden">'
                    f'<div style="height:100%;width:{ag_pct}%;background:{ag_color};border-radius:4px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        # ── RIGHT: Thresholds + RAROC breakdown ───────────────────
        with col_right:
            st.markdown("**Decision thresholds**")

            thresholds = [
                ("RAROC > 0.15",      "Approve",     "#10B981", raroc >= 0.15 and not veto),
                ("RAROC 0.08–0.15",   "Escalate",    "#F59E0B", 0.08 <= raroc < 0.15 and not veto),
                ("RAROC < 0.08",      "Reject",      "#EF4444", raroc < 0.08 and not veto),
                ("Compliance Veto",   "Auto-Reject", "#EF4444", veto),
            ]
            for cond, outcome, chex, matched in thresholds:
                tbg     = chex + "18" if matched else "transparent"
                tborder = f"1px solid {chex}44" if matched else "1px solid var(--border-subtle)"
                tarrow  = " ← active" if matched else ""
                st.markdown(
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'padding:9px 12px;border-radius:8px;margin-bottom:6px;'
                    f'background:{tbg};border:{tborder}">'
                    f'<code style="font-size:12px;color:var(--text-secondary);background:transparent">{cond}</code>'
                    f'<span style="font-size:12px;font-weight:600;color:{chex}">{outcome}{tarrow}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            if veto:
                st.markdown(
                    '<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
                    'border-radius:8px;padding:10px 12px;margin-top:4px;'
                    'font-size:12px;color:#EF4444;font-weight:600">'
                    'COMPLIANCE VETO ISSUED — deal blocked</div>',
                    unsafe_allow_html=True
                )

            st.markdown("**RAROC breakdown**")
            st.markdown(
                f'<div style="background:var(--navy-800);border:1px solid var(--border-subtle);'
                f'border-radius:8px;padding:12px 14px;font-family:monospace">'
                f'<div style="font-size:10px;color:var(--text-muted);margin-bottom:8px">'
                f'(Net Interest Income - Expected Loss) / Economic Capital</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">'
                f'<div style="font-size:11px;color:var(--text-muted)">RAROC</div>'
                f'<div style="font-size:12px;font-weight:600;color:{dec_color};text-align:right">{raroc:.4f}</div>'
                f'<div style="font-size:11px;color:var(--text-muted)">Hurdle rate</div>'
                f'<div style="font-size:12px;color:var(--text-secondary);text-align:right">0.1500</div>'
                f'<div style="font-size:11px;color:var(--text-muted)">Delta</div>'
                f'<div style="font-size:12px;font-weight:600;color:{delta_col};text-align:right">{delta_sign}{delta:.4f}</div>'
                f'<div style="font-size:11px;color:var(--text-muted)">Result</div>'
                f'<div style="font-size:12px;font-weight:600;color:{dec_color};text-align:right">{threshold_text}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

            # ── Sentiment ─────────────────────────────────────────
            if sentiment:
                st.markdown("**Sentiment analysis**")
                s_color = sentiment_color(sentiment.get("label", "Neutral"))
                t_color = tone_color(sentiment.get("borrower_tone", "Stable"))

                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
                    f'<span style="font-size:12px;color:var(--text-secondary)">Overall tone</span>'
                    f'<span style="background:{s_color}22;color:{s_color};border:1px solid {s_color}44;'
                    f'padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600">'
                    f'{sentiment.get("label", "Neutral")}</span></div>'
                    f'<div style="font-size:12px;color:var(--text-muted);margin-bottom:10px">'
                    f'Borrower tone: <span style="color:{t_color};font-weight:600">'
                    f'{sentiment.get("borrower_tone", "Stable")}</span></div>',
                    unsafe_allow_html=True
                )

                for s_label, s_score, s_color_bar in [
                    ("Confidence",      sentiment.get("confidence_score", 0.5),      "#2563EB"),
                    ("Growth optimism", sentiment.get("growth_optimism_score", 0.5), "#10B981"),
                    ("Risk language",   sentiment.get("risk_language_score", 0.5),   "#EF4444"),
                ]:
                    s_pct = int(s_score * 100)
                    st.markdown(
                        f'<div style="margin-bottom:8px">'
                        f'<div style="display:flex;justify-content:space-between;font-size:11px;'
                        f'color:var(--text-muted);margin-bottom:3px">'
                        f'<span>{s_label}</span><span style="font-family:monospace">{s_pct}%</span></div>'
                        f'<div style="height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden">'
                        f'<div style="height:100%;width:{s_pct}%;background:{s_color_bar};border-radius:3px"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                pos_signals  = sentiment.get("key_positive_signals", [])
                risk_signals = sentiment.get("key_risk_signals", [])
                for sig in pos_signals:
                    st.markdown(
                        f'<div style="font-size:11px;padding:4px 10px;border-radius:5px;margin-bottom:3px;'
                        f'background:rgba(16,185,129,0.1);color:#10B981;font-family:monospace">+ {sig}</div>',
                        unsafe_allow_html=True
                    )
                for sig in risk_signals:
                    st.markdown(
                        f'<div style="font-size:11px;padding:4px 10px;border-radius:5px;margin-bottom:3px;'
                        f'background:rgba(239,68,68,0.1);color:#EF4444;font-family:monospace">! {sig}</div>',
                        unsafe_allow_html=True
                    )



# ──────────────────────────────────────────
# TAB 4 — AGENT CHAT
# ──────────────────────────────────────────

with tab4:
    result = st.session_state.get("war_room_result", {})

    if not result:
        st.info("Run the engine first to chat with the agents.")
    else:
        decision   = clean_decision(result.get("final_decision", "PENDING"))
        company    = _safe(result.get("company_name", "—"))
        raroc_disp = raroc_fmt(result.get("raroc_score"), result.get("compliance_veto", False))

        st.markdown("### Agent Q&A")
        st.caption("Each agent stays in character — ask follow-up questions about the decision.")

        # ── Agent selector buttons ────────────────────────────────
        agent_names    = list(AGENT_PERSONAS.keys())
        selected_agent = st.session_state.get("selected_agent", "Moderator")

        btn_cols = st.columns(4)
        for i, aname in enumerate(agent_names):
            with btn_cols[i]:
                if st.button(aname, key=f"agent_btn_{i}", use_container_width=True):
                    st.session_state["selected_agent"] = aname
                    selected_agent = aname
                    st.rerun()

        persona = AGENT_PERSONAS[selected_agent]
        st.markdown("---")

        # ── Chat header ───────────────────────────────────────────
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'padding:10px 14px;background:var(--navy-800);'
            f'border:1px solid var(--border-subtle);border-radius:10px;margin-bottom:12px">'
            f'<div style="width:10px;height:10px;border-radius:50%;'
            f'background:{persona["color"]};flex-shrink:0"></div>'
            f'<span style="font-weight:600;color:var(--text-primary)">'
            f'{persona["icon"]} {selected_agent}</span>'
            f'<span style="margin-left:auto;font-size:11px;color:var(--text-muted)">'
            f'Loan: {company} · Decision: {decision}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Chat history key for this agent ───────────────────────
        ui_key = f"chat_ui_{selected_agent}"
        if ui_key not in st.session_state:
            st.session_state[ui_key] = []

        # ── Welcome message ───────────────────────────────────────
        st.markdown(
            f'<div style="background:var(--navy-800);border-left:3px solid {persona["color"]};'
            f'border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:8px;'
            f'font-size:13px;color:var(--text-secondary)">'
            f'Hello. I\'m the <strong style="color:var(--text-primary)">{selected_agent}</strong>'
            f' on the <strong style="color:var(--text-primary)">{company}</strong> case. '
            f'The loan was <strong style="color:var(--text-primary)">{decision}</strong>'
            f' with RAROC {raroc_disp}. What would you like to know?'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Previous messages ─────────────────────────────────────
        for msg in st.session_state[ui_key]:
            is_user = msg["role"] == "user"
            align   = "flex-end" if is_user else "flex-start"
            bg      = "rgba(37,99,235,0.15)" if is_user else "var(--navy-800)"
            border  = "1px solid rgba(37,99,235,0.3)" if is_user else "1px solid var(--border-subtle)"
            st.markdown(
                f'<div style="display:flex;justify-content:{align};margin-bottom:8px">'
                f'<div style="max-width:80%;background:{bg};border:{border};'
                f'border-radius:10px;padding:10px 14px;font-size:13px;'
                f'color:var(--text-primary);line-height:1.5">'
                f'{_html.escape(str(msg["message"]))}'
                f'</div></div>',
                unsafe_allow_html=True
            )

        # ── Suggested questions ───────────────────────────────────
        suggestions = get_suggested_questions(selected_agent, decision)
        st.markdown("**Suggested questions:**")
        sugg_cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            with sugg_cols[i]:
                if st.button(q, key=f"sugg_{selected_agent}_{i}", use_container_width=True):
                    st.session_state[f"pending_{selected_agent}"] = q
                    st.rerun()

        # ── Input ─────────────────────────────────────────────────
        # Use a per-agent pending key so suggested questions work reliably
        pending_key = f"pending_{selected_agent}"
        default_val = st.session_state.pop(pending_key, "")

        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_q = st.text_input(
                "Question",
                value=default_val,
                placeholder=f"Ask the {selected_agent} anything about this decision...",
                key=f"chat_input_{selected_agent}_{len(st.session_state[ui_key])}",
                label_visibility="collapsed",
            )
        with col_btn:
            send_btn = st.button("Send ↵", type="primary",
                                 use_container_width=True,
                                 key=f"send_{selected_agent}")

        # ── Send logic ────────────────────────────────────────────
        if send_btn and user_q.strip():
            question = user_q.strip()

            st.session_state[ui_key].append({"role": "user", "message": question})

            run_id = st.session_state.get("current_run_id")
            if run_id:
                save_chat_message(run_id, "user", question)

            hist = [{"role": m["role"], "message": m["message"]}
                    for m in st.session_state[ui_key][:-1]]

            with st.spinner(f"{selected_agent} is thinking..."):
                response = ask_agent(selected_agent, question, result, hist)

            st.session_state[ui_key].append({"role": selected_agent, "message": response})

            if run_id:
                save_chat_message(run_id, selected_agent, response)

            st.rerun()



# ──────────────────────────────────────────
# TAB 5 — AUDIT MEMO
# ──────────────────────────────────────────

with tab5:
    result = st.session_state.get("war_room_result", {})

    if not result:
        st.info("Run the engine to generate an audit memo.")
    else:
        memo     = str(result.get("decision_memo", "No memo generated."))
        company  = _safe(st.session_state.get("company_name", "loan"))
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M")
        decision = clean_decision(result.get("final_decision", "PENDING"))

        st.markdown("### Official Decision Memo")

        # Memo header with decision badge
        css = decision_css(decision)
        colors = {"approved": "#10b981", "rejected": "#ef4444",
                  "escalated": "#f59e0b", "pending": "#6b7280"}
        badge_color = colors.get(css, "#6b7280")

        st.markdown(f"""
        <div class="memo-box">
            <div class="memo-header">
                decision_memo_{company.lower().replace(' ', '_')}.txt
                &nbsp;·&nbsp; {ts}
                &nbsp;·&nbsp;
                <span style="color:{badge_color};font-weight:600">{decision}</span>
                &nbsp;·&nbsp; RAROC: {raroc_fmt(result.get('raroc_score'))}
            </div>
            <div class="memo-body">{memo}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Memo (.txt)",
                data=memo,
                file_name=f"decision_memo_{company}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            sentiment = st.session_state.get("sentiment_result", {})
            export = json.dumps({
                "company":          company,
                "timestamp":        ts,
                "final_decision":   decision,
                "raroc_score":      result.get("raroc_score"),
                "compliance_veto":  result.get("compliance_veto"),
                "critical_flags":   result.get("critical_flags"),
                "debate_rounds":    result.get("debate_round"),
                "decision_memo":    memo,
                "sentiment":        sentiment,
                "debate_history":   [str(h) for h in result.get("debate_history", [])],
            }, indent=2)
            st.download_button(
                "Export Audit Log (.json)",
                data=export,
                file_name=f"audit_log_{company}.json",
                mime="application/json",
                use_container_width=True,
            )


# ──────────────────────────────────────────
# TAB 6 — HISTORY (redesigned)
# ──────────────────────────────────────────

with tab6:
    stats    = get_decision_stats()
    all_runs = get_all_runs(20)

    # Logic consistency check — show if same company appears multiple times
    company_counts = {}
    for run in all_runs:
        c = run["company_name"]
        if c not in company_counts:
            company_counts[c] = []
        company_counts[c].append(run["final_decision"])

    repeat_companies = {c: v for c, v in company_counts.items() if len(v) > 1}
    if repeat_companies:
        consistency_lines = []
        for company, decisions in repeat_companies.items():
            unique_decisions = set(decisions)
            consistent = len(unique_decisions) == 1
            consistency_lines.append(
                f"{company[:20]}: {', '.join(decisions)} — "
                f"{'Consistent' if consistent else 'Inconsistent'}"
            )
        with st.expander(f"Logic Consistency Check — {len(repeat_companies)} company(s) run multiple times"):
            for line in consistency_lines:
                color = "#10B981" if "Consistent" in line else "#F59E0B"
                st.markdown(
                    f'<div style="font-size:12px;font-family:var(--font-mono);'
                    f'color:{color};padding:3px 0">{line}</div>',
                    unsafe_allow_html=True
                )

    total     = stats["total"]
    approved  = stats["by_decision"].get("APPROVED", 0)
    rejected  = stats["by_decision"].get("REJECTED", 0)
    escalated = stats["by_decision"].get("ESCALATED", 0)
    avg_raroc = stats["avg_raroc"]
    avg_rounds= stats["avg_rounds"]

    # approval rate
    approval_rate = round((approved / total * 100), 1) if total > 0 else 0

    # ── Summary KPI row ───────────────────────────────────────────
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px">
      <div style="background:var(--navy-800);border:1px solid var(--border-subtle);
                  border-radius:12px;padding:16px 14px">
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                    letter-spacing:.09em;margin-bottom:8px">Total Evaluations</div>
        <div style="font-size:28px;font-weight:600;color:var(--text-primary);
                    font-family:var(--font-mono)">{total}</div>
      </div>
      <div style="background:var(--navy-800);border:1px solid rgba(16,185,129,0.25);
                  border-radius:12px;padding:16px 14px">
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                    letter-spacing:.09em;margin-bottom:8px">Approved</div>
        <div style="font-size:28px;font-weight:600;color:#10B981;
                    font-family:var(--font-mono)">{approved}</div>
      </div>
      <div style="background:var(--navy-800);border:1px solid rgba(239,68,68,0.25);
                  border-radius:12px;padding:16px 14px">
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                    letter-spacing:.09em;margin-bottom:8px">Rejected</div>
        <div style="font-size:28px;font-weight:600;color:#EF4444;
                    font-family:var(--font-mono)">{rejected}</div>
      </div>
      <div style="background:var(--navy-800);border:1px solid rgba(245,158,11,0.25);
                  border-radius:12px;padding:16px 14px">
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                    letter-spacing:.09em;margin-bottom:8px">Avg RAROC</div>
        <div style="font-size:28px;font-weight:600;color:#F59E0B;
                    font-family:var(--font-mono)">{avg_raroc:.4f}</div>
      </div>
      <div style="background:var(--navy-800);border:1px solid var(--border-subtle);
                  border-radius:12px;padding:16px 14px">
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                    letter-spacing:.09em;margin-bottom:8px">Approval Rate</div>
        <div style="font-size:28px;font-weight:600;color:var(--text-primary);
                    font-family:var(--font-mono)">{approval_rate}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if total == 0:
        st.markdown("""
        <div style="background:var(--navy-800);border:1px solid var(--border-subtle);
                    border-radius:12px;padding:48px;text-align:center">
          <div style="font-size:14px;color:var(--text-muted)">
            No evaluations yet. Submit a loan application to begin.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Decision distribution — full width, no trend chart ─────
        st.markdown("""
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                    letter-spacing:.09em;border-bottom:1px solid var(--border-subtle);
                    padding-bottom:8px;margin-bottom:12px">Decision Distribution</div>
        """, unsafe_allow_html=True)

        # Show as styled stat pills instead of chart
        dec_map = stats.get("by_decision", {})
        total_d = sum(dec_map.values()) or 1
        cmap = {"APPROVED":"#10B981","REJECTED":"#EF4444",
                "ESCALATED":"#F59E0B","PENDING":"#6B7280"}

        pills_html = '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">'
        for dec, cnt in dec_map.items():
            pct  = round(cnt / total_d * 100)
            col  = cmap.get(dec, "#6B7280")
            pills_html += f"""
            <div style="background:{col}12;border:1px solid {col}33;
                         border-radius:10px;padding:14px 20px;min-width:140px">
              <div style="font-size:24px;font-weight:600;color:{col};
                           font-family:var(--font-mono)">{cnt}</div>
              <div style="font-size:11px;font-weight:600;color:{col};
                           margin-top:2px">{dec}</div>
              <div style="font-size:11px;color:var(--text-muted);margin-top:1px">
                {pct}% of total</div>
              <div style="height:3px;background:{col}22;border-radius:2px;
                           margin-top:8px;overflow:hidden">
                <div style="height:100%;width:{pct}%;background:{col};
                             border-radius:2px"></div>
              </div>
            </div>"""
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

        # ── Runs table ────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;
                    letter-spacing:.09em;border-bottom:1px solid var(--border-subtle);
                    padding-bottom:8px;margin:20px 0 0">Recent Evaluations</div>
        """, unsafe_allow_html=True)

        # Table header
        st.markdown("""
        <div style="display:grid;grid-template-columns:44px 1fr 100px 110px 90px 60px 90px;
                    gap:8px;padding:10px 14px;
                    border-bottom:1px solid var(--border-default);
                    font-size:10px;text-transform:uppercase;letter-spacing:.08em;
                    color:var(--text-muted)">
          <div>#</div>
          <div>Company</div>
          <div>Timestamp</div>
          <div>Decision</div>
          <div>RAROC</div>
          <div>Veto</div>
          <div>Download</div>
        </div>""", unsafe_allow_html=True)

        dec_colors = {
            "APPROVED":  "#10B981",
            "REJECTED":  "#EF4444",
            "ESCALATED": "#F59E0B",
            "PENDING":   "#6B7280",
        }

        for i, run in enumerate(all_runs):
            dec      = run["final_decision"]
            d_color  = dec_colors.get(dec, "#6B7280")
            raroc_v  = f"{run['raroc_score']:.4f}" if run['raroc_score'] else "—"
            ts       = run['timestamp'][:16] if run['timestamp'] else "—"
            veto_val = "Yes" if run["compliance_veto"] else "—"
            veto_col = "#EF4444" if run["compliance_veto"] else "var(--text-muted)"
            row_bg   = "var(--navy-800)" if i % 2 == 0 else "transparent"

            # RAROC mini bar
            raroc_pct = min(100, int((run['raroc_score'] or 0) / 0.35 * 100))
            bar_color = "#10B981" if (run['raroc_score'] or 0) >= 0.15 else                         "#F59E0B" if (run['raroc_score'] or 0) >= 0.08 else "#EF4444"

            # Use columns to mix HTML row with Streamlit download button
            row_col, btn_col = st.columns([10, 2])
            with row_col:
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:44px 1fr 100px 110px 90px 60px;
                            gap:8px;padding:10px 14px;
                            background:{row_bg};
                            border-bottom:1px solid var(--border-subtle);
                            font-size:13px;align-items:center">
                  <div style="font-family:var(--font-mono);font-size:11px;
                              color:var(--text-muted)">#{run['id']}</div>
                  <div style="color:var(--text-primary);font-weight:500;
                              white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    {run['company_name'][:28]}
                  </div>
                  <div style="font-size:11px;color:var(--text-muted);
                              font-family:var(--font-mono)">{ts}</div>
                  <div>
                    <span style="background:{d_color}18;color:{d_color};
                                 border:1px solid {d_color}44;
                                 padding:3px 10px;border-radius:20px;
                                 font-size:11px;font-weight:600;
                                 font-family:var(--font-mono)">
                      {dec}
                    </span>
                  </div>
                  <div>
                    <div style="font-size:11px;color:var(--text-secondary);
                                font-family:var(--font-mono);margin-bottom:3px">
                      {raroc_v}
                    </div>
                    <div style="height:3px;background:var(--navy-700);
                                 border-radius:2px;overflow:hidden">
                      <div style="height:100%;width:{raroc_pct}%;
                                   background:{bar_color};border-radius:2px"></div>
                    </div>
                  </div>
                  <div style="font-size:11px;font-weight:600;color:{veto_col}">
                    {veto_val}
                  </div>
                </div>""", unsafe_allow_html=True)

            with btn_col:
                # Fetch full run data for download
                full_run = get_run_by_id(run["id"])
                if full_run:
                    report_txt = generate_run_report(full_run)
                    safe_name  = run["company_name"].lower().replace(" ", "_")[:20]
                    # Inject per-button CSS to show color feedback on click
                    st.markdown(f"""
                    <style>
                    div[data-testid="stDownloadButton"]:has(
                        button[data-testid="baseButton-secondary"][key="dl_{run['id']}"]
                    ) button {{
                        transition: all 0.2s !important;
                    }}
                    </style>""", unsafe_allow_html=True)
                    st.download_button(
                        label="Download",
                        data=report_txt,
                        file_name=f"report_{safe_name}_{run['id']}.txt",
                        mime="text/plain",
                        key=f"dl_{run['id']}",
                        use_container_width=True,
                        type="primary",
                    )


# ── Footer ────────────────────────────────
st.markdown("""
<div class="footer">
    LendSynthetix v2.0
    &nbsp;&nbsp;|&nbsp;&nbsp;
    AI Credit Decision Platform
    &nbsp;&nbsp;|&nbsp;&nbsp;
    LangGraph &middot; Groq &middot; LlamaIndex &middot; Streamlit
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Internal Use Only
</div>
""", unsafe_allow_html=True)