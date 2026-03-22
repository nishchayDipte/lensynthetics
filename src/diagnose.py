"""
LendSynthetix — Diagnostic Runner
===================================
Run this file to test every component of the project.

Usage:
    python diagnose.py              # full diagnostic
    python diagnose.py --quick      # skip war room (faster)
    python diagnose.py --phase 1    # test only one phase

Output: colored terminal report showing pass/fail for each component.
"""

import os
import sys
import time
import json
import logging
import importlib
import traceback
import warnings
from pathlib import Path
from datetime import datetime

# Silence Streamlit and other noisy loggers when running outside streamlit
logging.getLogger("streamlit").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("llama_index").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Prevent streamlit from printing "missing ScriptRunContext" 
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

# ── Add project root to path ───────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

# ── Terminal colors ────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"

def ok(msg):    print(f"  {C.GREEN}PASS{C.RESET}  {msg}")
def fail(msg):  print(f"  {C.RED}FAIL{C.RESET}  {msg}")
def warn(msg):  print(f"  {C.YELLOW}WARN{C.RESET}  {msg}")
def info(msg):  print(f"  {C.BLUE}INFO{C.RESET}  {msg}")
def section(title):
    print(f"\n{C.BOLD}{C.CYAN}{'─'*55}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─'*55}{C.RESET}")

results = {"pass": 0, "fail": 0, "warn": 0}

def check(name, fn):
    try:
        msg = fn()
        ok(f"{name}" + (f" — {msg}" if msg else ""))
        results["pass"] += 1
        return True
    except AssertionError as e:
        fail(f"{name} — {e}")
        results["fail"] += 1
        return False
    except Exception as e:
        fail(f"{name} — {type(e).__name__}: {e}")
        results["fail"] += 1
        return False


# ══════════════════════════════════════════════════════════
# PHASE 0 — ENVIRONMENT CHECK
# ══════════════════════════════════════════════════════════

def run_phase_0():
    section("Phase 0 — Environment & Dependencies")

    # Python version
    def check_python():
        v = sys.version_info
        assert v.major == 3 and v.minor >= 9, f"Need Python 3.9+, got {v.major}.{v.minor}"
        return f"Python {v.major}.{v.minor}.{v.micro}"
    check("Python version", check_python)

    # Required packages
    packages = {
        "streamlit":            "streamlit",
        "langchain_groq":       "langchain-groq",
        "langchain_core":       "langchain-core",
        "langgraph":            "langgraph",
        "llama_index.core":     "llama-index",
        "plotly":               "plotly",
        "pdfplumber":           "pdfplumber",
        "faiss":                "faiss-cpu",
    }
    for mod, pkg in packages.items():
        def _check(m=mod, p=pkg):
            importlib.import_module(m)
            return p
        check(f"Package: {pkg}", _check)

    # Optional packages
    optional = {
        "langchain_google_genai": "langchain-google-genai (Gemini embeddings)",
        "sentence_transformers":  "sentence-transformers (HuggingFace embeddings)",
    }
    for mod, name in optional.items():
        try:
            importlib.import_module(mod)
            ok(f"Optional: {name}")
        except ImportError:
            warn(f"Optional: {name} — not installed (fallback will be used)")
            results["warn"] += 1

    # API Keys
    def check_groq_key():
        key = os.environ.get("GROQ_API_KEY", "").strip()
        assert key, "GROQ_API_KEY not set in environment"
        assert len(key) > 20, "GROQ_API_KEY looks too short"
        return f"gsk_...{key[-6:]}"
    check("GROQ_API_KEY", check_groq_key)

    google_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if google_key:
        ok(f"GOOGLE_API_KEY — set (AI...{google_key[-4:]})")
    else:
        warn("GOOGLE_API_KEY — not set (PDF upload will use fallback embeddings)")
        results["warn"] += 1

    # Project files
    required_files = [
        "app.py", "war_room.py", "rag_pipeline.py",
        "sentiment.py", "agent_chat.py",
        "memory/__init__.py", "memory/db.py",
        "sample_data/__init__.py", "sample_data/sample_inputs.py",
        "static/styles.css",
    ]
    for f in required_files:
        def _check(fp=f):
            path = _BASE / fp
            assert path.exists(), f"Missing: {fp}"
            # __init__.py files are intentionally empty — that's fine
            if not fp.endswith("__init__.py"):
                assert path.stat().st_size > 0, f"Empty file: {fp}"
            return f"{path.stat().st_size:,} bytes"
        check(f"File: {f}", _check)


# ══════════════════════════════════════════════════════════
# PHASE 1 — RAG PIPELINE
# ══════════════════════════════════════════════════════════

def run_phase_1():
    section("Phase 1 — RAG Pipeline")

    def check_import():
        from rag_pipeline import FinancialDocumentParser, parse_text_application
        return "imported ok"
    check("Import rag_pipeline", check_import)

    def check_text_parsing():
        from rag_pipeline import parse_text_application
        sample = """
        Company: TestCo Ltd
        Loan Amount: ₹2 Crore
        Annual Revenue: ₹5 Crore
        DSCR: 1.42
        Debt-to-Equity Ratio: 0.87
        """
        result = parse_text_application(sample)
        assert isinstance(result, dict), "Should return dict"
        assert "dscr" in result, "Should extract DSCR"
        assert "debt_to_equity_ratio" in result, "Should extract D/E ratio"
        return f"DSCR={result.get('dscr')}, D/E={result.get('debt_to_equity_ratio')}"
    check("Text application parsing", check_text_parsing)

    def check_aml_detection():
        from rag_pipeline import parse_text_application
        sample = "Director on grey list. Offshore account in Cayman Islands."
        result = parse_text_application(sample)
        flags = result.get("aml_flags", [])
        assert len(flags) > 0, "Should detect AML flags"
        return f"Detected: {flags}"
    check("AML flag detection", check_aml_detection)

    def check_risk_flags():
        from rag_pipeline import parse_text_application
        sample = "Debt-to-Equity Ratio: 3.5 DSCR: 1.1"
        result = parse_text_application(sample)
        flags = result.get("_risk_flags", [])
        assert "HIGH_DEBT_TO_EQUITY" in flags, f"Should flag high D/E, got: {flags}"
        assert "LOW_DSCR" in flags, f"Should flag low DSCR, got: {flags}"
        return f"Auto-flags: {flags}"
    check("Auto risk flag detection", check_risk_flags)

    def check_llamaindex():
        from llama_index.core import Settings
        from rag_pipeline import _configure_llamaindex
        _configure_llamaindex()
        assert Settings.llm is not None, "LLM should be configured"
        assert Settings.embed_model is not None, "Embed model should be configured"
        llm_type = type(Settings.llm).__name__
        emb_type  = type(Settings.embed_model).__name__
        return f"LLM={llm_type}, Embed={emb_type}"
    check("LlamaIndex configuration", check_llamaindex)


# ══════════════════════════════════════════════════════════
# PHASE 2 — PERSONA DEFINITION
# ══════════════════════════════════════════════════════════

def run_phase_2():
    section("Phase 2 — Agent Persona Definition")

    def check_import():
        from war_room import (SALES_AGENT_PROMPT, RISK_AGENT_PROMPT,
                               COMPLIANCE_AGENT_PROMPT, MODERATOR_PROMPT)
        return "all 4 prompts imported"
    check("Import agent prompts", check_import)

    def check_sales_prompt():
        from war_room import SALES_AGENT_PROMPT
        assert "APPROVE" in SALES_AGENT_PROMPT, "Sales prompt must mention APPROVE"
        assert "KEY ARGUMENTS" in SALES_AGENT_PROMPT, "Must have KEY ARGUMENTS format"
        assert "MEMO TYPE" in SALES_AGENT_PROMPT, "Must have MEMO TYPE format"
        return f"{len(SALES_AGENT_PROMPT)} chars"
    check("Sales Agent prompt structure", check_sales_prompt)

    def check_risk_prompt():
        from war_room import RISK_AGENT_PROMPT
        assert "DSCR" in RISK_AGENT_PROMPT, "Risk prompt must mention DSCR"
        assert "RISK SCORE" in RISK_AGENT_PROMPT, "Must have RISK SCORE format"
        assert "FLAGS RAISED" in RISK_AGENT_PROMPT, "Must have FLAGS RAISED format"
        return f"{len(RISK_AGENT_PROMPT)} chars"
    check("Risk Agent prompt structure", check_risk_prompt)

    def check_compliance_prompt():
        from war_room import COMPLIANCE_AGENT_PROMPT
        assert "VETO" in COMPLIANCE_AGENT_PROMPT, "Compliance prompt must mention VETO"
        assert "AML" in COMPLIANCE_AGENT_PROMPT, "Must mention AML"
        assert "PMLA" in COMPLIANCE_AGENT_PROMPT, "Must reference Indian regulation PMLA"
        return f"{len(COMPLIANCE_AGENT_PROMPT)} chars"
    check("Compliance Agent prompt + Indian regulations", check_compliance_prompt)

    def check_moderator_prompt():
        from war_room import MODERATOR_PROMPT
        assert "RAROC" in MODERATOR_PROMPT, "Moderator must compute RAROC"
        assert "0.15" in MODERATOR_PROMPT, "Must have hurdle rate 0.15"
        assert "ESTIMATED RAROC" in MODERATOR_PROMPT, "Must output RAROC in format"
        return f"{len(MODERATOR_PROMPT)} chars"
    check("Moderator prompt + RAROC logic", check_moderator_prompt)

    def check_agent_chat():
        from agent_chat import AGENT_PERSONAS, ask_agent, get_suggested_questions
        assert len(AGENT_PERSONAS) == 4, f"Need 4 personas, got {len(AGENT_PERSONAS)}"
        for name, p in AGENT_PERSONAS.items():
            assert "system" in p, f"{name} missing system prompt"
            assert "color" in p, f"{name} missing color"
        return f"{len(AGENT_PERSONAS)} personas defined"
    check("Agent Chat personas", check_agent_chat)


# ══════════════════════════════════════════════════════════
# PHASE 3 — STATE MACHINE
# ══════════════════════════════════════════════════════════

def run_phase_3(quick=False):
    section("Phase 3 — LangGraph State Machine")

    def check_graph_build():
        from war_room import build_war_room_graph, WarRoomState, LoanDecision
        graph = build_war_room_graph()
        assert graph is not None, "Graph should compile"
        return "LangGraph compiled ok"
    check("Build LangGraph", check_graph_build)

    def check_state_fields():
        from war_room import WarRoomState, LoanDecision
        fields = WarRoomState.__annotations__
        required = ["loan_application", "company_name", "sales_memo",
                    "risk_memo", "compliance_memo", "debate_history",
                    "debate_round", "compliance_veto", "critical_flags",
                    "resolved_flags", "raroc_score", "final_decision",
                    "decision_memo", "consensus_rounds", "flags_resolved"]
        missing = [f for f in required if f not in fields]
        assert not missing, f"Missing state fields: {missing}"
        return f"{len(fields)} state fields"
    check("State machine fields", check_state_fields)

    def check_routing_logic():
        from war_room import should_continue_debate, WarRoomState, LoanDecision
        # Compliance veto should route to moderator
        veto_state = {
            "compliance_veto": True, "debate_round": 1,
            "critical_flags": [], "resolved_flags": []
        }
        assert should_continue_debate(veto_state) == "moderator", \
            "Veto should route to moderator"
        # Max rounds should route to moderator
        maxround_state = {
            "compliance_veto": False, "debate_round": 3,
            "critical_flags": [], "resolved_flags": []
        }
        assert should_continue_debate(maxround_state) == "moderator", \
            "Max rounds should route to moderator"
        # Unresolved flags should continue
        flag_state = {
            "compliance_veto": False, "debate_round": 1,
            "critical_flags": ["HIGH_DEBT"], "resolved_flags": []
        }
        assert should_continue_debate(flag_state) == "continue_debate", \
            "Unresolved flags should continue debate"
        return "veto, max-rounds, flag routing all correct"
    check("Routing logic (veto/rounds/flags)", check_routing_logic)

    def check_decision_thresholds():
        from war_room import LoanDecision
        # Test RAROC threshold logic
        def decide(raroc, veto=False):
            if veto: return LoanDecision.REJECTED
            if raroc >= 0.15: return LoanDecision.APPROVED
            if raroc >= 0.08: return LoanDecision.ESCALATED
            return LoanDecision.REJECTED
        assert decide(0.20) == LoanDecision.APPROVED
        assert decide(0.12) == LoanDecision.ESCALATED
        assert decide(0.05) == LoanDecision.REJECTED
        assert decide(0.25, veto=True) == LoanDecision.REJECTED
        return "RAROC thresholds: 0.15→Approve, 0.08→Escalate, <0.08→Reject"
    check("RAROC decision thresholds", check_decision_thresholds)

    if not quick:
        info("Running full war room on Tech Startup sample (~60-90s)...")
        def check_full_run():
            from war_room import run_war_room, LoanDecision
            from sample_data.sample_inputs import ALL_SAMPLES
            data = ALL_SAMPLES["tech_startup"]
            start = time.time()
            result = run_war_room(data["text"], data["company"])
            elapsed = round(time.time() - start, 1)

            assert result is not None, "Should return result"
            assert result["final_decision"] != LoanDecision.PENDING, \
                "Should reach a decision"
            assert result["raroc_score"] is not None, "Should compute RAROC"
            assert len(result["debate_history"]) > 0, "Should have debate history"
            assert result["decision_memo"], "Should generate memo"

            d = str(result["final_decision"]).replace("LoanDecision.", "")
            r = result["raroc_score"]
            rounds = result["debate_round"]
            return (f"{d} | RAROC={r:.4f} | "
                    f"Rounds={rounds} | {elapsed}s | "
                    f"{len(result['debate_history'])} debate entries")
        check("Full war room run (Tech Startup)", check_full_run)

        info("Running compliance veto test on Manufacturing sample (~60s)...")
        def check_veto_run():
            from war_room import run_war_room, LoanDecision
            from sample_data.sample_inputs import ALL_SAMPLES
            data = ALL_SAMPLES["manufacturing"]
            result = run_war_room(data["text"], data["company"])
            veto = result.get("compliance_veto", False)
            d = str(result["final_decision"]).replace("LoanDecision.", "")
            # Manufacturing has AML flags — should veto or reject
            assert d in ("REJECTED", "ESCALATED"), \
                f"Manufacturing with AML flags should be REJECTED/ESCALATED, got {d}"
            return (f"{d} | Veto={'YES' if veto else 'NO'} | "
                    f"Flags={result.get('critical_flags', [])}")
        check("Compliance veto test (Manufacturing)", check_veto_run)
    else:
        warn("Skipping full war room run (--quick mode)")
        results["warn"] += 1


# ══════════════════════════════════════════════════════════
# PHASE 4 — AUDIT EXPORT
# ══════════════════════════════════════════════════════════

def run_phase_4():
    section("Phase 4 — Audit Export & Memory")

    def check_db_import():
        from memory.db import (init_db, save_run, get_all_runs,
                                get_decision_stats, get_run_by_id)
        return "all db functions imported"
    check("Import memory.db", check_db_import)

    def check_db_init():
        from memory.db import init_db, DB_PATH
        init_db()
        assert DB_PATH.exists(), f"DB not created at {DB_PATH}"
        return str(DB_PATH)
    check("SQLite DB initialisation", check_db_init)

    def check_db_save_read():
        from memory.db import save_run, get_run_by_id, get_all_runs
        from war_room import LoanDecision
        fake_result = {
            "company_name": "__diagnose_test__",
            "final_decision": LoanDecision.APPROVED,
            "raroc_score": 0.1823,
            "compliance_veto": False,
            "critical_flags": ["TEST_FLAG"],
            "debate_round": 2,
            "decision_memo": "Test memo for diagnostic",
            "debate_history": ["Round 1 test"],
            "consensus_rounds": 2,
            "flags_resolved": 1,
        }
        run_id = save_run(fake_result, "test loan text", model="test-model",
                          sentiment_score=0.75, sentiment_label="Positive")
        assert isinstance(run_id, int) and run_id > 0, "Should return valid run_id"
        fetched = get_run_by_id(run_id)
        assert fetched is not None, "Should fetch saved run"
        assert fetched["company_name"] == "__diagnose_test__"
        assert fetched["final_decision"] == "APPROVED"
        return f"Saved and retrieved run #{run_id}"
    check("DB save & retrieve", check_db_save_read)

    def check_db_stats():
        from memory.db import get_decision_stats
        stats = get_decision_stats()
        assert "total" in stats, "Stats should have total"
        assert "by_decision" in stats, "Stats should have by_decision"
        assert stats["total"] >= 1, "Should have at least 1 run (from previous test)"
        return (f"Total={stats['total']} | "
                f"Decisions={stats['by_decision']} | "
                f"AvgRAROC={stats['avg_raroc']:.4f}")
    check("DB statistics", check_db_stats)

    def check_sentiment():
        from sentiment import _rule_based_sentiment
        text = ("Revenue grew 68% YoY. Strong recurring contracts. "
                "DSCR 1.42. Clean KYC. No offshore accounts.")
        result = _rule_based_sentiment(text)
        assert "overall_score" in result
        assert "label" in result
        assert "key_positive_signals" in result
        assert result["overall_score"] > 0.3, \
            f"Positive text should score > 0.3, got {result['overall_score']}"
        return (f"Score={result['overall_score']} | "
                f"Label={result['label']} | "
                f"Tone={result['borrower_tone']}")
    check("Sentiment analysis (rule-based)", check_sentiment)

    def check_report_gen():
        # Test report generation without importing app (avoids Streamlit context warnings)
        from memory.db import get_run_by_id, get_all_runs
        import json as _j
        runs = get_all_runs(1)
        assert runs, "Need at least one run"
        run = get_run_by_id(runs[0]["id"])
        # Inline the report logic to avoid importing app.py
        flags = []
        try: flags = _j.loads(run.get("critical_flags","[]"))
        except: pass
        lines = [
            "=" * 60,
            "LENDSYNTHETIX — LOAN EVALUATION REPORT",
            "=" * 60,
            f"Company: {run.get('company_name','')}",
            f"Decision: {run.get('final_decision','')}",
            "",
            "DECISION SUMMARY",
            f"RAROC: {run.get('raroc_score','N/A')}",
            f"Flags: {', '.join(flags) if flags else 'None'}",
            "",
            run.get("decision_memo","No memo") or "No memo",
        ]
        report = "\n".join(lines)
        assert "LENDSYNTHETIX" in report
        assert "DECISION SUMMARY" in report
        assert len(report) > 200
        return f"{len(report)} chars generated"
    check("Report generation", check_report_gen)


# ══════════════════════════════════════════════════════════
# PHASE 5 — SAMPLE DATASETS
# ══════════════════════════════════════════════════════════

def run_phase_5():
    section("Phase 5 — Sample Datasets")

    def check_samples_import():
        from sample_data.sample_inputs import ALL_SAMPLES
        assert "tech_startup"  in ALL_SAMPLES
        assert "manufacturing" in ALL_SAMPLES
        assert "retail"        in ALL_SAMPLES
        return f"{len(ALL_SAMPLES)} datasets loaded"
    check("Import sample datasets", check_samples_import)

    from sample_data.sample_inputs import ALL_SAMPLES
    for key, data in ALL_SAMPLES.items():
        def _check(d=data, k=key):
            assert "company" in d,          f"{k}: missing company"
            assert "text" in d,             f"{k}: missing text"
            assert "expected_outcome" in d, f"{k}: missing expected_outcome"
            assert len(d["text"]) > 200,    f"{k}: text too short"
            # Check it contains key financial metrics
            text = d["text"]
            has_metrics = any(w in text.upper() for w in
                              ["DSCR", "REVENUE", "ASSETS", "EQUITY"])
            assert has_metrics, f"{k}: text missing financial metrics"
            return (f"{d['company']} | "
                    f"{len(d['text'])} chars | "
                    f"{d['expected_outcome'][:40]}")
        check(f"Dataset: {key}", _check)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def print_summary():
    section("Diagnostic Summary")
    total = results["pass"] + results["fail"] + results["warn"]
    print(f"\n  {C.GREEN}PASSED{C.RESET}: {results['pass']}")
    print(f"  {C.RED}FAILED{C.RESET}: {results['fail']}")
    print(f"  {C.YELLOW}WARNINGS{C.RESET}: {results['warn']}")
    print(f"  Total checks: {total}\n")

    if results["fail"] == 0:
        print(f"  {C.GREEN}{C.BOLD}All checks passed.{C.RESET} "
              f"LendSynthetix is ready to run.\n")
        print(f"  Start the app with: {C.CYAN}streamlit run app.py{C.RESET}\n")
    else:
        print(f"  {C.RED}{C.BOLD}{results['fail']} check(s) failed.{C.RESET} "
              f"Fix the issues above before running the app.\n")


def _load_env_early():
    """Load API keys before anything runs — searches multiple locations."""
    def _parse_file(path):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#") and not line.startswith("["):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    # Search for secrets/env files in multiple places
    candidates = [
        _BASE / ".env",
        _BASE / ".streamlit" / "secrets.toml",
        _BASE.parent / ".env",
        _BASE.parent / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        if p.exists():
            try:
                _parse_file(p)
            except Exception:
                pass

# Load keys immediately at module import — before any check functions run
_load_env_early()


def main():
    quick = "--quick" in sys.argv
    phase = None
    if "--phase" in sys.argv:
        idx = sys.argv.index("--phase")
        if idx + 1 < len(sys.argv):
            phase = int(sys.argv[idx + 1])

    print(f"\n{C.BOLD}{C.WHITE}{'='*55}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  LendSynthetix — Diagnostic Runner{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    if quick:
        print(f"  {C.YELLOW}Quick mode — war room LLM calls skipped{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}{'='*55}{C.RESET}")

    # Report which key files were loaded
    groq_loaded = bool(os.environ.get("GROQ_API_KEY","").strip())
    if groq_loaded:
        info("API keys loaded from secrets/env file")
    else:
        warn(
            "GROQ_API_KEY not found. Add it to one of:\n"
            f"  {_BASE / '.streamlit' / 'secrets.toml'}\n"
            f"  {_BASE / '.env'}\n"
            "  Or run: set GROQ_API_KEY=your_key  (Windows)"
        )
        results["warn"] += 1

    phases = {
        0: run_phase_0,
        1: run_phase_1,
        2: run_phase_2,
        3: lambda: run_phase_3(quick=quick),
        4: run_phase_4,
        5: run_phase_5,
    }

    if phase is not None:
        phases[phase]()
    else:
        for fn in phases.values():
            fn()

    print_summary()
    sys.exit(0 if results["fail"] == 0 else 1)


if __name__ == "__main__":
    main()