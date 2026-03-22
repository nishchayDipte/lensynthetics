"""
LendSynthetix — Persistent Memory Layer (SQLite)
Stores all war room runs, decisions, and agent debate history.
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).parent / "lendsynthetix.db"


@contextmanager
def _get_conn():
    """Context-manager connection — always closed, exceptions propagate cleanly."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables and indexes if they don't exist."""
    with _get_conn() as conn:
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS loan_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name    TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                final_decision  TEXT    NOT NULL,
                raroc_score     REAL,
                compliance_veto INTEGER DEFAULT 0,
                debate_rounds   INTEGER DEFAULT 1,
                critical_flags  TEXT,
                decision_memo   TEXT,
                loan_text       TEXT,
                debate_history  TEXT,
                model_used      TEXT    DEFAULT 'llama-3.1-8b-instant',
                sentiment_score REAL,
                sentiment_label TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_chat (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER REFERENCES loan_runs(id),
                timestamp   TEXT NOT NULL,
                role        TEXT NOT NULL,
                message     TEXT NOT NULL
            )
        """)

        # Indexes for fast lookups
        c.execute("CREATE INDEX IF NOT EXISTS idx_runs_id        ON loan_runs(id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON loan_runs(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_run_id    ON agent_chat(run_id)")


def save_run(
    result: Dict[str, Any],
    loan_text: str,
    model: str,
    sentiment_score: float = None,
    sentiment_label: str = None,
) -> int:
    """Save a completed war room run. Returns the new run ID."""
    decision = str(result.get("final_decision", "PENDING")).replace("LoanDecision.", "")
    flags    = json.dumps(result.get("critical_flags", []))
    history  = json.dumps([str(h) for h in result.get("debate_history", [])])

    try:
        with _get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO loan_runs
                    (company_name, timestamp, final_decision, raroc_score, compliance_veto,
                     debate_rounds, critical_flags, decision_memo, loan_text, debate_history,
                     model_used, sentiment_score, sentiment_label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.get("company_name", "Unknown"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                decision,
                result.get("raroc_score"),
                1 if result.get("compliance_veto") else 0,
                result.get("debate_round", 1),
                flags,
                str(result.get("decision_memo", "")),
                loan_text,          # store full text — no truncation
                history,
                model,
                sentiment_score,
                sentiment_label,
            ))
            return c.lastrowid
    except Exception as e:
        print(f"[DB] save_run failed: {e}")
        return -1


def save_chat_message(run_id: int, role: str, message: str):
    """Save a chat message linked to a run."""
    try:
        with _get_conn() as conn:
            conn.cursor().execute("""
                INSERT INTO agent_chat (run_id, timestamp, role, message)
                VALUES (?, ?, ?, ?)
            """, (run_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), role, message))
    except Exception as e:
        print(f"[DB] save_chat_message failed: {e}")


def get_chat_history(run_id: int) -> List[Dict]:
    try:
        with _get_conn() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT role, message, timestamp FROM agent_chat WHERE run_id=? ORDER BY id",
                (run_id,),
            )
            return [dict(r) for r in c.fetchall()]
    except Exception as e:
        print(f"[DB] get_chat_history failed: {e}")
        return []


def get_all_runs(limit: int = 50) -> List[Dict]:
    """Fetch recent runs for dashboard history."""
    try:
        with _get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT id, company_name, timestamp, final_decision, raroc_score,
                       compliance_veto, debate_rounds, critical_flags, sentiment_label
                FROM loan_runs
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            return [dict(r) for r in c.fetchall()]
    except Exception as e:
        print(f"[DB] get_all_runs failed: {e}")
        return []


def get_run_by_id(run_id: int) -> Optional[Dict]:
    try:
        with _get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM loan_runs WHERE id=?", (run_id,))
            row = c.fetchone()
            return dict(row) if row else None
    except Exception as e:
        print(f"[DB] get_run_by_id failed: {e}")
        return None


def get_decision_stats() -> Dict[str, Any]:
    """Aggregate stats for the dashboard history tab."""
    try:
        with _get_conn() as conn:
            c = conn.cursor()

            c.execute("SELECT COUNT(*) as total FROM loan_runs")
            total = c.fetchone()["total"]

            c.execute("SELECT final_decision, COUNT(*) as cnt FROM loan_runs GROUP BY final_decision")
            by_decision = {r["final_decision"]: r["cnt"] for r in c.fetchall()}

            c.execute("SELECT AVG(raroc_score) as avg_raroc FROM loan_runs WHERE raroc_score IS NOT NULL")
            row       = c.fetchone()
            avg_raroc = row["avg_raroc"] if row["avg_raroc"] is not None else 0.0

            c.execute("SELECT AVG(debate_rounds) as avg_rounds FROM loan_runs")
            row        = c.fetchone()
            avg_rounds = row["avg_rounds"] if row["avg_rounds"] is not None else 0.0

            c.execute("""
                SELECT company_name, timestamp, final_decision, raroc_score
                FROM loan_runs ORDER BY id DESC LIMIT 10
            """)
            recent = [dict(r) for r in c.fetchall()]

        return {
            "total":       total,
            "by_decision": by_decision,
            "avg_raroc":   avg_raroc,
            "avg_rounds":  avg_rounds,
            "recent":      recent,
        }
    except Exception as e:
        print(f"[DB] get_decision_stats failed: {e}")
        return {"total": 0, "by_decision": {}, "avg_raroc": 0.0, "avg_rounds": 0.0, "recent": []}


# Auto-init on import
init_db()