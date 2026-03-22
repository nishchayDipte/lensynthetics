"""
LendSynthetix -- Sentiment & Tone Analysis
Rule-based scoring (instant, no API) with optional LLM enrichment.
Falls back gracefully on rate-limit or key errors.
"""

import os
import re
import json
import time
from typing import Dict, Any
from langchain_groq import ChatGroq

VALID_LABELS = {"Strongly Positive", "Positive", "Neutral", "Cautious", "Negative"}
VALID_TONES  = {"Aggressive Growth", "Conservative", "Distressed", "Stable", "Speculative"}


def get_llm() -> ChatGroq:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(
        model=os.environ.get("LS_MODEL", "llama-3.1-8b-instant"),
        api_key=key,
        temperature=0.1,
        max_tokens=300,
    )


SENTIMENT_PROMPT = """You are a financial analyst. Analyze this commercial loan application text.
Return ONLY a valid JSON object with NO explanation, preamble, or markdown.

Extract REAL phrases and signals from the actual text provided.

Return this exact JSON structure:
{{
  "overall_score": <float 0.0-1.0, 1.0=most positive>,
  "label": <one of: "Strongly Positive", "Positive", "Neutral", "Cautious", "Negative">,
  "confidence_score": <float 0.0-1.0, borrower confidence level>,
  "risk_language_score": <float 0.0-1.0, 1.0=very risky tone>,
  "growth_optimism_score": <float 0.0-1.0, growth optimism>,
  "key_positive_signals": [<2-3 SHORT real phrases from the text showing strength>],
  "key_risk_signals": [<2-3 SHORT real phrases from the text showing risk>],
  "borrower_tone": <one of: "Aggressive Growth", "Conservative", "Distressed", "Stable", "Speculative">
}}

LOAN APPLICATION TEXT:
{text}"""


# ── Rule-based fallback (zero API calls) ─────────────────────────────────────

POSITIVE_WORDS = [
    "growth", "revenue", "profit", "contract", "recurring", "expanding",
    "strong", "healthy", "positive", "increase", "improve", "opportunity",
    "established", "experienced", "collateral", "assets", "clean", "compliant",
]
RISK_WORDS = [
    "loss", "decline", "debt", "overdue", "default", "offshore", "cayman",
    "suspicious", "investigation", "bounced", "grey list", "interpol",
    "volatile", "uncertain", "struggling", "negative", "risk", "concern",
]


def _rule_based_sentiment(text: str) -> Dict[str, Any]:
    """Fast word-boundary keyword scoring — no API call needed."""
    text_lower = text.lower()

    # Word-boundary matching to avoid substring false positives
    def _count(words):
        return sum(
            1 for w in words
            if re.search(r"\b" + re.escape(w) + r"\b", text_lower)
        )

    pos   = _count(POSITIVE_WORDS)
    neg   = _count(RISK_WORDS)
    total = max(pos + neg, 1)
    score = round(min(1.0, max(0.0, pos / total)), 2)

    if score >= 0.75:   label = "Strongly Positive"
    elif score >= 0.6:  label = "Positive"
    elif score >= 0.45: label = "Neutral"
    elif score >= 0.3:  label = "Cautious"
    else:               label = "Negative"

    if any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in ["interpol", "grey list", "bounced", "default"]):
        tone = "Distressed"
    elif any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in ["offshore", "cayman", "suspicious"]):
        tone = "Speculative"
    elif score >= 0.7:  tone = "Aggressive Growth"
    elif score >= 0.5:  tone = "Stable"
    else:               tone = "Conservative"

    def _extract_phrase(keyword: str, window: int = 40) -> str:
        m = re.search(r"\b" + re.escape(keyword) + r"\b", text_lower)
        if not m:
            return keyword
        start  = max(0, m.start() - 10)
        end    = min(len(text), m.end() + window)
        phrase = re.sub(r"\s+", " ", text[start:end]).strip()
        return phrase[:60].strip("., ")

    pos_signals  = [_extract_phrase(w) for w in POSITIVE_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text_lower)][:3]
    risk_signals = [_extract_phrase(w) for w in RISK_WORDS     if re.search(r"\b" + re.escape(w) + r"\b", text_lower)][:3]

    return {
        "overall_score":         score,
        "label":                 label,
        "confidence_score":      round(min(1.0, pos / 10), 2),
        "risk_language_score":   round(min(1.0, neg / 8),  2),
        "growth_optimism_score": round(score * 0.9, 2),
        "key_positive_signals":  pos_signals,
        "key_risk_signals":      risk_signals,
        "borrower_tone":         tone,
        "_source":               "rule_based",
    }


def analyze_sentiment(loan_text: str) -> Dict[str, Any]:
    """
    1. Run rule-based scoring instantly (no API cost).
    2. Attempt LLM enrichment with retry on 429.
    3. If LLM unavailable, return rule-based result.
    """
    if not loan_text or len(loan_text.strip()) < 50:
        return _default_sentiment()

    # Step 1: instant rule-based baseline
    result = _rule_based_sentiment(loan_text)

    # Step 2: LLM enrichment — use a larger window (2000 chars) for better accuracy
    max_retries = 3
    for attempt in range(max_retries):
        try:
            llm      = get_llm()
            prompt   = SENTIMENT_PROMPT.format(text=loan_text[:2000])
            response = llm.invoke(prompt).content.strip()
            response = re.sub(r"```json|```", "", response).strip()

            data = json.loads(response)

            # Clamp all numeric scores
            for key in ("overall_score", "confidence_score", "risk_language_score", "growth_optimism_score"):
                data[key] = max(0.0, min(1.0, float(data.get(key, result[key]))))

            # Validate label and tone — fall back to rule-based if LLM returns unexpected value
            if data.get("label") not in VALID_LABELS:
                data["label"] = result["label"]
            if data.get("borrower_tone") not in VALID_TONES:
                data["borrower_tone"] = result["borrower_tone"]

            data["_source"] = "llm"
            return data

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 8 * (attempt + 1)
                print(f"[SENTIMENT] Rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"[SENTIMENT] LLM error: {e} — using rule-based result")
                break

    print("[SENTIMENT] Using rule-based sentiment (LLM unavailable)")
    return result


def _default_sentiment() -> Dict[str, Any]:
    return {
        "overall_score": 0.5, "label": "Neutral",
        "confidence_score": 0.5, "risk_language_score": 0.5,
        "growth_optimism_score": 0.5,
        "key_positive_signals": [], "key_risk_signals": [],
        "borrower_tone": "Stable", "_source": "default",
    }


def sentiment_color(label: str) -> str:
    return {
        "Strongly Positive": "#10B981",
        "Positive":          "#34D399",
        "Neutral":           "#F59E0B",
        "Cautious":          "#F97316",
        "Negative":          "#EF4444",
    }.get(label, "#6B7280")


def tone_color(tone: str) -> str:
    return {
        "Aggressive Growth": "#3B82F6",
        "Conservative":      "#6B7280",
        "Distressed":        "#EF4444",
        "Stable":            "#10B981",
        "Speculative":       "#F59E0B",
    }.get(tone, "#6B7280")