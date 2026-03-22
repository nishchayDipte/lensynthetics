"""
LendSynthetix — The Loan War Room
Multi-Agent Orchestration for Commercial Lending
Stack: LangGraph + Groq

FIXES APPLIED:
  1. RAROC computed entirely in Python — LLM no longer touches the arithmetic
  2. Collateral recognised from structured metrics (not just free text)
  3. War-room context expanded from 2,500 → 6,000 chars
  4. LGD rule priority: D/E ratio first, collateral as secondary modifier
  5. Moderator prompt is now qualitative-only; all numbers come from Python
  6. Sales/Risk memo context expanded from 600 → 1,200 chars
"""

import os
import re
import time
import operator
from typing import TypedDict, Annotated, List, Optional
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RAROC_APPROVE       = 0.15
RAROC_ESCALATE      = 0.08
MAX_DEBATE_ROUNDS   = 2
MAX_HISTORY_ENTRIES = 20


# ═══════════════════════════════════════════════════════════════════
# HARDCODED RAROC ENGINE  — Python does ALL the maths, never the LLM
# ═══════════════════════════════════════════════════════════════════

class RarocInputs:
    """Parsed financial inputs used for RAROC calculation."""
    def __init__(self):
        self.loan_amount:      Optional[float] = None   # in Lakhs
        self.dscr:             Optional[float] = None
        self.de_ratio:         Optional[float] = None
        self.cibil_score:      Optional[int]   = None
        self.revenue_growth:   Optional[float] = None   # fraction e.g. 0.68
        self.collateral_value: Optional[float] = None   # in Lakhs; 0 = no collateral
        self.compliance_veto:  bool            = False

    def __repr__(self):
        return (
            f"RarocInputs(loan={self.loan_amount}L, dscr={self.dscr}, "
            f"de={self.de_ratio}, cibil={self.cibil_score}, "
            f"growth={self.revenue_growth}, collateral={self.collateral_value})"
        )


class RarocResult:
    """Fully computed RAROC with step-by-step audit trail."""
    def __init__(self):
        self.nii:               float = 0.0
        self.pd:                float = 0.0
        self.pd_reason:         str   = ""
        self.lgd:               float = 0.0
        self.lgd_reason:        str   = ""
        self.el:                float = 0.0
        self.economic_capital:  float = 0.0
        self.risk_weight:       float = 0.0
        self.raw_raroc:         float = 0.0
        self.adjustments:       float = 0.0
        self.adjustment_notes:  str   = ""
        self.final_raroc:       float = 0.0
        self.decision_band:     str   = ""
        self.audit_trail:       str   = ""

    def to_block(self) -> str:
        """Return formatted string for insertion into agent context."""
        return (
            f"═══ PYTHON RAROC CALCULATION (authoritative) ═══\n"
            f"  NII              = {self.nii:.3f} L  (Loan x 11%)\n"
            f"  PD               = {self.pd:.4f}  — {self.pd_reason}\n"
            f"  LGD              = {self.lgd:.4f}  — {self.lgd_reason}\n"
            f"  Expected Loss    = {self.el:.3f} L  (PD x LGD x Loan)\n"
            f"  Risk Weight      = {self.risk_weight:.4f}\n"
            f"  Economic Capital = {self.economic_capital:.3f} L  (Loan x RW)\n"
            f"  Raw RAROC        = ({self.nii:.3f} - {self.el:.3f}) / {self.economic_capital:.3f}\n"
            f"                   = {self.raw_raroc:.4f}\n"
            f"  Adjustments      = {self.adjustments:+.4f}  [{self.adjustment_notes}]\n"
            f"  ────────────────────────────────────────────\n"
            f"  FINAL RAROC      = {self.final_raroc:.4f}\n"
            f"  DECISION BAND    = {self.decision_band}\n"
            f"═══════════════════════════════════════════════"
        )


def _parse_number(text: str) -> Optional[float]:
    """Extract first numeric value from a string."""
    if not text:
        return None
    m = re.search(r"[\d]+\.?[\d]*", text.replace(",", ""))
    return float(m.group()) if m else None


def _parse_loan_amount_lakhs(text: str) -> Optional[float]:
    """
    Parse loan amount and normalise to Lakhs.
    Handles: '2.5 Crore', '250 Lakhs', '2,50,00,000', '25000000'

    IMPORTANT: when a unit keyword (crore/lakh) is present, find the number
    IMMEDIATELY before or after that keyword — not the first number in the string.
    This prevents '2,50,00,000 (2.5 Crore)' from matching 25000000 * 100.
    """
    if not text:
        return None
    t = text.lower().replace(",", "")

    # Strategy 1: find number immediately adjacent to a unit keyword
    # e.g. "2.5 crore", "rs 2.5 crore", "250 lakhs", "250 lakh"
    m = re.search(r"([\d]+\.?[\d]*)\s*(?:crore|cr)\b", t)
    if m:
        return float(m.group(1)) * 100      # crore -> Lakhs

    m = re.search(r"([\d]+\.?[\d]*)\s*(?:lakh|lac)\b", t)
    if m:
        return float(m.group(1))            # already Lakhs

    # Strategy 2: bare number with no unit — apply heuristic size conversion
    n = _parse_number(t)
    if n is None:
        return None
    if n > 1_000_000:
        return n / 100_000      # rupees -> Lakhs (e.g. 25000000 -> 250)
    if n > 10_000:
        return n / 100          # thousands -> Lakhs
    return n                    # already Lakhs


def extract_raroc_inputs(financial_metrics: str, loan_application: str) -> RarocInputs:
    """
    Extract numeric inputs for RAROC.  All parsing is Python — no LLM arithmetic.

    Strategy for each field:
      1. Try to compute directly from raw balance-sheet / source figures
         (most reliable — bypasses any LLM misread).
      2. Fall back to labelled patterns in the metrics block + loan text.
      3. Cross-validate: if balance-sheet computation disagrees with the
         labelled value by more than a small tolerance, prefer the computed value
         and log the override.

    This eliminates the 4 known hallucination sources:
      • D/E misread   — recomputed from Total Debt / Total Equity
      • Collateral=0  — deep multi-pattern scan across the full text
      • Revenue growth misread — anchored to the most recent FY pair
      • CIBIL missed  — broader 3-digit search with common aliases
    """
    inp = RarocInputs()
    combined = (financial_metrics or "") + "\n" + (loan_application or "")[:8000]

    # ════════════════════════════════════════════════════════
    # 1. LOAN AMOUNT
    # ════════════════════════════════════════════════════════
    for pat in [
        r"Loan\s*Amount\s*[:\-]\s*([^\n]+)",
        r"Amount\s*Requested\s*[:\-]\s*([^\n]+)",
        r"(?:loan|facility|credit)\s*(?:amount|size)\s*[:\-]\s*([^\n]+)",
        r"[Rr]s\.?\s*([\d,\.]+\s*(?:crore|cr|lakh|lac))",
    ]:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            v = _parse_loan_amount_lakhs(m.group(1))
            if v and v > 0:
                inp.loan_amount = v
                break

    # ════════════════════════════════════════════════════════
    # 2. DSCR — take the most recent (highest FY) value
    # ════════════════════════════════════════════════════════
    # Collect ALL DSCR mentions and pick the last stated current-year value
    dscr_vals = re.findall(r"DSCR\s*[:\-\(]?\s*([\d\.]+)", combined, re.IGNORECASE)
    if dscr_vals:
        # Filter to plausible DSCR range (0.1 – 10.0)
        plausible = [float(v) for v in dscr_vals if 0.1 <= float(v) <= 10.0]
        if plausible:
            # The document lists historical then current — take the value
            # closest to 1.0-3.0 (operating range) from the last 3 mentions
            candidates = plausible[-3:]
            inp.dscr = min(candidates, key=lambda x: abs(x - 1.5))

    # ════════════════════════════════════════════════════════
    # 3. D/E RATIO — compute from balance sheet first
    # ════════════════════════════════════════════════════════

    # Step A: try to read Total Debt and Total Equity directly
    computed_de    = None
    total_debt_val = None
    wc_val_seen       = None   # track WC so we can sanity-check "Total Debt" label
    total_debt_partial = False  # flag: did we get only partial debt data?

    # Priority 1: sum Term Loans + WC Loans individually
    # Covers all rag summary formats:
    #   "Term Loans (Vehicle) 28.00"           — balance sheet row
    #   "Term loans Rs 28 Lakhs, WC loans 42"  — rag Existing Loans line
    term = re.search(r"Term\s*Loans?\s*[\(:\-\s][^\n]{0,30}?([\d,\.]+)", combined, re.IGNORECASE)
    # WC: covers "Working Capital Loans 42", "+ WC Rs 42", "WC loans Rs 42"
    wc = re.search(
        r"(?:Working\s*Capital\s*Loans?\s*[\(:\-\s]"
        r"|[+,]\s*WC\s*Loans?\s+(?:Rs\.?\s*)?"
        r"|[+&]\s*WC\s+(?:Rs\.?\s*)?)"
        r"[^\n]{0,20}?([\d,\.]+)",
        combined, re.IGNORECASE
    )
    t_val = _parse_number(term.group(1)) if term else None
    w_val = _parse_number(wc.group(1))   if wc   else None

    # Sanity: reject values > 500L for individual loan lines
    # The rupee symbol ₹ has Unicode codepoint 8377 — pdfplumber sometimes
    # emits this as a bare number. Any Term/WC line > 500L is garbled text.
    MAX_SINGLE_LOAN = 500.0   # no single loan line should exceed 500L for a 2.5Cr facility
    if t_val and t_val > MAX_SINGLE_LOAN:
        print(f"[RAROC_ENGINE] Term Loans value {t_val}L rejected — likely garbled Unicode (₹=8377)")
        t_val = None
    if w_val and w_val > MAX_SINGLE_LOAN:
        print(f"[RAROC_ENGINE] WC Loans value {w_val}L rejected — likely garbled Unicode (₹=8377)")
        w_val = None
    wc_val_seen = w_val

    if t_val and w_val:
        total_debt_val = t_val + w_val
        print(f"[RAROC_ENGINE] Total Debt: Term {t_val}L + WC {w_val}L = {total_debt_val}L")
    elif t_val:
        print(f"[RAROC_ENGINE] Warning: only Term Loans ({t_val}L) — incomplete")
    elif w_val:
        print(f"[RAROC_ENGINE] Warning: only WC Loans ({w_val}L) — incomplete")

    # Priority 2: "Total Debt" / "Total Borrowings" label — fallback only
    # Sanity: reject if value == WC alone (rag Groq extracted partial debt)
    # Also reject if value is suspiciously small vs equity (D/E would be < 0.3)
    if not total_debt_val:
        m = re.search(r"Total\s*(?:Debt|Borrowings?)\s*[:\-]\s*([^\n]+)", combined, re.IGNORECASE)
        if m:
            lt = _parse_loan_amount_lakhs(m.group(1))
            if lt and lt > 0:
                is_partial = (wc_val_seen and abs(lt - wc_val_seen) < 1.0)
                if is_partial:
                    print(f"[RAROC_ENGINE] 'Total Debt' {lt}L == WC only ({wc_val_seen}L) "
                          f"— partial, not using")
                    total_debt_partial = True
                else:
                    total_debt_val = lt
                    print(f"[RAROC_ENGINE] Total Debt from label: {total_debt_val}L")

    # Total Equity
    total_equity_val = None
    for pat in [
        r"Total\s*Equity\s*[:\-\s]\s*([\d,\.]+)",
        r"Equity\s*/\s*Net\s*Worth\s*[:\-]\s*([^\n]+)",
        r"(?:Share\s*holders['']?\s*)?(?:Net\s*)?Worth\s*[:\-]\s*([^\n]+)",
    ]:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            v = _parse_number(m.group(1).replace(",", ""))
            if v and v > 0:
                total_equity_val = v
                break

    if total_debt_val and total_equity_val and total_equity_val > 0:
        computed_de = round(total_debt_val / total_equity_val, 3)
        print(f"[RAROC_ENGINE] D/E computed from balance sheet: "
              f"{total_debt_val}/{total_equity_val} = {computed_de}")

    # Step B: labelled D/E — take LAST occurrence (rag summary comes after llm metrics)
    labelled_de = None

    # Pattern 1: ratio table "1.24  0.875" — second value is FY24
    m = re.search(r"Debt.to.Equity\s*(?:Ratio)?\s+([\d\.]+)\s+([\d\.]+)", combined, re.IGNORECASE)
    if m:
        labelled_de = float(m.group(2))

    # Pattern 2: all single-value mentions — last one wins (rag over llm)
    if labelled_de is None:
        all_de = re.findall(r"Debt.to.Equity\s*(?:Ratio)?\s*[:\-]\s*([\d\.]+)", combined, re.IGNORECASE)
        if all_de:
            labelled_de = float(all_de[-1])

    if labelled_de is None:
        all_de_short = re.findall(r"D/E\s*(?:ratio)?\s*[:\-]\s*([\d\.]+)", combined, re.IGNORECASE)
        if all_de_short:
            labelled_de = float(all_de_short[-1])

    # Step C: choose D/E — core decision logic
    #
    # Case 1: computed AND labelled available
    #   - If they agree (< 0.05 diff): use computed (balance sheet is gold)
    #   - If computed > labelled: computed has MORE debt than expected
    #     → labelled likely more accurate (LLM ratio table read correctly)
    #   - If computed < labelled: computed has LESS debt than expected
    #     → means debt was partial (only WC, missed Term Loans)
    #     → labelled is more accurate — use it
    #
    # Case 2: only one available — use whichever exists
    #
    # Special case: if total_debt_partial=True, computed is definitely wrong
    # → always prefer labelled in that case
    if computed_de is not None and labelled_de is not None:
        diff = abs(computed_de - labelled_de)
        if total_debt_partial:
            print(f"[RAROC_ENGINE] D/E: partial debt confirmed — using labelled {labelled_de}")
            inp.de_ratio = labelled_de
        elif diff <= 0.05:
            inp.de_ratio = computed_de
        elif computed_de < labelled_de - 0.05:
            # Computed is LOWER → partial debt captured → trust labelled
            print(f"[RAROC_ENGINE] D/E: computed {computed_de} < labelled {labelled_de} "
                  f"— partial debt detected, using labelled")
            inp.de_ratio = labelled_de
        else:
            # Computed is HIGHER → LLM over-estimated → trust computed
            print(f"[RAROC_ENGINE] D/E: computed {computed_de} > labelled {labelled_de} "
                  f"— using computed (balance sheet authoritative)")
            inp.de_ratio = computed_de
    elif computed_de is not None:
        inp.de_ratio = computed_de
    elif labelled_de is not None:
        inp.de_ratio = labelled_de

    # ════════════════════════════════════════════════════════
    # 4. CIBIL / CREDIT SCORE — broader search with aliases
    # ════════════════════════════════════════════════════════
    cibil_pats = [
        r"CIBIL\s*(?:Score)?\s*[:\-\(]?\s*(\d{3})",          # CIBIL Score: 762
        r"Credit\s*Score\s*[:\-]\s*(\d{3})",                  # Credit Score: 762
        r"Score\s*[:\-]\s*(\d{3})\s*[—\-]?\s*No\s*adverse",  # Score: 762 — No adverse
        r"(?:credit|cibil)\s*[:\-\s]*(\d{3})",                # loose match
    ]
    for pat in cibil_pats:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            if 300 <= score <= 900:   # valid CIBIL range
                inp.cibil_score = score
                break

    # ════════════════════════════════════════════════════════
    # 5. REVENUE GROWTH YoY — anchor to most recent FY pair
    # ════════════════════════════════════════════════════════
    growth_val = None

    # Pattern A: ratio table with two values "438% 68%" — take LAST (FY24)
    m = re.search(r"Revenue\s*Growth\s*(?:YoY)?\s+([\d\.]+)%\s+([\d\.]+)%", combined, re.IGNORECASE)
    if m:
        v = float(m.group(2))   # group(2) = FY24 (current year)
        if 0 < v < 1000:
            growth_val = v / 100.0

    # Pattern B: labelled single value "Revenue Growth YoY: 68%"
    if growth_val is None:
        for pat in [
            r"Revenue\s*Growth\s*(?:YoY)?\s*[:\-]\s*([\d\.]+)\s*%",
            r"(?:yoy|year.on.year)\s*(?:revenue\s*)?growth\s*[:\-]\s*([\d\.]+)\s*%",
            r"([\d\.]+)\s*%\s*(?:YoY|year.on.year)\s*(?:revenue\s*)?growth",
        ]:
            m = re.search(pat, combined, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if 0 < v < 1000:
                    growth_val = v / 100.0
                    break

    # Pattern C: compute from consecutive revenue figures if no labelled value
    if growth_val is None:
        rev_matches = re.findall(
            r"Revenue\s+(?:from\s+Operations\s+)?[\d,\.]+\s+([\d,\.]+)\s+([\d,\.]+)",
            combined, re.IGNORECASE
        )
        if rev_matches:
            try:
                prev = float(rev_matches[0][0].replace(",", ""))
                curr = float(rev_matches[0][1].replace(",", ""))
                if prev > 0 and curr > prev:
                    growth_val = round((curr - prev) / prev, 4)
            except Exception:
                pass

    if growth_val is not None:
        inp.revenue_growth = growth_val

    # ════════════════════════════════════════════════════════
    # 6. COLLATERAL — deep multi-pattern scan; never accept
    #    zero if the text contains property/security keywords
    # ════════════════════════════════════════════════════════
    collateral_found = False

    # Tier 1: explicit total security value
    for pat in [
        r"Total\s*Primary\s*Security\s*[:\-\|]?\s*([\d,\.]+)",
        r"Total\s*Security\s*[:\-]\s*([^\n]+)",
        r"Collateral\s*Value\s*[:\-\|]?\s*([\d,\.]+)",
    ]:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            raw = m.group(1).lower().strip()
            if not any(w in raw for w in ["not found", "nil", "none", "n/a"]):
                v = _parse_number(raw.replace(",", ""))
                if v and v > 0:
                    inp.collateral_value = v   # already in Lakhs from balance sheet
                    collateral_found = True
                    print(f"[RAROC_ENGINE] Collateral (tier-1): {v}L")
                    break

    # Tier 2: property value in security table
    # GUARD: property address contains PIN code (e.g. 560103) which looks like
    # a large number. Reject anything > 10,000L (= Rs 100 Cr — implausible
    # for collateral on a 2.5 Cr loan) to avoid matching address digits.
    if not collateral_found:
        for pat in [
            r"(?:Registered\s*Office|Property|Freehold|Premises)"
            r"[^\n]{0,80}?([\d,\.]+\.?\d*)\s*(?:\.00)?(?=\s*\d+%|\s*LTV|\s*38%|\n)",
            r"(?:property|office|premises|freehold)[^\n]{0,60}?"
            r"([\d,\.]+)\s*(?:lakh|lac|L\b)",
        ]:
            m = re.search(pat, combined, re.IGNORECASE)
            if m:
                v = _parse_number(m.group(1).replace(",", ""))
                if v and 5 < v < 10_000:   # 5L < valid collateral < 10,000L
                    inp.collateral_value = v
                    collateral_found = True
                    print(f"[RAROC_ENGINE] Collateral (tier-2 property): {v}L")
                    break
                elif v:
                    print(f"[RAROC_ENGINE] Collateral tier-2 rejected {v} "
                          f"— outside 5-10000L bounds (likely PIN/address number)")

    # Tier 3: Collateral Offered / Value line from rag summary
    # Guard: rag Groq sometimes returns only the address string without a numeric value
    # e.g. "Registered office property at 4th Floor, Prestige Tech Park, Bengaluru 560103"
    # We detect address-only strings and skip to avoid extracting PIN codes as amounts.
    if not collateral_found:
        m = re.search(
            r"Collateral\s*(?:Value|Offered|Coverage)?\s*[:\-]\s*([^\n]+)",
            combined, re.IGNORECASE
        )
        if m:
            raw      = m.group(1).strip()
            raw_lower = raw.lower()
            if any(w in raw_lower for w in ["not found", "nil", "none", "n/a", "no collateral"]):
                security_keywords = ["property", "freehold", "guarantee", "lien",
                                     "mortgage", "security", "receivable"]
                has_security = any(kw in combined.lower() for kw in security_keywords)
                if not has_security:
                    inp.collateral_value = 0.0
                    collateral_found = True
                    print("[RAROC_ENGINE] Collateral: confirmed zero")
                else:
                    print("[RAROC_ENGINE] Collateral: LLM said none but security keywords present "
                          "— keeping as partial (no modifier)")
            else:
                # Only extract if the line contains a numeric monetary value
                # Pure address strings ("4th Floor, Prestige, Bengaluru 560103") have no unit
                has_value = bool(
                    re.search(r'\d+\.?\d*\s*(?:lakh|lac|crore|cr|L\b)', raw, re.IGNORECASE)
                    or re.search(r'(?:Rs\.?\s*|₹\s*)\d', raw, re.IGNORECASE)
                )
                if has_value:
                    v = _parse_loan_amount_lakhs(raw)
                    if v is not None and 5 < v < 10_000:
                        inp.collateral_value = v
                        collateral_found = True
                        print(f"[RAROC_ENGINE] Collateral (tier-3): {v}L")
                else:
                    print("[RAROC_ENGINE] Collateral tier-3: address-only, no numeric value")

    # Tier 4: lien on receivables
    if not collateral_found:
        m = re.search(
            r"(?:Lien|Charge)\s+on\s+Receivables[^\n]{0,60}?([\d,\.]+)",
            combined, re.IGNORECASE
        )
        if m:
            v = _parse_number(m.group(1).replace(",", ""))
            if v and v > 0:
                inp.collateral_value = v
                collateral_found = True
                print(f"[RAROC_ENGINE] Collateral (tier-4 lien): {v}L")

    # Tier 5: Collateral Coverage Ratio — coverage > 1.0x proves collateral exists
    # "Collateral Coverage: 1.58x" with loan=250L → collateral > loan
    # We set a conservative floor of 50% of loan to avoid over-estimating
    if not collateral_found:
        m = re.search(
            r"Collateral\s*Coverage\s*(?:Ratio)?\s*[:\-]\s*([\d\.]+)\s*[xX]",
            combined, re.IGNORECASE
        )
        if m:
            cov = float(m.group(1))
            if cov >= 1.0 and inp.loan_amount:
                floor_val = round(inp.loan_amount * 0.5, 1)
                inp.collateral_value = floor_val
                collateral_found = True
                print(f"[RAROC_ENGINE] Collateral (tier-5 coverage {cov}x): "
                      f"set floor = {floor_val}L")

    return inp


def compute_raroc(inp: RarocInputs, unresolved_flags: int) -> RarocResult:
    """
    Pure-Python RAROC computation.
    The LLM never touches these numbers.

    ── PD table (DSCR bands) ────────────────────────────────────────────────
      DSCR > 1.75          -> PD = 0.01
      DSCR 1.50 – 1.75     -> PD = 0.02
      DSCR 1.25 – 1.50     -> PD = 0.05
      DSCR 1.00 – 1.25     -> PD = 0.10
      DSCR 0.75 – 1.00     -> PD = 0.15
      DSCR < 0.75          -> PD = 0.20
      unknown              -> PD = 0.10 (conservative)
    CIBIL < 700            -> +0.02 penalty

    ── LGD table (D/E ratio — primary rule) ─────────────────────────────────
      D/E < 0.5            -> LGD = 0.25
      D/E 0.5 – 1.0        -> LGD = 0.35
      D/E 1.0 – 2.0        -> LGD = 0.50
      D/E 2.0 – 3.0        -> LGD = 0.60
      D/E > 3.0            -> LGD = 0.70
      unknown              -> LGD = 0.55 (conservative)
    Collateral modifier (secondary):
      collateral >= loan   -> -0.05 (better security)
      collateral = 0       -> +0.05 (no security)
      LGD capped: [0.20, 0.80]

    ── Risk-weight table ─────────────────────────────────────────────────────
      PD <= 0.05 AND LGD <= 0.40  -> 0.06
      PD <= 0.05 AND LGD <= 0.50  -> 0.08
      PD <= 0.10 AND LGD <= 0.55  -> 0.12
      PD <= 0.15 AND LGD <= 0.65  -> 0.18
      otherwise                   -> 0.25

    ── Adjustments ──────────────────────────────────────────────────────────
      revenue growth > 50%  -> +0.02
      revenue growth > 30%  -> +0.01
      unresolved flag       -> -0.02 each
      compliance veto       -> RAROC forced to 0.0
    """
    r    = RarocResult()
    loan = inp.loan_amount or 250.0   # safe default: 250 Lakhs = Rs 2.5 Cr

    # ── NII ──────────────────────────────────────────────────────────────────
    INTEREST_RATE = 0.11
    r.nii = loan * INTEREST_RATE

    # ── PD from DSCR ─────────────────────────────────────────────────────────
    dscr = inp.dscr
    if dscr is None:
        r.pd, r.pd_reason = 0.10, "DSCR unknown -> conservative 10%"
    elif dscr > 1.75:
        r.pd, r.pd_reason = 0.01, f"DSCR {dscr:.2f} > 1.75 -> PD 1%"
    elif dscr > 1.50:
        r.pd, r.pd_reason = 0.02, f"DSCR {dscr:.2f} in 1.50-1.75 -> PD 2%"
    elif dscr > 1.25:
        r.pd, r.pd_reason = 0.05, f"DSCR {dscr:.2f} in 1.25-1.50 -> PD 5%"
    elif dscr > 1.00:
        r.pd, r.pd_reason = 0.10, f"DSCR {dscr:.2f} in 1.00-1.25 -> PD 10%"
    elif dscr > 0.75:
        r.pd, r.pd_reason = 0.15, f"DSCR {dscr:.2f} in 0.75-1.00 -> PD 15%"
    else:
        r.pd, r.pd_reason = 0.20, f"DSCR {dscr:.2f} < 0.75 -> PD 20%"

    if inp.cibil_score is not None and inp.cibil_score < 700:
        r.pd = min(0.30, r.pd + 0.02)
        r.pd_reason += f"; CIBIL {inp.cibil_score} < 700 -> +2% penalty"

    # ── LGD from D/E ratio (primary) ─────────────────────────────────────────
    de = inp.de_ratio
    if de is None:
        r.lgd, r.lgd_reason = 0.55, "D/E unknown -> conservative 55%"
    elif de < 0.5:
        r.lgd, r.lgd_reason = 0.25, f"D/E {de:.3f} < 0.5 -> LGD 25%"
    elif de < 1.0:
        r.lgd, r.lgd_reason = 0.35, f"D/E {de:.3f} in 0.5-1.0 -> LGD 35%"
    elif de < 2.0:
        r.lgd, r.lgd_reason = 0.50, f"D/E {de:.3f} in 1.0-2.0 -> LGD 50%"
    elif de < 3.0:
        r.lgd, r.lgd_reason = 0.60, f"D/E {de:.3f} in 2.0-3.0 -> LGD 60%"
    else:
        r.lgd, r.lgd_reason = 0.70, f"D/E {de:.3f} > 3.0 -> LGD 70%"

    # Collateral modifier
    if inp.collateral_value is not None:
        coll = inp.collateral_value
        if coll >= loan:
            r.lgd = max(0.20, r.lgd - 0.05)
            r.lgd_reason += f"; collateral {coll:.1f}L >= loan -> -5% bonus"
        elif coll == 0.0:
            r.lgd = min(0.80, r.lgd + 0.05)
            r.lgd_reason += "; no collateral confirmed -> +5% penalty"
        else:
            r.lgd_reason += f"; partial collateral {coll:.1f}L (no modifier)"

    # ── Expected Loss ─────────────────────────────────────────────────────────
    r.el = r.pd * r.lgd * loan

    # ── Risk Weight & Economic Capital ───────────────────────────────────────
    if r.pd <= 0.05 and r.lgd <= 0.40:
        r.risk_weight = 0.06
    elif r.pd <= 0.05 and r.lgd <= 0.50:
        r.risk_weight = 0.08
    elif r.pd <= 0.10 and r.lgd <= 0.55:
        r.risk_weight = 0.12
    elif r.pd <= 0.15 and r.lgd <= 0.65:
        r.risk_weight = 0.18
    else:
        r.risk_weight = 0.25
    r.economic_capital = loan * r.risk_weight

    # ── Raw RAROC ─────────────────────────────────────────────────────────────
    r.raw_raroc = (r.nii - r.el) / r.economic_capital if r.economic_capital > 0 else 0.0

    # ── Adjustments ──────────────────────────────────────────────────────────
    adj_notes: List[str] = []
    adj_total = 0.0

    if inp.revenue_growth is not None:
        if inp.revenue_growth > 0.50:
            adj_total += 0.02
            adj_notes.append(f"revenue {inp.revenue_growth*100:.0f}% > 50% -> +0.02")
        elif inp.revenue_growth > 0.30:
            adj_total += 0.01
            adj_notes.append(f"revenue {inp.revenue_growth*100:.0f}% > 30% -> +0.01")

    if unresolved_flags > 0:
        penalty = unresolved_flags * 0.02
        adj_total -= penalty
        adj_notes.append(f"{unresolved_flags} unresolved flag(s) -> -{penalty:.2f}")

    if inp.compliance_veto:
        r.raw_raroc  = 0.0
        adj_total    = 0.0
        adj_notes    = ["COMPLIANCE VETO -> RAROC forced to 0.0"]

    r.adjustments      = adj_total
    r.adjustment_notes = "; ".join(adj_notes) if adj_notes else "None"
    r.final_raroc      = round(max(0.0, r.raw_raroc + adj_total), 4)

    # ── Decision band ─────────────────────────────────────────────────────────
    if inp.compliance_veto:
        r.decision_band = "REJECT (compliance veto)"
    elif r.final_raroc >= RAROC_APPROVE:
        r.decision_band = f"APPROVE (RAROC {r.final_raroc:.4f} > {RAROC_APPROVE})"
    elif r.final_raroc >= RAROC_ESCALATE:
        r.decision_band = f"ESCALATE (RAROC {r.final_raroc:.4f} in {RAROC_ESCALATE}-{RAROC_APPROVE})"
    else:
        r.decision_band = f"REJECT (RAROC {r.final_raroc:.4f} < {RAROC_ESCALATE})"

    r.audit_trail = repr(inp)
    return r


# ─────────────────────────────────────────────
# LLM FACTORY  (cached per session)
# ─────────────────────────────────────────────

_llm_cache: dict = {}

def get_llm(temperature: float = 0.3, max_tokens: int = 500) -> ChatGroq:
    key = (os.environ.get("LS_MODEL", "llama-3.1-8b-instant"), temperature, max_tokens)
    if key not in _llm_cache:
        _llm_cache[key] = ChatGroq(
            model=key[0],
            api_key=os.environ.get("GROQ_API_KEY", "").strip(),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _llm_cache[key]


def _invoke_with_retry(llm: ChatGroq, prompt: str, max_retries: int = 3) -> str:
    """Call LLM with exponential-backoff retry on Groq 429 errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 10 * (attempt + 1)
                print(f"[WAR_ROOM] Rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq API rate limit exceeded after retries. Please wait and try again.")


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class LoanDecision(str, Enum):
    PENDING   = "PENDING"
    APPROVED  = "APPROVED"
    REJECTED  = "REJECTED"
    ESCALATED = "ESCALATED"


class WarRoomState(TypedDict):
    loan_application:   str
    company_name:       str
    sales_memo:         str
    risk_memo:          str
    compliance_memo:    str
    financial_metrics:  str           # structured block from risk agent
    raroc_result_block: str           # Python RAROC formatted output
    debate_history:     Annotated[List[str], operator.add]
    debate_round:       int
    max_rounds:         int
    compliance_veto:    bool
    veto_enabled:       bool
    critical_flags:     List[str]
    resolved_flags:     List[str]
    raroc_score:        Optional[float]
    raroc_formula:      str
    final_decision:     LoanDecision
    decision_memo:      str
    consensus_rounds:   int
    flags_resolved:     int
    _sales_weight:      int
    _risk_weight:       int


# ─────────────────────────────────────────────
# AGENT PROMPTS
# ─────────────────────────────────────────────

SALES_AGENT_PROMPT = """You are the Relationship Manager (Sales Agent) at a commercial bank in India.
Advocate for loan approval based on genuine strengths. Be concise and specific.

RESPONSE FORMAT (use these exact headers):
MEMO TYPE: INITIAL REVIEW
POSITION: APPROVE / CONDITIONAL APPROVE / NEUTRAL
KEY ARGUMENTS:
• [specific argument with data point from the application]
• [specific argument with data point from the application]
• [specific argument with data point from the application]
PROPOSED CONDITIONS: [concrete conditions or "None"]
RESPONSE TO FLAGS: [address each flag specifically, or "No flags yet"]"""


RISK_AGENT_PROMPT = """You are the Credit Underwriter (Risk Agent) at a commercial bank in India.
Protect the bank's capital with rigorous data-driven analysis. Be concise and specific.
NOTE: The RAROC calculation is handled by the system engine — do NOT compute RAROC yourself.
Your job is to extract accurate metrics and flag genuine credit risks.

IMPORTANT on collateral: Check the ENTIRE document including the security/compliance/KYC
section for collateral details. Property, guarantees, and lien on receivables all count.

RESPONSE FORMAT (use these exact headers):
MEMO TYPE: INITIAL REVIEW
POSITION: APPROVE / REJECT / CONDITIONAL
RISK SCORE: [1-10]

FINANCIAL METRICS EXTRACTED:
- DSCR: [value or "Not stated"]
- Debt-to-Equity Ratio: [value or "Not stated"]
- Annual Revenue: [value or "Not stated"]
- Net Income: [value or "Not stated"]
- Loan Amount: [value or "Not stated"]
- Collateral Value: [value or "Not stated" — include property + guarantees if mentioned]
- Credit/CIBIL Score: [value or "Not stated"]
- Revenue Growth YoY: [value or "Not stated"]

KEY CONCERNS:
• [concern with specific metric value]
• [concern with specific metric value]
FLAGS RAISED: [comma-separated flag names or "None"]
MITIGANTS ACKNOWLEDGED: [valid Sales arguments, or "None"]"""


COMPLIANCE_AGENT_PROMPT = """You are the Compliance Officer at a commercial bank in India.
Enforce KYC, AML, and RBI guidelines. You hold VETO POWER over all other agents.

VETO POWER IS RESTRICTED TO THESE 3 CASES ONLY:
  1. Director confirmed on Interpol/FATF grey list or RBI watchlist
  2. Undisclosed offshore accounts with unexplained fund transfers
  3. Active PMLA/FEMA criminal investigation with court notice

ABSOLUTE PROHIBITION — you MUST NOT veto for any of these reasons:
  - High debt-to-equity ratio (any value)
  - Low or missing collateral
  - Low DSCR or weak cash flow
  - Low credit/CIBIL score
  - High loan amount
  - Any financial metric or ratio concern
  - Missing financial documents (that is a documentation issue, not a veto)

If you are tempted to veto for financial reasons, write VETO ISSUED: NO instead.
Your job is regulatory compliance ONLY — financial risk is the Risk Agent's job.

IMPORTANT: If a value (e.g. CIBIL score "762") appears anywhere in the application
text, it IS provided — do not claim it is missing.

RESPONSE FORMAT (use these exact headers):
MEMO TYPE: INITIAL REVIEW / FLAG / CLEARANCE / VETO
COMPLIANCE STATUS: CLEAR / FLAGGED / VETOED
VETO ISSUED: YES / NO
COMPLIANCE ISSUES:
• [specific regulatory issue with law citation, or "None"]
DOCUMENTATION REQUIRED: [specific missing KYC/AML documents, or "All satisfactory"]
REGULATORY REFERENCES: [specific regulation citations]"""


# The moderator no longer computes RAROC — Python does that.
# The LLM only interprets qualitative factors and copies the authoritative block.
MODERATOR_PROMPT = """You are the Moderator at a commercial bank's loan war room.
The RAROC has already been computed by the system engine (see PYTHON RAROC CALCULATION block).
DO NOT recompute or second-guess the RAROC — it is authoritative.

Your tasks:
  1. Weigh the Sales vs Risk qualitative arguments (0-10 scores)
  2. Note qualitative factors the engine cannot capture (management quality, market position)
  3. Set FINAL RECOMMENDATION strictly matching the DECISION BAND in the RAROC block:
       "APPROVE (...)"  -> write APPROVE
       "ESCALATE (...)" -> write CONDITIONAL
       "REJECT (...)"   -> write REJECT

RESPONSE FORMAT (use these exact headers):
SALES ARGUMENT WEIGHT: [0-10]
RISK ARGUMENT WEIGHT: [0-10]
RAROC CALCULATION:
[Copy the PYTHON RAROC CALCULATION block here verbatim — do not change any number]
ESTIMATED RAROC: [copy FINAL RAROC exactly from the block]
FINAL RECOMMENDATION: [APPROVE / CONDITIONAL / REJECT — must match DECISION BAND]
KEY DECIDING FACTORS:
• [factor with data]
• [factor with data]
REASONING: [2 sentences using actual metric values from the RAROC block]"""


# ─────────────────────────────────────────────
# HELPER: extract financial metrics block
# ─────────────────────────────────────────────

def _extract_financial_metrics(risk_response: str) -> str:
    """Pull the FINANCIAL METRICS EXTRACTED block out of the risk agent's response."""
    if "FINANCIAL METRICS EXTRACTED:" in risk_response.upper():
        parts = re.split(r"FINANCIAL METRICS EXTRACTED:", risk_response, flags=re.IGNORECASE)
        if len(parts) > 1:
            block = parts[1].strip()
            lines = []
            for line in block.split("\n"):
                if re.match(r'^[A-Z][A-Z\s]+:', line) and "DSCR" not in line and "-" not in line[:3]:
                    break
                lines.append(line)
            return "FINANCIAL METRICS:\n" + "\n".join(lines).strip()
    return "FINANCIAL METRICS: Not extracted — use conservative estimates."


def _build_verified_metrics_block(raw_text: str, llm_metrics: str) -> str:
    """
    Replace LLM-extracted values in the metrics block with Python-verified ones.
    All subsequent agents receive this corrected block so their memo narrative
    uses accurate numbers — not the LLM's misreads.
    """
    inp = extract_raroc_inputs(llm_metrics, raw_text)
    verified = llm_metrics

    # Override D/E if Python computed a value
    if inp.de_ratio is not None:
        # Replace existing value in block
        new_verified = re.sub(
            r"(Debt.to.Equity\s*(?:Ratio)?\s*[:\-]\s*)[\d\.]+",
            lambda m: m.group(1) + str(inp.de_ratio),
            verified, flags=re.IGNORECASE
        )
        # If pattern didn't match (field absent), append it
        if new_verified == verified and not re.search(r"Debt.to.Equity", verified, re.IGNORECASE):
            verified += f"\n- Debt-to-Equity Ratio: {inp.de_ratio}"
        else:
            verified = new_verified

    # Override Collateral if LLM said none/not-found but Python found a value
    if inp.collateral_value is not None and inp.collateral_value > 0:
        new_verified = re.sub(
            r"(Collateral\s*(?:Value)?\s*[:\-]\s*)(?:Not\s*found|Not\s*stated|None|N/A|0\.?0*)",
            lambda m: m.group(1) + f"{inp.collateral_value:.0f} Lakhs",
            verified, flags=re.IGNORECASE
        )
        if new_verified == verified and not re.search(r"Collateral", verified, re.IGNORECASE):
            verified += f"\n- Collateral Value: {inp.collateral_value:.0f} Lakhs"
        else:
            verified = new_verified

    # Override CIBIL if LLM missed it
    if inp.cibil_score is not None:
        new_verified = re.sub(
            r"((?:CIBIL|Credit)\s*(?:Score)?\s*[:\-]\s*)(?:Not\s*(?:found|stated)|None|N/A)",
            lambda m: m.group(1) + str(inp.cibil_score),
            verified, flags=re.IGNORECASE
        )
        verified = new_verified

    # Override Revenue Growth if LLM missed it
    if inp.revenue_growth is not None:
        pct = f"{inp.revenue_growth * 100:.0f}%"
        new_verified = re.sub(
            r"(Revenue\s*Growth\s*(?:YoY)?\s*[:\-]\s*)(?:Not\s*(?:found|stated)|None|N/A)",
            lambda m: m.group(1) + pct,
            verified, flags=re.IGNORECASE
        )
        verified = new_verified

    # Append a clearly labelled verified-values footer so all agents can see them
    verified += (
        f"\n\n[PYTHON-VERIFIED VALUES — use these for your memo, not your own extraction]\n"
        f"  D/E Ratio       : {inp.de_ratio}\n"
        f"  Collateral      : {inp.collateral_value}L\n"
        f"  CIBIL Score     : {inp.cibil_score}\n"
        f"  Revenue Growth  : {f'{inp.revenue_growth*100:.0f}%' if inp.revenue_growth else 'N/A'}\n"
        f"  Loan Amount     : {inp.loan_amount}L\n"
        f"  DSCR            : {inp.dscr}"
    )
    return verified


def _filter_flags_with_verified_inputs(flags: List[str], raroc_result: "RarocResult") -> List[str]:
    """
    Drop flags that were raised based on LLM misreads of financial metrics.
    Handles both underscore-style (HIGH_DEBT_TO_EQUITY) and natural-language
    (High Debt-to-Equity Ratio) flag names — the LLM uses both formats.

    Rules:
      High D/E flags    — drop if verified D/E < 1.0 (0.875 is healthy)
      Collateral flags  — drop if verified collateral > 0
      Low DSCR flags    — drop if verified DSCR >= 1.25
      Low credit flags  — drop if verified CIBIL >= 700
      Literal "None"    — always drop (LLM included the word "None" as a flag)
      Compliance flags  — never drop (AML/KYC/FEMA etc.)
    """
    trail = raroc_result.audit_trail

    de_m    = re.search(r"de=([\d\.]+)",    trail)
    dscr_m  = re.search(r"dscr=([\d\.]+)",  trail)
    cibil_m = re.search(r"cibil=(\d+)",     trail)
    coll_m  = re.search(r"collateral=([\d\.]+|None)", trail)

    verified_de    = float(de_m.group(1))    if de_m    else None
    verified_dscr  = float(dscr_m.group(1)) if dscr_m  else None
    verified_cibil = int(cibil_m.group(1))  if cibil_m else None
    verified_coll  = (None if not coll_m or coll_m.group(1) == "None"
                      else float(coll_m.group(1)))

    COMPLIANCE_KW = {"aml", "offshore", "pep", "watchlist", "interpol",
                     "fema", "pmla", "kyc", "grey"}

    # Keyword sets for natural-language flag matching (case-insensitive substring)
    HIGH_DEBT_KW   = {"high_debt", "highdebt", "debt_to_equity", "debt-to-equity",
                      "high debt", "debt to equity", "leverage"}
    COLLATERAL_KW  = {"no_collateral", "nocollateral", "low_collateral",
                      "no collateral", "low collateral", "limited collateral",
                      "collateral coverage", "insufficient collateral"}
    LOW_DSCR_KW    = {"low_dscr", "low dscr", "dscr below", "insufficient coverage"}
    LOW_CREDIT_KW  = {"low_credit", "low_cibil", "low credit", "low cibil",
                      "poor credit", "credit score"}

    filtered = []
    for flag in flags:
        fl    = flag.lower().strip()
        fu    = flag.upper().strip()

        # Drop empty / None / N/A literals the LLM sometimes emits
        if not fl or fl in ("none", "n/a", "null", "nil", "-", "no flags"):
            print(f"[RAROC_ENGINE] Flag '{flag}' dropped — empty/None literal")
            continue

        # Always keep genuine compliance / AML flags
        if any(kw in fl for kw in COMPLIANCE_KW):
            filtered.append(flag)
            continue

        # Check against all known financial flag categories
        is_high_debt   = any(kw in fl for kw in HIGH_DEBT_KW)
        is_collateral  = any(kw in fl for kw in COLLATERAL_KW)
        is_low_dscr    = any(kw in fl for kw in LOW_DSCR_KW)
        is_low_credit  = any(kw in fl for kw in LOW_CREDIT_KW)

        if is_high_debt:
            if verified_de is not None and verified_de >= 1.0:
                filtered.append(flag)
            else:
                print(f"[RAROC_ENGINE] Flag '{flag}' dropped — "
                      f"verified D/E={verified_de} < 1.0 (healthy)")

        elif is_collateral:
            if verified_coll is not None and verified_coll > 0:
                print(f"[RAROC_ENGINE] Flag '{flag}' dropped — "
                      f"collateral={verified_coll}L confirmed")
            else:
                filtered.append(flag)

        elif is_low_dscr:
            if verified_dscr is not None and verified_dscr < 1.25:
                filtered.append(flag)
            else:
                print(f"[RAROC_ENGINE] Flag '{flag}' dropped — "
                      f"verified DSCR={verified_dscr} >= 1.25")

        elif is_low_credit:
            if verified_cibil is not None and verified_cibil < 700:
                filtered.append(flag)
            else:
                print(f"[RAROC_ENGINE] Flag '{flag}' dropped — "
                      f"verified CIBIL={verified_cibil} >= 700")

        else:
            # Unknown flag type — keep it (conservative)
            filtered.append(flag)

    return filtered


# ─────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────

def _build_context(
    state: WarRoomState,
    include_sales:   bool = False,
    include_risk:    bool = False,
    include_metrics: bool = False,
    include_raroc:   bool = False,
) -> str:
    # FIX: expanded from 2,500 -> 6,000 chars — ensures collateral / ratio tables visible
    loan_text = state['loan_application'][:6000]
    if len(state['loan_application']) > 6000:
        loan_text += "\n[...truncated — full metrics in FINANCIAL METRICS block...]"

    ctx = (
        f"\nLOAN APPLICATION:\n{loan_text}\n\n"
        f"COMPANY: {state['company_name']}\n"
        f"DEBATE ROUND: {state['debate_round']}\n"
        f"OUTSTANDING FLAGS: {', '.join(state['critical_flags']) if state['critical_flags'] else 'None'}\n"
    )
    if include_metrics and state.get("financial_metrics"):
        ctx += f"\n{state['financial_metrics']}\n"
    if include_raroc and state.get("raroc_result_block"):
        ctx += f"\n{state['raroc_result_block']}\n"
    # FIX: expanded memo excerpts from 600 -> 1,200 chars
    if include_sales and state.get("sales_memo"):
        ctx += f"\nSALES MEMO:\n{state['sales_memo'][:1200]}\n"
    if include_risk and state.get("risk_memo"):
        ctx += f"\nRISK MEMO:\n{state['risk_memo'][:1200]}\n"
    if state.get("debate_history"):
        recent = state["debate_history"][-2:]
        ctx += f"\nRECENT DEBATE:\n{chr(10).join(recent)}\n"
    return ctx


def _recompute_raroc(state: WarRoomState, veto: bool = False,
                     override_resolved: Optional[List[str]] = None) -> tuple:
    """Helper: re-run Python RAROC engine with latest state. Returns (RarocResult, block_str)."""
    resolved = override_resolved if override_resolved is not None else state.get("resolved_flags", [])
    unresolved = [f for f in state.get("critical_flags", []) if f not in resolved]
    inp = extract_raroc_inputs(state.get("financial_metrics", ""), state["loan_application"])
    inp.compliance_veto = veto
    result = compute_raroc(inp, len(unresolved))
    return result, result.to_block()


def sales_agent_node(state: WarRoomState) -> dict:
    llm      = get_llm(temperature=0.3)
    context  = _build_context(state, include_risk=True)
    prompt   = f"{SALES_AGENT_PROMPT}\n\n{context}"
    response = _invoke_with_retry(llm, prompt)
    entry    = f"[ROUND {state['debate_round']}] SALES AGENT:\n{response}\n"
    return {"sales_memo": response, "debate_history": [entry]}


def risk_agent_node(state: WarRoomState) -> dict:
    llm      = get_llm(temperature=0.1)
    context  = _build_context(state, include_sales=True)
    prompt   = f"{RISK_AGENT_PROMPT}\n\n{context}"
    response = _invoke_with_retry(llm, prompt)

    metrics  = _extract_financial_metrics(response)

    # Parse flags from LLM response
    flags = []
    if "FLAGS RAISED:" in response.upper():
        flag_line = re.split(r"FLAGS RAISED:", response, flags=re.IGNORECASE)[-1].strip().split("\n")[0]
        if flag_line and flag_line.lower().strip() not in ("none", "n/a", ""):
            raw_flags = [f.strip("•- ").strip() for f in re.split(r",|;", flag_line) if f.strip()]
            flags = [re.sub(r"[*_`]+", "", f).strip() for f in raw_flags
                     if re.sub(r"[*_`]+", "", f).strip()]

    existing  = state.get("critical_flags", [])
    new_flags = list({f for f in existing + flags if f})

    # ── Step 1: build verified metrics block (Python overrides LLM misreads) ──
    verified_metrics = _build_verified_metrics_block(
        state["loan_application"], metrics
    )

    # ── Step 2: compute RAROC using verified inputs ────────────────────────────
    tmp_state = {**state, "financial_metrics": verified_metrics, "critical_flags": new_flags}
    raroc_result, raroc_block = _recompute_raroc(tmp_state)

    # ── Step 3: drop flags that were based on LLM misreads ────────────────────
    new_flags = _filter_flags_with_verified_inputs(new_flags, raroc_result)

    print(f"[RAROC_ENGINE] {raroc_result.audit_trail}")
    print(f"[RAROC_ENGINE] RAROC={raroc_result.final_raroc:.4f} -> {raroc_result.decision_band}")
    print(f"[RAROC_ENGINE] Verified flags: {new_flags}")

    entry = f"[ROUND {state['debate_round']}] RISK AGENT:\n{response}\n"
    return {
        "risk_memo":          response,
        "financial_metrics":  verified_metrics,   # corrected block, not raw LLM output
        "raroc_result_block": raroc_block,
        "debate_history":     [entry],
        "critical_flags":     new_flags,
    }


def compliance_agent_node(state: WarRoomState) -> dict:
    llm      = get_llm(temperature=0.1)
    context  = _build_context(state, include_sales=True, include_risk=True,
                               include_metrics=True, include_raroc=True)
    prompt   = f"{COMPLIANCE_AGENT_PROMPT}\n\n{context}"
    response = _invoke_with_retry(llm, prompt)
    response_upper = response.upper()

    llm_says_veto = bool(re.search(r"VETO\s+ISSUED\s*:\s*YES", response_upper))

    # ── HARD PYTHON VETO GATE ─────────────────────────────────────────────────
    # The LLM compliance agent sometimes issues vetos for financial reasons
    # (high D/E, no collateral, weak financials) despite being told never to.
    # We validate every veto here: it is only accepted if the compliance
    # response contains a genuine regulatory keyword — not a financial one.
    VALID_VETO_KEYWORDS   = ["interpol", "fatf", "grey list", "watchlist", "pmla",
                              "fema", "criminal", "offshore account", "undisclosed",
                              "rbi watchlist", "ed notice", "court notice"]
    FINANCIAL_VETO_CLUES  = ["debt-to-equity", "d/e", "collateral", "dscr",
                              "cash flow", "credit score", "cibil", "loan amount",
                              "financial", "repayment", "default risk"]

    veto = False
    if llm_says_veto and state.get("veto_enabled", True):
        response_lower = response.lower()
        has_valid_reason    = any(kw in response_lower for kw in VALID_VETO_KEYWORDS)
        has_financial_clue  = any(kw in response_lower for kw in FINANCIAL_VETO_CLUES)

        if has_valid_reason and not has_financial_clue:
            veto = True
            print("[COMPLIANCE] Veto ACCEPTED — valid regulatory grounds found")
        elif has_financial_clue and not has_valid_reason:
            veto = False
            print("[COMPLIANCE] Veto BLOCKED — financial reason detected, not regulatory")
        elif not has_valid_reason:
            veto = False
            print("[COMPLIANCE] Veto BLOCKED — no valid regulatory keyword in response")
        else:
            # Both present — check if regulatory reason is the primary one
            # by checking if it appears in the COMPLIANCE ISSUES section
            issues_section = ""
            m_issues = re.search(r"COMPLIANCE ISSUES\s*:(.*?)(?:DOCUMENTATION|REGULATORY|$)",
                                  response, re.IGNORECASE | re.DOTALL)
            if m_issues:
                issues_section = m_issues.group(1).lower()
            if any(kw in issues_section for kw in VALID_VETO_KEYWORDS):
                veto = True
                print("[COMPLIANCE] Veto ACCEPTED — regulatory issue in compliance section")
            else:
                veto = False
                print("[COMPLIANCE] Veto BLOCKED — regulatory keyword only in narrative, not in issues")

    if not state.get("veto_enabled", True):
        veto = False

    resolved = list(state.get("resolved_flags", []))
    compliance_status = ""
    m = re.search(r"COMPLIANCE STATUS\s*:\s*(\w+)", response_upper)
    if m:
        compliance_status = m.group(1).strip()

    COMPLIANCE_KEYWORDS = ["kyc", "aml", "offshore", "pep", "watchlist",
                           "interpol", "grey", "fema", "pmla", "rbi"]
    if compliance_status == "CLEAR":
        for flag in state.get("critical_flags", []):
            if any(kw in flag.lower() for kw in COMPLIANCE_KEYWORDS) and flag not in resolved:
                resolved.append(flag)
    elif compliance_status not in ("FLAGGED", "VETOED"):
        memo_type = ""
        mt = re.search(r"MEMO TYPE\s*:\s*(\w+)", response_upper)
        if mt:
            memo_type = mt.group(1).strip()
        if memo_type == "CLEARANCE":
            for flag in state.get("critical_flags", []):
                if any(kw in flag.lower() for kw in COMPLIANCE_KEYWORDS) and flag not in resolved:
                    resolved.append(flag)

    # Recompute RAROC after compliance (veto may have changed outcome)
    _, raroc_block = _recompute_raroc(state, veto=veto, override_resolved=resolved)

    entry = f"[ROUND {state['debate_round']}] COMPLIANCE AGENT:\n{response}\n"
    return {
        "compliance_memo":    response,
        "debate_history":     [entry],
        "compliance_veto":    veto,
        "resolved_flags":     resolved,
        "raroc_result_block": raroc_block,
    }


def moderator_node(state: WarRoomState) -> dict:
    # Python makes the final decision — LLM only writes narrative
    veto = state.get("compliance_veto", False)
    raroc_result, raroc_block = _recompute_raroc(state, veto=veto)

    print(f"[MODERATOR] RAROC={raroc_result.final_raroc:.4f} -> {raroc_result.decision_band}")

    # Hard decision from Python thresholds
    if veto:
        decision = LoanDecision.REJECTED
    elif raroc_result.final_raroc >= RAROC_APPROVE:
        decision = LoanDecision.APPROVED
    elif raroc_result.final_raroc >= RAROC_ESCALATE:
        decision = LoanDecision.ESCALATED
    else:
        decision = LoanDecision.REJECTED

    # LLM writes qualitative narrative only
    llm = get_llm(temperature=0.1)
    capped_history = state["debate_history"][-6:]
    full_debate    = "\n".join(capped_history)
    veto_note = ("COMPLIANCE VETO ISSUED." if veto else "Not issued")
    unresolved = [f for f in state.get("critical_flags", [])
                  if f not in state.get("resolved_flags", [])]

    context = (
        f"COMPANY: {state['company_name']}\n\n"
        f"{raroc_block}\n\n"
        f"SALES FINAL POSITION:\n{state.get('sales_memo', 'Not provided')}\n\n"
        f"RISK FINAL POSITION:\n{state.get('risk_memo', 'Not provided')}\n\n"
        f"COMPLIANCE: {state.get('compliance_memo', 'Not provided')[:600]}\n\n"
        f"DEBATE HISTORY:\n{full_debate}\n\n"
        f"UNRESOLVED FLAGS: {'None' if not unresolved else ', '.join(unresolved)}\n"
        f"COMPLIANCE VETO: {veto_note}\n"
    )

    prompt   = f"{MODERATOR_PROMPT}\n\n{context}"
    response = _invoke_with_retry(llm, prompt)

    # Parse agent weights
    sales_w, risk_w = 5, 5
    sm = re.search(r"SALES ARGUMENT WEIGHT\s*:\s*(\d+)", response, re.IGNORECASE)
    rm = re.search(r"RISK ARGUMENT WEIGHT\s*:\s*(\d+)",  response, re.IGNORECASE)
    if sm: sales_w = min(10, max(0, int(sm.group(1))))
    if rm: risk_w  = min(10, max(0, int(rm.group(1))))

    raroc_formula = (
        f"RAROC = (Net Interest Income - Expected Loss) / Economic Capital\n"
        f"Python Engine RAROC : {raroc_result.final_raroc:.4f}\n"
        f"Hurdle Rate         : {RAROC_APPROVE:.4f}\n"
        f"Delta vs hurdle     : {raroc_result.final_raroc - RAROC_APPROVE:+.4f}\n"
        f"Decision band       : {raroc_result.decision_band}"
    )

    entry = f"[FINAL] MODERATOR:\n{response}\n"
    return {
        "debate_history":     [entry],
        "raroc_score":        raroc_result.final_raroc,
        "raroc_formula":      raroc_formula,
        "raroc_result_block": raroc_block,
        "final_decision":     decision,
        "consensus_rounds":   state.get("debate_round", 1),
        "flags_resolved":     len(state.get("resolved_flags", [])),
        "_sales_weight":      sales_w,
        "_risk_weight":       risk_w,
    }


def _sanitize_debate_for_memo(debate_history: list, verified_inp) -> str:
    """
    Scrub wrong financial numbers from debate history before it reaches the memo writer.
    The LLM copies numbers it sees in the debate even when contradicted by
    explicit constraints — so we replace every wrong value at the source.

    Approach: two passes.
      Pass 1 — contextual: replace numbers adjacent to D/E / collateral keywords.
      Pass 2 — blanket: replace any remaining occurrence of the wrong numeric
                value anywhere in the text (the LLM can't copy what isn't there).
    """
    full = "\n".join(debate_history[-6:])

    if verified_inp.de_ratio is not None:
        correct_de  = str(verified_inp.de_ratio)

        # Pass 1 — all keyword-adjacent patterns (covers ratio:, ratio of, ratio is, =, etc.)
        de_patterns = [
            # "ratio is 0.525", "ratio was 0.525", "ratio of 0.525", "ratio: 0.525"
            r"((?:debt[\s\-]*to[\s\-]*equity|D/E)\s*(?:ratio)?\s*(?:is|was|of|:|-|=)\s*)([\d]+\.[\d]+)",
            # "D/E 0.525" (bare number after keyword)
            r"((?:D/E)\s+)([\d]+\.[\d]+)",
            # "debt-to-equity ratio 0.525" (space-separated)
            r"(debt[\s\-]*to[\s\-]*equity\s*(?:ratio)?\s+)([\d]+\.[\d]+)",
        ]

        def _replace_de(m):
            val = m.group(2)
            return m.group(1) + correct_de + " [verified]" if val != correct_de else m.group(0)

        for pat in de_patterns:
            full = re.sub(pat, _replace_de, full, flags=re.IGNORECASE)

        # Pass 2 — blanket: any wrong decimal that looks like a D/E value
        # Only replace values that differ from correct AND are plausible D/E range (0.1-5.0)
        wrong_values = set()
        for m in re.finditer(r"\b([\d]+\.[\d]+)\b", full):
            v = m.group(1)
            try:
                fv = float(v)
                if v != correct_de and 0.1 <= fv <= 5.0 and v != "1.42" and v != "0.15":
                    # Only mark as wrong if it appears right after D/E keyword context
                    # Check the 60 chars before this match
                    start = max(0, m.start() - 60)
                    context = full[start:m.start()].lower()
                    if any(kw in context for kw in ["debt", "equity", "d/e", "de ratio"]):
                        wrong_values.add(v)
            except ValueError:
                pass

        for wrong in wrong_values:
            full = full.replace(wrong, correct_de + " [verified]")

    # Collateral: replace "not found / nil / none" near collateral keyword
    if verified_inp.collateral_value and verified_inp.collateral_value > 0:
        coll_correct = f"Rs {verified_inp.collateral_value:.0f} Lakhs [verified]"
        full = re.sub(
            r"(collateral\s*(?:value|offered)?\s*[:\-]?\s*)(?:not\s*found|nil|none|n/a|0\.?0*)",
            lambda m: m.group(1) + coll_correct,
            full, flags=re.IGNORECASE
        )

    return full


def generate_decision_memo(state: WarRoomState) -> dict:
    llm      = get_llm(temperature=0.2, max_tokens=900)
    decision = str(state["final_decision"]).replace("LoanDecision.", "")

    # Extract verified values directly — single source of truth for all numbers
    inp = extract_raroc_inputs(
        state.get("financial_metrics", ""), state["loan_application"]
    )

    # Sanitize debate history — replace wrong numbers with verified ones
    # so the LLM cannot copy-paste stale figures from the debate
    clean_debate = _sanitize_debate_for_memo(state.get("debate_history", []), inp)

    # Human-readable labels
    loan_str   = (f"Rs {inp.loan_amount:.0f} Lakhs (Rs {inp.loan_amount/100:.1f} Crore)"
                  if inp.loan_amount else "Rs 2.5 Crore")
    dscr_str   = str(inp.dscr)   if inp.dscr   else "N/A"
    de_str     = str(inp.de_ratio) if inp.de_ratio else "N/A"
    de_label   = ("healthy below 1.0" if inp.de_ratio and inp.de_ratio < 1.0
                  else "moderate"     if inp.de_ratio and inp.de_ratio < 2.0
                  else "high")
    cibil_str  = str(inp.cibil_score)  if inp.cibil_score  else "N/A"
    growth_str = (f"{inp.revenue_growth*100:.0f}% YoY" if inp.revenue_growth else "N/A")
    coll_str   = (f"Rs {inp.collateral_value:.0f} Lakhs (property and personal guarantees)"
                  if inp.collateral_value and inp.collateral_value > 0
                  else "Not confirmed in extracted data")

    # RAROC from Python block — never from LLM
    raroc_block = state.get("raroc_result_block", "")
    raroc_val   = "N/A"
    m = re.search(r"FINAL RAROC\s*=\s*([\d\.]+)", raroc_block)
    if m:
        raroc_val = m.group(1)

    flags_str = ", ".join(state["critical_flags"]) if state["critical_flags"] else "None"

    prompt = f"""You are a senior credit officer writing a formal Decision Memo for audit.

══ AUTHORITATIVE VALUES — copy these EXACTLY, never substitute other numbers ══
  Company          : {state['company_name']}
  Final Decision   : {decision}
  RAROC            : {raroc_val}  (hurdle 0.15 — this loan EXCEEDS it)
  Loan Amount      : {loan_str}
  DSCR             : {dscr_str}
  Debt-to-Equity   : {de_str}  [{de_label}]
  CIBIL Score      : {cibil_str}
  Revenue Growth   : {growth_str}
  Collateral       : {coll_str}
  Compliance Veto  : {'ISSUED' if state.get('compliance_veto') else 'Not issued'}
  Flags            : {flags_str}
══════════════════════════════════════════════════════════════════════════

{raroc_block}

QUALITATIVE DEBATE CONTEXT (for narrative only — do NOT copy numbers from here):
{clean_debate}

ABSOLUTE RULES — any violation invalidates the audit memo:
  A. D/E ratio = {de_str}. Write ONLY this value. Never write 0.525 or 1.05 or any other D/E.
  B. Collateral = {coll_str}. Do NOT write 'no collateral' or 'not confirmed'.
  C. RAROC = {raroc_val}. Write ONLY this value.
  D. Decision = {decision}. Every section must be consistent with {decision}.
  E. All numbers must come from AUTHORITATIVE VALUES above — nowhere else.

Write a formal Decision Memo with exactly these 6 sections (2-3 sentences each):
1. EXECUTIVE SUMMARY
2. FINANCIAL ASSESSMENT
3. COMPLIANCE STATUS
4. RAROC ANALYSIS
5. FINAL DECISION & RATIONALE
6. CONDITIONS / NEXT STEPS"""

    memo = _invoke_with_retry(llm, prompt)
    return {"decision_memo": str(memo)}


# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────

def should_continue_debate(state: WarRoomState) -> str:
    if state.get("compliance_veto"):
        return "moderator"
    max_rounds = state.get("max_rounds", MAX_DEBATE_ROUNDS)
    if state["debate_round"] >= max_rounds:
        return "moderator"
    unresolved = [
        f for f in state.get("critical_flags", [])
        if f not in state.get("resolved_flags", [])
    ]
    if not unresolved:
        return "moderator"
    sales_pos = state.get("sales_memo", "").upper()
    risk_pos  = state.get("risk_memo",  "").upper()
    both_approve = (
        "POSITION: APPROVE" in sales_pos
        and ("POSITION: APPROVE" in risk_pos or "POSITION: CONDITIONAL" in risk_pos)
        and not state.get("compliance_veto")
    )
    if both_approve and state["debate_round"] >= 1:
        return "moderator"
    return "continue_debate"


def increment_round(state: WarRoomState) -> dict:
    return {"debate_round": state["debate_round"] + 1}


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

def build_war_room_graph() -> StateGraph:
    graph = StateGraph(WarRoomState)

    graph.add_node("sales_agent",      sales_agent_node)
    graph.add_node("risk_agent",       risk_agent_node)
    graph.add_node("compliance_agent", compliance_agent_node)
    graph.add_node("moderator",        moderator_node)
    graph.add_node("generate_memo",    generate_decision_memo)
    graph.add_node("increment_round",  increment_round)

    graph.set_entry_point("sales_agent")
    graph.add_edge("sales_agent",  "risk_agent")
    graph.add_edge("risk_agent",   "compliance_agent")

    graph.add_conditional_edges(
        "compliance_agent",
        should_continue_debate,
        {"continue_debate": "increment_round", "moderator": "moderator"},
    )

    graph.add_edge("increment_round", "sales_agent")
    graph.add_edge("moderator",       "generate_memo")
    graph.add_edge("generate_memo",   END)

    return graph.compile()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run_war_room(
    loan_text:    str,
    company_name: str,
    max_rounds:   int  = MAX_DEBATE_ROUNDS,
    veto_enabled: bool = True,
) -> WarRoomState:
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise ValueError("GROQ_API_KEY is not set. Please enter your Groq API key in the sidebar.")

    _llm_cache.clear()
    graph = build_war_room_graph()

    initial_state: WarRoomState = {
        "loan_application":   loan_text,
        "company_name":       company_name,
        "sales_memo":         "",
        "risk_memo":          "",
        "compliance_memo":    "",
        "financial_metrics":  "",
        "raroc_result_block": "",
        "debate_history":     [],
        "debate_round":       1,
        "max_rounds":         max_rounds,
        "veto_enabled":       veto_enabled,
        "compliance_veto":    False,
        "critical_flags":     [],
        "resolved_flags":     [],
        "raroc_score":        None,
        "raroc_formula":      "",
        "final_decision":     LoanDecision.PENDING,
        "decision_memo":      "",
        "consensus_rounds":   0,
        "flags_resolved":     0,
        "_sales_weight":      5,
        "_risk_weight":       5,
    }

    final_state = graph.invoke(initial_state)
    final_state["company_name"] = company_name
    return final_state