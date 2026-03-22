"""
LendSynthetix -- RAG Pipeline (pdfplumber + Groq direct)
No llama_index dependency — works with just pdfplumber and langchain-groq.

Extracts text and tables from financial PDFs, then uses Groq to answer
structured extraction questions directly from the text.
"""

import os
import re
import json
import pdfplumber
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False

from langchain_groq import ChatGroq


# ── LLM helper ───────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.environ.get("LS_MODEL", "llama-3.1-8b-instant"),
        api_key=os.environ.get("GROQ_API_KEY", "").strip(),
        temperature=0.0,
        max_tokens=300,
    )


# ── PDF text extraction ───────────────────────────────────────────────────────

def _clean_pdf_text(text: str) -> str:
    """
    Clean common PDF encoding corruption before any parsing occurs.

    Problems fixed:
      1. HTML entity &#8377; (rupee symbol ₹ as decimal codepoint) -> Rs
         This appears when PDFs embed the Unicode codepoint literally
      2. Bare number 8377 adjacent to currency context -> Rs
         pdfplumber sometimes strips the &#...; wrapper leaving just 8377
      3. Other common HTML entities (&#x20B9; hex form, &amp;, &lt; etc.)
      4. Garbled ligatures and common PDF extraction artifacts
    """
    import html

    # Step 1: decode all HTML entities (&amp; &lt; &#8377; &#x20B9; etc.)
    text = html.unescape(text)

    # Step 2: replace ₹ symbol with "Rs" for reliable ASCII parsing
    text = text.replace("₹", "Rs ")
    text = text.replace("\u20b9", "Rs ")   # Unicode codepoint directly

    # Step 3: replace bare "8377" that appears right before a space+number
    # Pattern: "8377 42.00" -> "Rs 42.00"  (pdfplumber dropped the HTML wrapper)
    import re
    text = re.sub(r"\b8377\s+", "Rs ", text)
    text = re.sub(r"\b8377\.", "Rs 0.",  text)  # "8377.00" edge case

    # Step 4: common PDF ligature fixes
    text = text.replace("\ufb01", "fi")   # ﬁ ligature
    text = text.replace("\ufb02", "fl")   # ﬂ ligature
    text = text.replace("\u2013", "-")    # en dash
    text = text.replace("\u2014", "-")    # em dash
    text = text.replace("\u2019", "'")    # right single quote
    text = text.replace("\u201c", '"')    # left double quote
    text = text.replace("\u201d", '"')    # right double quote

    return text


def _extract_text_pdfplumber(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                pt = page.extract_text()
                if pt:
                    text += f"\n--- Page {i} ---\n{pt}\n"
    except Exception as e:
        print(f"[RAG] pdfplumber text error: {e}")
    return _clean_pdf_text(text)


def _extract_text_pypdf2(pdf_path: str) -> str:
    if not _HAS_PYPDF2:
        return ""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                pt = page.extract_text()
                if pt:
                    text += f"\n--- Page {i} ---\n{pt}\n"
    except Exception as e:
        print(f"[RAG] PyPDF2 error: {e}")
    return _clean_pdf_text(text)


def _extract_tables_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    tables_found = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for t_idx, table in enumerate(page.extract_tables() or []):
                    if not table:
                        continue
                    md_lines = []
                    for row_idx, row in enumerate(table):
                        clean = [_clean_pdf_text(str(c or "")).strip() for c in row]
                        md_lines.append(" | ".join(clean))
                        if row_idx == 0:
                            md_lines.append(" | ".join(["---"] * len(clean)))
                    tables_found.append({
                        "page":        page_num,
                        "table_index": t_idx,
                        "markdown":    "\n".join(md_lines),
                        "raw_rows":    table,
                    })
    except Exception as e:
        print(f"[RAG] Table extraction error: {e}")
    return tables_found


def extract_full_document(pdf_path: str) -> Dict[str, Any]:
    print(f"[RAG] Extracting: {pdf_path}")
    tables = _extract_tables_pdfplumber(pdf_path)
    text   = _extract_text_pdfplumber(pdf_path)

    if not text.strip() and _HAS_PYPDF2:
        print("[RAG] Falling back to PyPDF2...")
        text = _extract_text_pypdf2(pdf_path)

    table_section = ""
    if tables:
        table_section = "\n\n=== EXTRACTED TABLES ===\n"
        for t in tables:
            table_section += (
                f"\n[Table {t['table_index']+1} -- Page {t['page']}]\n"
                f"{t['markdown']}\n"
            )

    combined = text + table_section
    print(f"[RAG] Done: {len(text):,} chars text + {len(tables)} tables")
    return {"text": text, "tables": tables, "combined": combined}


# ── Extraction questions ──────────────────────────────────────────────────────

EXTRACTION_QUERIES = {
    "company_name":         "What is the name of the company applying for the loan?",
    "loan_amount":          "What is the total loan amount requested?",
    "loan_purpose":         "What is the purpose of this loan?",
    "annual_revenue":       "What is the annual revenue or total sales? Include fiscal year.",
    "net_income":           "What is the net income or net profit? Include fiscal year.",
    "ebitda":               "What is the EBITDA value?",
    "total_assets":         "What is the total assets value from the balance sheet?",
    "total_liabilities":    "What is the total liabilities amount from the balance sheet?",
    "total_debt":           "What is the total debt or borrowings?",
    "equity":               "What is the total equity or net worth?",
    "debt_to_equity_ratio": "What is the debt-to-equity ratio? If not stated, calculate: Total Debt / Equity.",
    "dscr":                 "What is the Debt Service Coverage Ratio (DSCR)?",
    "collateral":           "What collateral or security is offered? Include property value, personal guarantees, lien on receivables, and total primary security value.",
    "collateral_coverage":  "What is the collateral coverage ratio or total security value relative to the loan amount?",
    "cash_flow":            "What is the operating cash flow or net cash from operations?",
    "years_in_business":    "How many years has the company been in business? When was it founded?",
    "directors":            "Who are the directors, promoters, or key management personnel?",
    "offshore_accounts":    "Are there any offshore bank accounts, foreign deposits, or overseas entities?",
    "aml_flags":            "Are there any AML concerns, suspicious transactions, or grey list mentions?",
    "recurring_contracts":  "What are the recurring revenue contracts or long-term clients?",
    "growth_rate":          "What is the revenue growth rate year-over-year?",
    "credit_score":         "What is the credit score, CIBIL score, or credit rating mentioned?",
    "existing_loans":       "What existing loans, EMIs, or debt obligations does the company have?",
}


# ── Groq-based metric extraction (no vector index needed) ────────────────────

def _ask_groq(full_text: str, question: str) -> str:
    """Ask Groq a direct question about the document text."""
    # Expanded from 6000 -> 10000 chars — ensures security/collateral section is included
    truncated = full_text[:10000]
    prompt = (
        f"You are a financial document analyst. Answer the following question "
        f"based ONLY on the document text provided. Give a concise, specific answer "
        f"(one sentence or a number). If the information is not in the document, "
        f"say 'Not found'.\n\n"
        f"DOCUMENT TEXT:\n{truncated}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    try:
        llm = _get_llm()
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"Extraction error: {e}"


# ── Risk flag detection ───────────────────────────────────────────────────────

def _detect_risk_flags(metrics: Dict[str, Any]) -> List[str]:
    flags = []

    de = re.search(r"([\d.]+)", str(metrics.get("debt_to_equity_ratio", "")))
    if de and float(de.group(1)) > 2.5:
        flags.append("HIGH_DEBT_TO_EQUITY")

    dscr = re.search(r"([\d.]+)", str(metrics.get("dscr", "")))
    if dscr and float(dscr.group(1)) < 1.25:
        flags.append("LOW_DSCR")

    offshore = str(metrics.get("offshore_accounts", "")).lower()
    if any(w in offshore for w in ["offshore", "cayman", "bvi", "mauritius", "emirates"]):
        flags.append("OFFSHORE_ACCOUNT_DETECTED")

    aml = str(metrics.get("aml_flags", "")).lower()
    if any(w in aml for w in ["unusual", "suspicious", "cash", "grey list", "interpol", "unexplained"]):
        flags.append("POTENTIAL_AML_FLAG")

    credit = str(metrics.get("credit_score", ""))
    score = re.search(r"(\d{3})", credit)
    if score and int(score.group(1)) < 700:
        flags.append("LOW_CREDIT_SCORE")

    directors = str(metrics.get("directors", "")).lower()
    if any(w in directors for w in ["grey list", "interpol", "watchlist", "pep"]):
        flags.append("DIRECTOR_ON_WATCHLIST")

    return flags


# ── Main parser class ─────────────────────────────────────────────────────────

class FinancialDocumentParser:
    """
    PDF parser using pdfplumber for extraction and Groq for metric extraction.
    No llama_index or vector index required.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.raw_data: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self._text_ready = False

    def _ensure_text(self):
        if self._text_ready:
            return
        self.raw_data = extract_full_document(self.pdf_path)
        if not self.raw_data["combined"].strip():
            raise ValueError(
                "Could not extract text from this PDF. "
                "It may be a scanned image — use a text-based PDF."
            )
        self._text_ready = True

    def extract_metric(self, question: str) -> str:
        self._ensure_text()
        return _ask_groq(self.raw_data["combined"], question)

    def extract_all_metrics(self) -> Dict[str, Any]:
        self._ensure_text()

        print("[RAG] Extracting metrics via Groq...")

        # Batch all questions into ONE Groq call to save time and API cost
        questions_block = "\n".join(
            f"{i+1}. {q}" for i, (_, q) in enumerate(EXTRACTION_QUERIES.items())
        )
        keys = list(EXTRACTION_QUERIES.keys())

        prompt = (
            f"You are a financial document analyst. Extract the following information "
            f"from the document. Answer each question with a short, specific value. "
            f"If not found, write 'Not found'.\n\n"
            f"DOCUMENT TEXT:\n{self.raw_data['combined'][:10000]}\n\n"
            f"Answer each question on its own line in the format:\n"
            f"1. <answer>\n2. <answer>\netc.\n\n"
            f"QUESTIONS:\n{questions_block}\n\nANSWERS:"
        )

        try:
            llm = ChatGroq(
                model=os.environ.get("LS_MODEL", "llama-3.1-8b-instant"),
                api_key=os.environ.get("GROQ_API_KEY", "").strip(),
                temperature=0.0,
                max_tokens=1200,
            )
            raw_answers = llm.invoke(prompt).content.strip()

            # Parse numbered answers
            lines = [l.strip() for l in raw_answers.split("\n") if l.strip()]
            for i, key in enumerate(keys):
                # Match "1. answer" or just the answer if parsing is loose
                answer = ""
                for line in lines:
                    m = re.match(rf"^{i+1}[\.\)]\s*(.+)", line)
                    if m:
                        answer = m.group(1).strip()
                        break
                self.metrics[key] = answer if answer else "Not found"

        except Exception as e:
            print(f"[RAG] Batch extraction failed: {e} — falling back to individual queries")
            for name, question in EXTRACTION_QUERIES.items():
                print(f"  -> {name}")
                self.metrics[name] = _ask_groq(self.raw_data["combined"], question)

        # Metadata — raw text extended to 8000 chars to cover ratio tables (page 7+)
        self.metrics["_raw_text_preview"] = self.raw_data["text"][:8000]
        self.metrics["_tables_found"]     = len(self.raw_data["tables"])
        self.metrics["_table_previews"]   = [
            {"page": t["page"], "preview": t["markdown"][:300]}
            for t in self.raw_data["tables"][:5]
        ]
        self.metrics["_risk_flags"] = _detect_risk_flags(self.metrics)
        self.metrics["company_name"] = self.metrics.get("company_name", "Unknown")

        if self.metrics["_risk_flags"]:
            print(f"[RAG] Risk flags: {self.metrics['_risk_flags']}")

        return self.metrics

    def to_summary_text(self) -> str:
        n_tables = self.metrics.get("_tables_found", 0)
        flags    = self.metrics.get("_risk_flags", [])

        lines = [
            f"COMPANY:              {self.metrics.get('company_name', 'Unknown')}",
            f"LOAN AMOUNT:          {self.metrics.get('loan_amount', 'N/A')}",
            f"LOAN PURPOSE:         {self.metrics.get('loan_purpose', 'N/A')}",
            f"PDF TABLES FOUND:     {n_tables}",
            "",
            "=== FINANCIAL METRICS ===",
            f"Annual Revenue:       {self.metrics.get('annual_revenue', 'N/A')}",
            f"Net Income:           {self.metrics.get('net_income', 'N/A')}",
            f"EBITDA:               {self.metrics.get('ebitda', 'N/A')}",
            f"Total Assets:         {self.metrics.get('total_assets', 'N/A')}",
            f"Total Liabilities:    {self.metrics.get('total_liabilities', 'N/A')}",
            f"Total Debt:           {self.metrics.get('total_debt', 'N/A')}",
            f"Equity / Net Worth:   {self.metrics.get('equity', 'N/A')}",
            f"Debt-to-Equity Ratio: {self.metrics.get('debt_to_equity_ratio', 'N/A')}",
            f"DSCR:                 {self.metrics.get('dscr', 'N/A')}",
            f"Operating Cash Flow:  {self.metrics.get('cash_flow', 'N/A')}",
            f"Revenue Growth (YoY): {self.metrics.get('growth_rate', 'N/A')}",
            f"Credit / CIBIL Score: {self.metrics.get('credit_score', 'N/A')}",
            f"Existing Loans/EMIs:  {self.metrics.get('existing_loans', 'N/A')}",
            "",
            "=== BUSINESS PROFILE ===",
            f"Years in Business:    {self.metrics.get('years_in_business', 'N/A')}",
            f"Key Directors:        {self.metrics.get('directors', 'N/A')}",
            f"Collateral Offered:   {self.metrics.get('collateral', 'N/A')}",
            f"Collateral Coverage:  {self.metrics.get('collateral_coverage', 'N/A')}",
            f"Recurring Contracts:  {self.metrics.get('recurring_contracts', 'N/A')}",
            "",
            "=== COMPLIANCE INDICATORS ===",
            f"Offshore Accounts:    {self.metrics.get('offshore_accounts', 'N/A')}",
            f"AML / Unusual Txns:   {self.metrics.get('aml_flags', 'N/A')}",
            "",
            "=== AUTO-DETECTED RISK FLAGS ===",
        ]

        lines += [f"  WARNING: {f}" for f in flags] if flags else ["  None detected"]

        previews = self.metrics.get("_table_previews", [])
        if previews:
            lines += ["", "=== EXTRACTED TABLE PREVIEWS ==="]
            for p in previews:
                lines.append(f"\n[Page {p['page']}]\n{p['preview']}")

        # Append raw PDF text so downstream parsers (extract_raroc_inputs) can
        # read ratio tables, balance sheet rows, and security details directly.
        # This is the source of truth — covers data that Groq LLM may misread.
        raw_text = self.metrics.get("_raw_text_preview", "")
        if raw_text:
            lines += ["", "=== RAW PDF TEXT (source of truth for key ratios) ===", raw_text]

        return "\n".join(lines)


# ── Fallback: plain text input ────────────────────────────────────────────────

def parse_text_application(text: str) -> Dict[str, Any]:
    """Regex-based extraction for pasted text (no PDF needed)."""
    metrics = {}
    patterns = {
        "annual_revenue":       r"(?:revenue|sales|turnover)[:\s]*[Rs.₹$]?([\d,\.]+\s*(?:cr|lakh|crore|million|billion)?)",
        "loan_amount":          r"(?:loan amount|credit facility|loan request)[:\s]*[Rs.₹$]?([\d,\.]+\s*(?:cr|lakh|crore|million)?)",
        "debt_to_equity_ratio": r"debt.to.equity(?:[\s\-]*ratio)?[:\s]*([\d\.]+)",
        "dscr":                 r"dscr[:\s]*([\d\.]+)",
        "net_income":           r"(?:net income|net profit)[:\s]*[Rs.₹$]?([\d,\.]+\s*(?:cr|lakh|crore|million)?)",
        "ebitda":               r"ebitda[:\s]*[Rs.₹$]?([\d,\.]+\s*(?:cr|lakh|crore|million)?)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        metrics[key] = m.group(0) if m else "Not found"

    aml_keywords = ["offshore", "cayman", "bvi", "mauritius", "interpol", "grey list",
                    "suspicious", "unexplained"]
    metrics["aml_flags"] = [k for k in aml_keywords if k.lower() in text.lower()]
    metrics["_risk_flags"] = _detect_risk_flags(metrics)
    return metrics