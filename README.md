# ⚔️ LendSynthetix — The Loan War Room

> **Multi-Agent Orchestration for Commercial Lending**  
> Hackathon Project | Stack: LangGraph + HuggingFace Inference API

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOAN APPLICATION (PDF)                    │
│                    rag_pipeline.py                           │
│        PyPDF2 → LlamaIndex RAG → Structured Metrics         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               LANGGRAPH STATE MACHINE (war_room.py)          │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐      │
│  │  SALES   │───▶│  RISK    │───▶│   COMPLIANCE      │      │
│  │  AGENT   │    │  AGENT   │    │   AGENT (VETO)    │      │
│  └──────────┘    └──────────┘    └─────────┬─────────┘      │
│       ▲                                    │                  │
│       └────────── debate loop ─────────────┘                 │
│                    (max 3 rounds)          │                  │
│                                           ▼                  │
│                              ┌──────────────────────┐        │
│                              │     MODERATOR        │        │
│                              │  (RAROC Computation) │        │
│                              └──────────┬───────────┘        │
└─────────────────────────────────────────┼─────────────────── ┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   DECISION MEMO       │
                              │   (Audit Export)      │
                              └───────────────────────┘
```

### Consensus Algorithm

- **Compliance Veto**: Compliance Officer can reject regardless of Sales/Risk
- **Weighted RAROC**:
  - `RAROC ≥ 0.15` → APPROVED
  - `0.08 ≤ RAROC < 0.15` → ESCALATED (human review)
  - `RAROC < 0.08` → REJECTED

---

## 📁 Project Structure

```
lendsynthetix/
├── src/
│   ├── war_room.py         # Main LangGraph orchestration (Phase 3)
│   ├── rag_pipeline.py     # PDF parsing + RAG extraction (Phase 1)
│   └── app.py              # Streamlit dashboard UI
├── sample_data/
│   └── sample_inputs.py    # Two test cases (tech startup + manufacturing)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set HuggingFace token

```bash
export HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
```

Get your **free token** at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Run the dashboard

```bash
cd src
streamlit run app.py
```

### 4. Run CLI directly

```bash
# Test with sample data
python src/war_room.py

# Parse a PDF
python src/rag_pipeline.py path/to/loan_application.pdf
```

---

## 🤖 Agent Personas

### Agent 1 — Sales Agent (Relationship Manager)

- **Goal**: Maximize loan approval
- **Behavior**: Highlights growth, recurring revenue, and business potential
- **Tools**: CRM history, market sentiment analysis, contract pipeline data

### Agent 2 — Risk Agent (Credit Underwriter)

- **Goal**: Protect bank capital
- **Behavior**: Skeptical, data-driven; focuses on DSCR, D/E ratio, collateral
- **Raises Flags**: HIGH_DEBT_TO_EQUITY, LOW_DSCR, INSUFFICIENT_COLLATERAL
- **Risk Score**: 1–10 per evaluation

### Agent 3 — Compliance Agent (Auditor)

- **Goal**: Zero regulatory violations (KYC/AML/Fair Lending)
- **Behavior**: Procedural and rigid; blocks any AML/FATF red flags
- **Has VETO POWER** — overrides all other agents
- **Raises Flags**: OFFSHORE_ACCOUNT, FATF_GREY_LIST, MISSING_KYC

### Moderator — Conflict Resolution

- **Goal**: Compute RAROC and break Sales vs Risk ties
- **Output**: Final weighted recommendation + RAROC score

---

## 📊 LangGraph State Schema

```python
class WarRoomState(TypedDict):
    loan_application: str      # Raw extracted text
    company_name: str
    sales_memo: str            # Agent memos
    risk_memo: str
    compliance_memo: str
    debate_history: List[str]  # Full debate transcript
    debate_round: int          # Current round (max 3)
    compliance_veto: bool      # Veto flag
    critical_flags: List[str]  # Unresolved flags
    resolved_flags: List[str]  # Cleared flags
    raroc_score: float         # 0.0 – 1.0
    final_decision: LoanDecision
    decision_memo: str         # Human-readable audit output
```

---

## 🧪 Test Scenarios

### Sample A — Tech Startup (Tests Sales vs Risk)

- **Company**: NexaCloud Technologies Pvt. Ltd.
- **Profile**: High growth (+47% YoY), no physical collateral, borderline DSCR 1.18
- **Expected**: Heated debate; Sales wins with recurring contract evidence

### Sample B — Manufacturing Firm (Tests Compliance Veto)

- **Company**: Bharat Steel & Forge Industries Ltd.
- **Profile**: Excellent financials (D/E 0.28x, DSCR 2.87), BUT director on FATF grey list + unexplained ₹1Cr offshore deposit
- **Expected**: Compliance VETO despite strong financials

---

## 📈 Success Metrics

| Metric              | Target                                         |
| ------------------- | ---------------------------------------------- |
| Logic Consistency   | Same red flags caught every run                |
| Consensus Speed     | Decision within 3 debate rounds                |
| Explainability      | Human auditor can read and understand the memo |
| Compliance Accuracy | 100% veto on grey-list / AML cases             |

---

## 🛠️ Tech Stack

| Component           | Technology                                               |
| ------------------- | -------------------------------------------------------- |
| Agent Orchestration | LangGraph 0.2+                                           |
| LLM                 | Mistral-7B-Instruct via HuggingFace Inference API (free) |
| Embeddings          | sentence-transformers/all-MiniLM-L6-v2 (free)            |
| RAG Pipeline        | LlamaIndex + FAISS (local)                               |
| PDF Parsing         | PyPDF2                                                   |
| UI Dashboard        | Streamlit                                                |
| State Storage       | LangGraph in-memory (Redis optional for prod)            |

---

## 📄 Output — Decision Memo Example

```
COMMERCIAL LENDING DECISION MEMO
=================================
Company: NexaCloud Technologies Pvt. Ltd.
Date: [auto-generated]
Loan Amount: ₹8 Crore
Decision: APPROVED (Conditional)

1. EXECUTIVE SUMMARY
The War Room reached consensus after 2 debate rounds...

2. SALES ASSESSMENT
Agent highlighted 47% YoY growth and 82% recurring revenue...

3. RISK ASSESSMENT
Risk Agent flagged borderline DSCR of 1.18 and lack of physical collateral...

4. COMPLIANCE STATUS
CLEAR — All KYC/AML checks passed. No watchlist hits.

5. FINAL DECISION
CONDITIONALLY APPROVED | RAROC: 0.162
Condition: Monthly cash flow monitoring for first 12 months

6. CONDITIONS
- Assign recurring contracts as primary collateral
- Personal guarantees from both founders
- Quarterly DSCR review covenant
```

---

## 🏆 Hackathon Deliverables

- [x] Phase 1: RAG Pipeline (`rag_pipeline.py`)
- [x] Phase 2: Agent Persona Prompts (in `war_room.py`)
- [x] Phase 3: LangGraph State Machine (`war_room.py`)
- [x] Phase 4: Audit Export + Decision Memo (in `app.py` + `war_room.py`)
- [x] Streamlit Dashboard with live debate visualization
- [x] JSON audit log export
- [x] Two sample datasets matching hackathon test cases
