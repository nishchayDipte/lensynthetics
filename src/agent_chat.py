"""
LendSynthetix — Agent Chat Interface
Allows loan officers to ask follow-up questions to any agent post-decision.
Each agent stays in character using the war room debate context.
"""

import os
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def get_llm() -> ChatGroq:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Please enter your key in the sidebar.")
    return ChatGroq(
        model=os.environ.get("LS_MODEL", "llama-3.1-8b-instant"),
        api_key=key,
        temperature=0.4,
        max_tokens=600,
    )


AGENT_PERSONAS = {
    "Sales Agent": {
        "icon": "💼",
        "color": "#3B82F6",
        "system": """You are the Relationship Manager (Sales Agent) in a bank's loan war room.
You championed this loan application and argued for approval.
Stay in character — you are optimistic, relationship-focused, and growth-oriented.
Reference the war room debate and decision when answering follow-up questions.
Keep answers concise (3-5 sentences max). Be helpful to the loan officer reviewing the case.""",
    },
    "Risk Agent": {
        "icon": "⚖️",
        "color": "#F59E0B",
        "system": """You are the Credit Underwriter (Risk Agent) in a bank's loan war room.
You analyzed the financial metrics and evaluated credit risk in this application.
Stay in character — you are analytical, skeptical, and data-driven.
Reference the war room debate and decision when answering follow-up questions.
Keep answers concise (3-5 sentences max). Cite specific financial metrics where possible.""",
    },
    "Compliance Agent": {
        "icon": "🛡️",
        "color": "#10B981",
        "system": """You are the Compliance Officer in a bank's loan war room.
You reviewed this application for KYC/AML and regulatory compliance issues.
Stay in character — you are procedural, precise, and non-negotiable on regulations.
Reference the war room debate and decision when answering follow-up questions.
Keep answers concise (3-5 sentences max). Cite specific regulations or flags where relevant.""",
    },
    "Moderator": {
        "icon": "🎯",
        "color": "#8B5CF6",
        "system": """You are the Moderator who resolved the war room debate and computed the RAROC score.
You have a balanced, objective view of all three agents' positions.
You can explain the final decision rationale, RAROC calculation, and any conditions attached.
Keep answers concise (3-5 sentences max). Be clear and auditor-friendly in your explanations.""",
    },
}


def ask_agent(
    agent_name: str,
    question: str,
    war_room_result: Dict[str, Any],
    chat_history: List[Dict],
) -> str:
    """
    Ask a specific agent a follow-up question about the loan decision.
    Returns the agent's response string.
    """
    persona  = AGENT_PERSONAS.get(agent_name, AGENT_PERSONAS["Moderator"])
    decision = str(war_room_result.get("final_decision", "PENDING")).replace("LoanDecision.", "")
    raroc    = war_room_result.get("raroc_score", 0.0)
    company  = war_room_result.get("company_name", "the applicant")
    veto     = war_room_result.get("compliance_veto", False)
    flags    = war_room_result.get("critical_flags", [])
    debate   = "\n".join(war_room_result.get("debate_history", [])[-6:])

    context = f"""
WAR ROOM CONTEXT:
Company: {company}
Final Decision: {decision}
RAROC Score: {f'{raroc:.4f}' if raroc else 'N/A'}
Compliance Veto: {'Yes' if veto else 'No'}
Critical Flags: {', '.join(flags) if flags else 'None'}

RECENT DEBATE EXCERPT:
{debate}
"""

    messages = [SystemMessage(content=persona["system"] + "\n\n" + context)]

    # Build conversation history with correct message types
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["message"]))
        else:
            # Agent's own previous replies must be AIMessage, not HumanMessage
            messages.append(AIMessage(content=msg["message"]))

    messages.append(HumanMessage(content=question))

    try:
        llm      = get_llm()
        response = llm.invoke(messages).content.strip()
        return response
    except ValueError as e:
        return f"⚠️ {e}"
    except Exception as e:
        return f"[Agent unavailable: {str(e)}]"


def get_suggested_questions(agent_name: str, decision: str) -> List[str]:
    """Return suggested questions based on agent role and decision outcome."""
    base = {
        "Sales Agent": [
            "What are the strongest growth signals in this application?",
            "How does this borrower compare to similar clients?",
            "What conditions would make you more confident?",
        ],
        "Risk Agent": [
            "What is the biggest financial red flag here?",
            "How was the DSCR calculated?",
            "What collateral would strengthen this application?",
        ],
        "Compliance Agent": [
            "Which KYC documents are still missing?",
            "Is there an AML concern I should escalate?",
            "What regulation applies to the offshore flag?",
        ],
        "Moderator": [
            "Why was this RAROC score assigned?",
            "What would change the final decision?",
            "Summarize the key debate points for my file note.",
        ],
    }

    if decision == "REJECTED":
        base["Moderator"].append("What would need to change for approval?")
    elif decision == "ESCALATED":
        base["Moderator"].append("Who should review this escalated case?")

    return base.get(agent_name, base["Moderator"])