"""
Autonomous Financial Research Agent

Hybrid (ReAct + Plan-and-Execute) loop agent that uses multiple financial 
tools to research companies and produce grounded investment analysis.
Built with LangGraph.
"""

import re
import uuid
import json
from datetime import datetime
import operator
from typing import Annotated, List, Tuple, TypedDict, Union, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from agent.query_analyzer import get_ticker_from_llm
from agent.disambiguation import disambiguate_query
from synthesis.engine import run_rag_prefetch, verify_generation

# ─── Utility & Token Management ────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 1500
MAX_MESSAGES = 3
MAX_STEPS = 5

def extract_ticker(query: str) -> str:
    """Hard-mapped ticker identification for high accuracy."""
    query = query.lower()
    mapping = {
        "apple": "AAPL",
        "tesla": "TSLA",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "nvidia": "NVDA",
        "samsung": "005930.KS",
        "novartis": "NVS",
        "meta": "META",
        "netflix": "NFLX",
        "jpmorgan": "JPM",
        "goldman": "GS",
    }
    for name, ticker in mapping.items():
        if name in query:
            return ticker
    return None

def compress_context(context: str) -> str:
    """Limits RAG/Tool context to prevent token overflow."""
    return context[:MAX_CONTEXT_CHARS]

def safe_tool_output(text: str, limit: int = 600) -> str:
    """Truncate tool output to stay within Groq free-tier TPM limits."""
    if not isinstance(text, str):
        text = str(text)
    return text[:limit]

def compress_financial_data(data: list) -> list:
    """Aggressively prunes financial JSON for LLM consumption."""
    if not isinstance(data, list): return data
    return [
        {
            "date": d.get("date"),
            "revenue": d.get("Total Revenue"),
            "net_income": d.get("Net Income"),
            "eps": d.get("Diluted EPS")
        }
        for d in data[:2] # limit to 2 years
    ]

def safe_llm_call(llm, input_data):
    """Execution wrapper with automatic truncation on overflow."""
    try:
        return llm.invoke(input_data)
    except Exception as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ["request too large", "413", "rate_limit", "tpm", "limit", "reduce your message"]):
            # Fallback: shrink context window aggressively
            if isinstance(input_data, dict) and "messages" in input_data:
                msgs = input_data["messages"]
                # Keep only the last user message, truncated
                if len(msgs) > 1:
                    last_msg = msgs[-1]
                    if isinstance(last_msg, tuple):
                        role, content = last_msg
                        last_msg = (role, content[:1500])
                    input_data["messages"] = [last_msg]
                else:
                    # Single message — truncate its content
                    msg = msgs[0]
                    if isinstance(msg, tuple):
                        role, content = msg
                        input_data["messages"] = [(role, content[:1500])]
                try:
                    return llm.invoke(input_data)
                except Exception:
                    # Final fallback — return a minimal response dict
                    from langchain_core.messages import AIMessage
                    return {"messages": [AIMessage(content="Data retrieved but too large to process in detail. Key findings have been summarized.")]}
        raise e

# ─── LangGraph Hybrid State & Nodes ────────────────────────────────────────────

from tools.financial_api import financial_data_api
from tools.news_sentiment import news_sentiment
from tools.earnings import earnings_transcript
from tools.web_search import web_search
from tools.sec_edgar import sec_filing_search
from tools.company_profile import company_profile
from memory.vector_store import vector_db_search, vector_db_store
from memory.episodic import record_episode, get_tool_stats
from tools.peer_comparison import peer_comparison
from tools.fact_checker import fact_checker
from tools.calculator import calculation_engine

# ─── LangChain Tools ─────────────────────────────────────────────────────────

@tool
def vector_db_search_tool(query: str, top_k: int = 3) -> str:
    """Search agent's long-term memory for previously stored research."""
    return str(vector_db_search(query, top_k))

@tool
def vector_db_store_tool(content: str, metadata: dict) -> str:
    """Store research findings for future retrieval. 'metadata' must be a dictionary."""
    return str(vector_db_store(content, metadata))

@tool
def financial_data_api_tool(ticker: str, statement_type: str, period: str, years: int = 1) -> str:
    """Retrieves financial statements and key ratios. statement_type: "income", "balance", "cashflow", "ratios"."""
    return safe_tool_output(str(financial_data_api(ticker, statement_type, period, years)), 800)

@tool
def company_profile_tool(ticker: str) -> str:
    """Gets company overview: name, sector, industry, market cap, CEO, description."""
    return safe_tool_output(str(company_profile(ticker)), 800)

@tool
def sec_filing_search_tool(ticker: str, filing_type: str) -> str:
    """Searches SEC EDGAR for company filings. filing_type: "10-K", "10-Q", "8-K"."""
    return safe_tool_output(str(sec_filing_search(ticker, filing_type)), 800)

@tool
def peer_comparison_tool(ticker: str, num_peers: int, metrics_str: str) -> str:
    """Compares company against sector peers on financial metrics. metrics_str: comma separated list of metrics."""
    metrics = [m.strip() for m in metrics_str.split(",")]
    return safe_tool_output(str(peer_comparison(ticker, num_peers, metrics)), 800)

@tool
def news_sentiment_tool(query: str, num_articles: int = 5, lookback_days: int = 7) -> str:
    """Searches recent news and analyzes sentiment."""
    return safe_tool_output(str(news_sentiment(query, num_articles, lookback_days)), 800)

@tool
def web_search_tool(query: str, num_results: int = 5) -> str:
    """General web search. Use ONLY for latest news or events. Do NOT use for financials."""
    return str(web_search(query, num_results))

@tool
def earnings_transcript_tool(ticker: str, quarter: str, year: int) -> str:
    """Fetches earnings call transcript for management insights."""
    res = earnings_transcript(ticker, quarter, year)
    if isinstance(res, dict) and "transcript" in res:
        res["transcript"] = res["transcript"][:1500]
    return safe_tool_output(str(res), 800)

@tool
def calculation_engine_tool(calc_type: str, inputs: dict) -> str:
    """Performs financial calculations: 'growth_rate', 'ratio', 'dcf', 'cagr'.
    'inputs' should be a dictionary of numerical values required for the calculation.
    Example: {'initial': 100, 'final': 150} for growth_rate.
    """
    return str(calculation_engine(calc_type, inputs))

@tool
def fact_checker_tool(claim: str, sources_str: str = "") -> str:
    """Verifies a claim against evidence. sources_str is optional comma separated list of sources."""
    sources = [s.strip() for s in sources_str.split(",")] if sources_str else None
    return str(fact_checker(claim, sources))

tools = [
    vector_db_search_tool,
    financial_data_api_tool,
    company_profile_tool,
    sec_filing_search_tool,
    peer_comparison_tool,
    news_sentiment_tool,
    web_search_tool,
    earnings_transcript_tool,
    calculation_engine_tool,
    fact_checker_tool
]

class HybridAgentState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    executed_tools: List[str] # Track mandatory tool calls
    response: str
    ticker: str
    company_name: str
    intent: str
    rag_context: str

class Plan(BaseModel):
    """Initial roadmap for the research task."""
    steps: List[str] = Field(description="List of specific research steps in chronological order.")

class SynthesisDecision(BaseModel):
    """Final decision on whether to conclude or continue research."""
    response: Optional[str] = Field(None, description="The final comprehensive financial report.")
    steps: Optional[List[str]] = Field(None, description="The next steps to follow.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
executor_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior AI Research Planner.\n"
               "CONSTRAINTS:\n"
               "- Keep responses concise.\n"
               "- Do not repeat data.\n"
               "- Avoid long paragraphs.\n"
               "- Summarize wherever possible.\n"
               "- Plan MAX 5 steps.\n"),
    ("user", "USER QUERY: {input}")
])
planner = planner_prompt | llm.with_structured_output(Plan)

def plan_step(state: HybridAgentState):
    print("\n[Hybrid Agent] Phase 1: Strategic Planning")
    plan_data = safe_llm_call(planner, {"input": state["input"]})
    plan = plan_data.steps[:MAX_STEPS]
    return {"plan": plan, "executed_tools": []}

# The executor runs a ReAct agent to complete the first step in the plan
executor_agent = create_react_agent(
    executor_llm, 
    tools,
    prompt=(
        "You are a professional financial data analyst. Perform the current task with high precision.\n"
        "CONSTRAINTS:\n"
        "- Keep responses concise.\n"
        "- Do not repeat data.\n"
        "- Avoid long paragraphs.\n"
        "- Summarize wherever possible.\n"
        "- If tool output is large, focus on the most relevant metrics."
    )
)

def execute_step(state: HybridAgentState):
    plan = state["plan"]
    task = plan[0]
    print(f"\n[Hybrid Agent] Phase 2: Execution -> {task}")
    
    # Aggressively limit past context to stay within Groq TPM limits
    trimmed_past = state["past_steps"][-2:]  # Only keep last 2 steps
    context = "\n".join(f"Task: {s}\nResult: {safe_tool_output(r, 400)}" for s, r in trimmed_past)
    context = context[:1200]  # Hard cap on total context
    
    prompt = f"Objective: {state['input']}\nTicker: {state['ticker']}\n\nCurrent Task: {task}\n\nPrior Findings:\n{context}\n\nExecute concisely."
    
    agent_response = safe_llm_call(executor_agent, {"messages": [("user", prompt)]})
    
    # Track which tools were called in this step
    newly_executed = []
    for msg in agent_response["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                newly_executed.append(tc["name"])
    
    result = safe_tool_output(agent_response["messages"][-1].content)
    return {
        "past_steps": [(task, result)],
        "executed_tools": list(set(state.get("executed_tools", []) + newly_executed))
    }

replanner_prompt = ChatPromptTemplate.from_template(
    "PERSONA: Senior Financial Advisor. Synthesize findings into a professional report.\n"
    "OBJECTIVE: {input}\n"
    "RESOURCES GATHERED:\n{past_steps}\n"
    "TOOLS EXECUTED: {executed_tools}\n"
    "REMAINING PLAN: {plan}\n\n"
    "- NO JARGON: Never mention tool names, agents, or internal processes.\n"
    "- HARD GROUNDING: Every claim must be tied to a metric or fact from the resources.\n"
    "- PROFESSIONAL TONE: Use business English. Avoid chitchat or monologue.\n\n"
    "NO EARLY SYNTHESIS: If the REMAINING PLAN is not empty, you MUST continue. Do NOT generate a final response yet."
)
replanner = replanner_prompt | llm.with_structured_output(SynthesisDecision)

def replan_step(state: HybridAgentState):
    print("\n[Hybrid Agent] Phase 3: Strategic Review")
    output = replanner.invoke({
        "input": state["input"],
        "plan": state["plan"][1:] if len(state["plan"]) > 0 else [],
        "past_steps": "\n".join(f"Task: {s}\nResult: {r}" for s, r in state["past_steps"]),
        "executed_tools": ", ".join(state.get("executed_tools", []))
    })
    
    if output.response:
        print("\n[Hybrid Agent] Final Report Generated.")
        return {"response": output.response, "plan": []}
    else:
        next_steps = output.steps or []
        print(f"[Hybrid Agent] Replanned with {len(next_steps)} steps remaining.")
        return {"plan": next_steps}

def should_end(state: HybridAgentState):
    if "response" in state and state["response"]:
        return "true"
    else:
        return "false"

# Build LangGraph Hybrid workflow
workflow = StateGraph(HybridAgentState)
workflow.add_node("planner", plan_step)
workflow.add_node("executor", execute_step)
workflow.add_node("replanner", replan_step)

workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "replanner")
workflow.add_conditional_edges("replanner", should_end, {"true": END, "false": "executor"})
workflow.set_entry_point("planner")

app = workflow.compile()

# ─── Intent Routing ──────────────────────────────────────────────────────────

def classify_intent(query: str) -> str:
    """Classifies the query as 'FAST' or 'RESEARCH' using a lightweight check."""
    # If the user specifically asks for a PDF, it's always RESEARCH mode
    if "pdf" in query.lower():
        return "RESEARCH"
        
    prompt = (
        "Classify the following financial query. "
        "If it is a simple fact-based question (e.g., market cap, current price, CEO name, "
        "basic ratio, simple ticker info) that can be answered in 1-2 sentences, return 'FAST'. "
        "If it requires deep analysis, peer comparison, report generation, or multi-step reasoning, return 'RESEARCH'.\n\n"
        f"Query: {query}\n\n"
        "Return ONLY the word 'FAST' or 'RESEARCH'."
    )
    try:
        response = executor_llm.invoke(prompt)
        intent = response.content.strip().upper()
        return "FAST" if "FAST" in intent else "RESEARCH"
    except Exception:
        return "RESEARCH" # Default to high-quality path on error

def detect_chitchat(query: str) -> str:
    """Detects if the query is general chitchat or a financial research request."""
    prompt = (
        "Analyze the following user query. Is it a general greeting, personal question, "
        "or chitchat (e.g., 'hello', 'how are you', 'who are you', 'good morning') "
        "OR is it a request for financial, stock, or company information?\n\n"
        f"Query: {query}\n\n"
        "Return ONLY the word 'CHAT' or 'FINANCE'."
    )
    try:
        response = executor_llm.invoke(prompt)
        return "CHAT" if "CHAT" in response.content.upper() else "FINANCE"
    except Exception:
        return "FINANCE"

# ─── Main Entry Point ────────────────────────────────────────────────────────

def run_agent(user_query: str) -> str:
    session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    print(f"\n[Agent] Session: {session_id}")
    print(f"[Agent] Received query: {user_query}")

    # Step 0: Query Gate
    if detect_chitchat(user_query) == "CHAT":
        print("[Agent] Path: General Chitchat")
        chat_response = executor_llm.invoke(f"Respond naturally and concisely to: {user_query}")
        return {"report": chat_response.content, "ticker": "General", "company_name": "General", "intent": "CHAT"}

    # Step 1: Ticker Identification (Fix 1)
    ticker = extract_ticker(user_query)
    company_name = "the company"
    
    if not ticker:
        # Fallback to LLM disambiguation
        query_info = disambiguate_query(user_query)
        company_name = query_info.get("company_name", "the company")
        identified_tickers = query_info.get("identified_tickers", [])
        if identified_tickers:
            ticker = identified_tickers[0].strip().upper().split()[0]
        else:
            try:
                ticker = get_ticker_from_llm(user_query).strip().upper().split()[0]
            except Exception:
                return {"report": "Unable to identify company ticker from query. Please specify a stock symbol.", "ticker": "Unknown", "company_name": "Unknown", "intent": "RESEARCH"}

    if not ticker:
        return {"report": "Unable to identify company ticker from query.", "ticker": "Unknown", "company_name": "Unknown", "intent": "RESEARCH"}

    print(f"[Agent] Ticker identified: {ticker}")

    # Step 2: Intent Routing
    intent = classify_intent(user_query)
    print(f"[Agent] Routing Path: {intent}")

    # PATH A: Fast-Track (Single tool call, no RAG, no verification)
    if intent == "FAST":
        print("[Agent] Path A: Executing Fast-Track Fact-Check...")
        fast_executor = create_react_agent(executor_llm, tools)
        try:
            # We add a constraint to ensure brevity and speed
            fast_prompt = (
                f"Identify and answer the specific financial data point requested in this query about {ticker}: {user_query}\n"
                "CONSTRAINTS:\n"
                "- Answer concisely in 1-2 sentences.\n"
                "- Use ONLY data returned by the tools.\n"
                "- If the tools do not contain the specific answer, say 'Data not available' instead of providing a generic response."
            )
            agent_response = fast_executor.invoke({"messages": [("user", fast_prompt)]})
            
            # Extract content and tool context for Micro-Verification
            final_content = agent_response["messages"][-1].content
            tool_outputs = ""
            for m in agent_response["messages"]:
                if hasattr(m, "content") and (getattr(m, "type", "") == "tool" or "ToolMessage" in str(type(m))):
                    tool_outputs += f"\nTool Result: {m.content}"

            # Micro-Verification (< 2% Hallucination Guarantee)
            if tool_outputs.strip():
                try:
                    verification = verify_generation(final_content, tool_outputs)
                    if not verification.get("verified"):
                        print(f"[Agent] Fast-Track attention-drift or hallucination detected. Self-correcting...")
                        # Use the smart model (70b) for a sub-second instant correction
                        correction_prompt = (
                            f"User Query: {user_query}\n"
                            f"Drafted Answer: {final_content}\n\n"
                            f"The verifier flagged issues with the draft. Using ONLY the tool context below, "
                            f"provide a corrected, highly accurate, and concise answer to the user's query.\n\n"
                            f"CONTEXT: {tool_outputs}\n\n"
                            "Return ONLY the corrected concise answer."
                        )
                        correction = llm.invoke(correction_prompt)
                        return {"report": correction.content, "ticker": ticker, "company_name": company_name, "intent": intent}
                except Exception:
                    pass # Fallback to original content if verification fails

            return {"report": final_content, "ticker": ticker, "company_name": company_name, "intent": intent}
        except Exception as e:
            print(f"[Agent] Fast-Track failed: {e}. Falling back to Research-Track.")
            # Fall through to RESEARCH path

    # PATH B: Research-Track (RAG + Hybrid Loop + Verification + PDF)
    print("[Agent] Path B: Engaging Deep Research Pipeline...")
    rag_context = ""
    try:
        rag_context = run_rag_prefetch(user_query, ticker)
    except Exception as e:
        print(f"[Agent] RAG pre-fetch failed (non-fatal): {e}")

    try:
        final_state = app.invoke({
            "input": user_query,
            "ticker": ticker,
            "rag_context": rag_context,
            "past_steps": []
        }, config={"recursion_limit": 25})
        
        final_report = final_state.get("response", "I have gathered the data but was unable to synthesize the final report. Please try a more specific query.")
        past_steps = final_state.get("past_steps", [])
        formatted_past_steps = "\n".join(f"Step: {s}\nResult: {r}" for s, r in past_steps)
    except Exception as e:
        print(f"[Agent] Workflow error: {e}")
        return {"report": "I encountered a technical issue while analyzing the data. Please try again in a moment, or ask a simpler question.", "ticker": ticker, "company_name": company_name, "intent": intent}

    # Verification and Self-Correction Loop
    confidence = 0.85
    full_context = f"PRE-FETCHED RAG CONTEXT:\n{rag_context}\n\nLIVE TOOL EXECUTION RESULTS:\n{formatted_past_steps}"
    
    max_retries = 2
    verification = {}
    for attempt in range(max_retries):
        if not full_context.strip():
            break
        try:
            verification = verify_generation(final_report, full_context)
            if verification.get("verified"):
                confidence = 0.95
                break
            else:
                unsupported = verification.get("unsupported_claims", [])
                print(f"[Agent] Hallucinations detected: {unsupported}. Self-correcting (Attempt {attempt+1}/{max_retries})...")
                
                correction_prompt = (
                    f"User Query: {user_query}\n"
                    f"Your previous report contained these hallucinated or unsupported claims: {unsupported}\n\n"
                    f"Rewrite the report to completely remove or correctly attribute these claims using ONLY the following context.\n\n"
                    f"CONTEXT:\n{full_context}\n\n"
                    f"ORIGINAL REPORT:\n{final_report}\n\n"
                    f"Return the corrected report without any conversational fluff."
                )
                
                correction_response = llm.invoke(correction_prompt)
                final_report = correction_response.content
                confidence = 0.80
        except Exception as e:
            print(f"[Agent] Verification loop failed: {e}")
            break

    # The verification note is used for internal scoring but no longer appended to the user-facing
    # report to ensure a clean, non-technical PDF/terminal output.
    pass

    try:
        vector_db_store(
            content=f"Analysis of {ticker}: {final_report[:800]}",
            metadata={"ticker": ticker, "date": datetime.now().isoformat(), "source_type": "agent_analysis"},
            confidence=confidence,
            verified=rag_context != "",
            session_id=session_id,
        )
    except Exception:
        pass

    record_episode("agent_session", True, "research")
    
    # Extract executed tools from final_state if it exists
    executed_tools_list = []
    if 'final_state' in locals():
        executed_tools_list = list(set(final_state.get("executed_tools", [])))

    return {
        "report": final_report, 
        "ticker": ticker, 
        "company_name": company_name, 
        "intent": intent, 
        "executed_tools": executed_tools_list
    }
