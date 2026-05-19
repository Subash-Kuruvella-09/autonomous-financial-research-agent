"""
Autonomous Financial Research Agent — Main Entry Point

Commands:
  python main.py                    — Interactive research mode
  python main.py challenge 1       — Run challenge 1
  python main.py challenge all     — Run all 8 challenges
  python main.py challenge evaluate — Evaluate completed challenges (RQB)
  python main.py challenge score   — Show leaderboard
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Route to interactive mode or challenge mode based on arguments."""
    args = sys.argv[1:]

    if args and args[0] == "challenge":
        _handle_challenge(args[1:])
    else:
        _interactive_mode()


# ─── Interactive Research Mode ────────────────────────────────────────────────

def _interactive_mode():
    """Interactive loop to query the financial research agent."""
    from agent.core import run_agent

    print("=" * 60)
    print("  AUTONOMOUS FINANCIAL RESEARCH AGENT (ARA-1)")
    print("=" * 60)
    print()
    print("  Tools: 11 (memory, financials, news, earnings,")
    print("         SEC filings, peers, fact-check, calc)")
    print("  RAG:   6-stage pipeline (pre-fetch + post-verify)")
    print()
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'challenge' to see challenge commands.")
    print("=" * 60)

    while True:
        print()
        user_input = input(">> Your query: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if user_input.lower() == "challenge":
            print("\n  Challenge commands (run from terminal):")
            print("    python main.py challenge 1       — Run challenge 1-8")
            print("    python main.py challenge all     — Run all 8")
            print("    python main.py challenge evaluate — Score with RQB")
            print("    python main.py challenge score   — Leaderboard")
            continue

        from tools.report_gen import generate_report
        report_data = generate_report(user_input, user_input) # Ticker resolution is handled inside generate_report/run_agent
        
        result = report_data.get("report", "")
        ticker = report_data.get("ticker", "Agent")
        company_name = report_data.get("company_name", ticker)
        intent = report_data.get("intent", "DETAILED")

        print("\n" + "=" * 60)
        print("  AGENT REPORT")
        print("=" * 60)
        print()
        print(result)
        print()
        print("=" * 60)

        # PDF Generation Hook
        if intent == "PDF" or "pdf" in user_input.lower():
            if result.startswith("I encountered a technical issue") or result.startswith("I have gathered the data"):
                print("\n[WARNING] Skipping PDF generation due to incomplete analysis.")
            else:
                print("\n[Generating professional stylized PDF...]")
                try:
                    from utils.pdf_generator import generate_pdf
                    pdf_path = generate_pdf(result, company_name)
                    print(f"[OK] PDF successfully generated and saved to:\n   {pdf_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to generate PDF: {e}")


# ─── Challenge Mode ──────────────────────────────────────────────────────────

def _handle_challenge(args: list):
    """Handle challenge subcommands."""
    if not args:
        _print_challenge_help()
        return

    cmd = args[0].lower()

    if cmd == "all":
        from evaluation.benchmarks.challenges import run_all_challenges
        run_all_challenges()

    elif cmd == "evaluate":
        from evaluation.metrics import evaluate_all
        evaluate_all()

    elif cmd == "score":
        from evaluation.benchmarks.challenges import show_scores
        show_scores()

    elif cmd == "help":
        _print_challenge_help()

    elif cmd.isdigit():
        num = int(cmd)
        if 1 <= num <= 8:
            from evaluation.benchmarks.challenges import run_challenge
            result = run_challenge(num)

            # Auto-evaluate after running
            if result and "error" not in result:
                print("\n[Auto-evaluating with Research Quality Board...]")
                from evaluation.metrics import evaluate_challenge
                evaluate_challenge(num)
        else:
            print(f"Invalid challenge number: {num}. Choose 1-8.")
    else:
        print(f"Unknown command: {cmd}")
        _print_challenge_help()


def _print_challenge_help():
    """Print challenge command help."""
    print("\n" + "=" * 60)
    print("  🏆 RESEARCH CHALLENGES")
    print("=" * 60)
    print()
    print("  Usage: python main.py challenge <command>")
    print()
    print("  Commands:")
    print("    1-8        Run a specific challenge (auto-evaluates)")
    print("    all        Run all 8 challenges sequentially")
    print("    evaluate   Score all completed challenges with RQB")
    print("    score      Show the leaderboard")
    print()
    print("  Challenges:")
    print("    1. Single-Company Profile     (★☆☆☆☆)")
    print("    2. Earnings Analysis           (★★☆☆☆)")
    print("    3. Risk Assessment             (★★☆☆☆)")
    print("    4. Industry Comparison         (★★★☆☆)")
    print("    5. Contradictory Data          (★★★☆☆)")
    print("    6. Ambiguous Query             (★★★★☆)")
    print("    7. Sector Analysis + Memory    (★★★★☆)")
    print("    8. Full Report + Degradation   (★★★★★)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()