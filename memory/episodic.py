import json
import os
from datetime import datetime

EPISODIC_FILE = os.path.join(os.path.dirname(__file__), "..", "episodic_memory.json")

def record_episode(tool_name: str, success: bool, query_type: str = ""):
    """Record a tool usage episode for learning."""
    try:
        if os.path.exists(EPISODIC_FILE):
            with open(EPISODIC_FILE, "r", encoding="utf-8") as f:
                episodes = json.load(f)
        else:
            episodes = []

        episodes.append({
            "tool": tool_name,
            "success": success,
            "query_type": query_type,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only last 200 episodes to avoid unbounded growth
        episodes = episodes[-200:]

        with open(EPISODIC_FILE, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2)

    except Exception:
        pass  # non-critical


def get_tool_stats() -> dict:
    """Get success rates per tool from episodic memory."""
    try:
        if not os.path.exists(EPISODIC_FILE):
            return {}

        with open(EPISODIC_FILE, "r", encoding="utf-8") as f:
            episodes = json.load(f)

        stats = {}
        for ep in episodes:
            tool = ep["tool"]
            if tool not in stats:
                stats[tool] = {"total": 0, "success": 0}
            stats[tool]["total"] += 1
            if ep.get("success"):
                stats[tool]["success"] += 1

        # Compute success rates
        for tool in stats:
            total = stats[tool]["total"]
            stats[tool]["success_rate"] = round(
                stats[tool]["success"] / total, 2
            ) if total > 0 else 0

        return stats

    except Exception:
        return {}
