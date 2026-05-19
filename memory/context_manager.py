MAX_CONTEXT_CHARS = 12000  # Summarize short-term memory when exceeded

def summarize_short_term_memory(messages: list) -> list:
    """
    Short-Term Memory Management.

    When context grows too large, summarize earlier observations
    to keep within LLM context window limits. Keeps the system prompt
    and recent messages intact, summarizing middle observations.
    """
    total_chars = sum(len(m.get("content", "")) for m in messages)

    if total_chars <= MAX_CONTEXT_CHARS:
        return messages

    # Keep: system prompt (idx 0), original query (idx 1), last 4 messages
    preserved_start = messages[:2]
    preserved_end = messages[-4:]
    middle = messages[2:-4]

    if not middle:
        return messages

    # Summarize the middle observations
    summary_parts = []
    for msg in middle:
        content = msg.get("content", "")
        role = msg.get("role", "")
        if role == "user" and content.startswith("Observation:"):
            # Keep first 300 chars of each observation
            summary_parts.append(content[:300])
        elif role == "assistant":
            # Keep just the Thought + Action line
            lines = content.split("\n")
            key_lines = [l for l in lines if l.startswith(("Thought:", "Action:"))]
            summary_parts.append("\n".join(key_lines) if key_lines else content[:150])

    summarized_msg = {
        "role": "user",
        "content": f"[Summary of earlier research steps]:\n" + "\n---\n".join(summary_parts),
    }

    return preserved_start + [summarized_msg] + preserved_end
