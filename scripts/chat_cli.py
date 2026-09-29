"""Interactive terminal chat with the agent, printing the ReAct trace.

Usage:
    python scripts/chat_cli.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenthumb.agent import GreenThumbAgent  # noqa: E402
from greenthumb.config import get_settings  # noqa: E402


def main() -> None:
    """Read messages from stdin until 'exit' and print structured answers."""
    agent = GreenThumbAgent.from_settings(get_settings())
    session_id = None
    print("GreenThumb assistant - scrivi 'exit' per uscire.")
    while True:
        message = input("\nTu: ").strip()
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        result = agent.chat(message, session_id=session_id)
        session_id = result.trace.session_id
        for call in result.trace.tool_calls:
            print(f"  [step {call.step}] {call.tool}({json.dumps(call.arguments, ensure_ascii=False)}) -> {call.status}")
        print(json.dumps(result.response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
