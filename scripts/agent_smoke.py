"""Smoke-test one Antigravity Managed Agent: inline source mount, sandbox execution, JSON report back.

    uv run python scripts/agent_smoke.py
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.argon import ARGON, Argon, check_budget, parse_json, record_spend  # noqa: E402
from core.swarm import AGENT, AGENT_MAX_TOKENS, ALLOWLIST  # noqa: E402

PROGRAM = """import json, platform, sys
print(json.dumps({"python": sys.version.split()[0], "machine": platform.machine(), "sum": sum(range(101))}))
"""


def main() -> None:
    client = Argon(".worldgym/agent_smoke").client
    env = {"type": "remote",
           "sources": [{"type": "inline", "target": "worldgym/probe.py", "encoding": "base64",
                        "content": base64.b64encode(PROGRAM.encode()).decode()}],
           "network": {"allowlist": [{"domain": d} for d in ALLOWLIST]}}
    prompt = ("Run `python3 worldgym/probe.py` in the sandbox and report its JSON output. Then reply with ONE JSON "
              'object: {"ran": true, "probe": <the JSON it printed>, "cwd_listing": [files in the working directory]}.')
    check_budget("agent", "agent_smoke")
    t0 = time.time()
    it = client.interactions.create(agent=AGENT, agent_config={"type": "antigravity", "model": ARGON,
                                                               "max_total_tokens": AGENT_MAX_TOKENS},
                                    input=prompt, environment=env, background=True)
    record_spend("agent", "agent_smoke", AGENT, None, True, it.id)
    print("launched", it.id, it.status, flush=True)
    while it.status in ("in_progress", "queued"):
        time.sleep(10)
        it = client.interactions.get(it.id)
        print(f"  {time.time() - t0:5.0f}s {it.status}", flush=True)
        if time.time() - t0 > 900:
            client.interactions.cancel(it.id)
            break
    u = it.usage.model_dump() if getattr(it, "usage", None) is not None else {}
    record_spend("agent_usage", "agent_smoke", AGENT, u, it.status == "completed", it.id)
    text = it.output_text or ""
    print("status", it.status, "env", getattr(it, "environment_id", None), "seconds", round(time.time() - t0))
    print("tokens", {k: u.get(k) for k in ("total_input_tokens", "total_output_tokens", "total_thought_tokens")})
    try:
        print("report", json.dumps(parse_json(text)))
    except Exception:  # noqa: BLE001
        print("raw output", text[:1500])
    if getattr(it, "errors", None):
        print("errors", it.errors)


if __name__ == "__main__":
    main()
