"""Claude backend: one call shape (system + text + images -> validated pydantic object), with the same safety rails
as core.argon: a request bundle per call, a content-addressed cache so identical requests never re-hit the API,
a usage ledger in .worldgym/api_ledger.jsonl, and a runaway-loop guard (not a spend limit).

Claude takes images, not video: clips go in as contact sheets. The key is ANTHROPIC_API_KEY from the environment,
the repo .env or the parent folder's .env.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / ".worldgym"
CACHE = STATE / "claude_cache"
MODEL = "claude-opus-5-5"
USD_PER_M_IN, USD_PER_M_OUT = 4.0, 20.0
MAX_CALLS = int(os.environ.get("WORLDGYM_MAX_CALLS", 400))

T = TypeVar("T", bound=BaseModel)

for _p in (REPO / ".env", REPO.parent / ".env"):
    if _p.exists():
        load_dotenv(_p, override=False)


class LLMUnavailable(Exception):
    pass


_live_calls: list[str] = []


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _image_block(p: Path) -> dict:
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": mime,
                                        "data": base64.standard_b64encode(p.read_bytes()).decode()}}


def _key(system: str, prompt: str, images: list[Path], schema: type[BaseModel], effort: str) -> str:
    h = hashlib.sha256()
    for part in (MODEL, system, prompt, json.dumps(schema.model_json_schema(), sort_keys=True), effort):
        h.update(part.encode())
    for p in images:
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:24]


def _ledger(name: str, usage, ok: bool, seconds: float) -> None:
    tin = getattr(usage, "input_tokens", 0) or 0
    tin += (getattr(usage, "cache_read_input_tokens", 0) or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    tout = getattr(usage, "output_tokens", 0) or 0
    row = {"day": date.today().isoformat(), "time": datetime.now().strftime("%H:%M:%S"), "kind": "model",
           "name": name, "model": MODEL, "ok": ok, "input_tokens": tin, "output_tokens": tout, "seconds": round(seconds, 1),
           "usd": round(tin / 1e6 * USD_PER_M_IN + tout / 1e6 * USD_PER_M_OUT, 5)}
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "api_ledger.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def ask(name: str, system: str, prompt: str, schema: type[T], images: list[str | Path] = (),
        bundle_dir: str | Path | None = None, effort: str = "high") -> T:
    """One structured call. Cached by content; writes system.md, prompt.md, images and response.json to bundle_dir."""
    images = [Path(p) for p in images]
    key = _key(system, prompt, images, schema, effort)
    if bundle_dir:
        b = Path(bundle_dir) / name
        b.mkdir(parents=True, exist_ok=True)
        (b / "system.md").write_text(system)
        (b / "prompt.md").write_text(prompt + "\n\nimages: " + ", ".join(p.name for p in images))
    cached = CACHE / f"{key}.json"
    if cached.exists():
        out = schema.model_validate_json(cached.read_text())
    else:
        if not available():
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set (environment, .env or ../.env)")
        if len(_live_calls) >= MAX_CALLS or _live_calls.count(name) >= 4:
            raise LLMUnavailable(f"loop guard: {len(_live_calls)} live calls, {name!r} x{_live_calls.count(name)}")
        _live_calls.append(name)
        import anthropic
        client = anthropic.Anthropic()
        content = [_image_block(p) for p in images] + [{"type": "text", "text": prompt}]
        t0 = time.time()
        with client.messages.stream(model=MODEL, max_tokens=32000, system=system,
                                    messages=[{"role": "user", "content": content}],
                                    output_config={"effort": effort, "format": {
                                        "type": "json_schema", "schema": _strict_schema(schema)}}) as stream:
            msg = stream.get_final_message()
        ok = msg.stop_reason == "end_turn"
        _ledger(name, msg.usage, ok, time.time() - t0)
        if msg.stop_reason == "refusal":
            raise LLMUnavailable(f"{name}: refused")
        text = next(blk.text for blk in msg.content if blk.type == "text")
        out = schema.model_validate_json(text)
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(out.model_dump_json())
    if bundle_dir:
        (Path(bundle_dir) / name / "response.json").write_text(out.model_dump_json(indent=1))
    return out


def _strict_schema(schema: type[BaseModel]) -> dict:
    """Pydantic JSON schema in the shape structured outputs accept: every property required (optional ones nullable),
    additionalProperties false, and no numeric bounds (validated by pydantic afterwards)."""
    s = schema.model_json_schema()

    def fix(node):
        if isinstance(node, dict):
            for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "default", "title"):
                node.pop(k, None)
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
    fix(s)
    return s
