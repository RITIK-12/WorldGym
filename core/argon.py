"""One adapter for every model call, with three backends selected by WORLDGYM_LLM:

- gemini : live Interactions API (background=True + polling, so "high" thinking never hits an HTTP timeout)
- replay : read <replay_dir>/<name>.json (stage-outage safety net; record once the demo video works)
- manual : write the request bundle and stop, so a human/other agent can answer it

Every request writes a bundle runs/<id>/argon/<name>/{system.md, prompt.md, attachments/, response.json}.
JSON hygiene: ask for ONE object, strip code fences, validate with pydantic, re-ask once with the validation error.

Runaway-loop guard (live backend only; NOT a spend limit). Identical requests are served from a content-addressed
cache and never re-hit the API. A process trips the guard when it makes more than WORLDGYM_MAX_CALLS (default 400)
live calls, issues the same request name more than WORLDGYM_MAX_REPEAT (default 4) times, or launches more than
WORLDGYM_MAX_AGENTS (default 16) Managed Agents. Every call is logged with its token usage to
.worldgym/api_ledger.jsonl for the scorecard.

Argon runs at thinking_level "high" with the full 65,536-token output window by default (thinking + answer share
it). Media calls are synchronous; text-only calls run in background mode and may ask for more than 65,536 tokens.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

REPO = Path(__file__).resolve().parents[1]

ARGON = "gemini-4-argon"
FLASH = "gemini-3.8-flash"
LITE = "gemini-3.5-flash-lite"
ROBOTICS_ER = "gemini-robotics-er-2-preview"   # spatial pointing / boxes: the dense-labelling subagent
REPLAY_DIR = Path(os.environ.get("WORLDGYM_REPLAY_DIR", REPO / "replay_cache"))
STATE = Path(os.environ.get("WORLDGYM_STATE", REPO / ".worldgym"))
INLINE_VIDEO_MAX = 18 * 1024 * 1024
POLL_S = 3.0
TIMEOUT_S = 20 * 60
DEFAULT_MAX_OUTPUT = 65536
USD_PER_M_IN, USD_PER_M_OUT = 2.0, 10.0
RETRYABLE = ("invalid json", "malformed", "please retry", "internal error", "unavailable", "deadline")
DROPPED = ("server disconnected", "connection reset", "connection aborted", "remote end closed")

T = TypeVar("T", bound=BaseModel)


def _load_env() -> None:
    """Repo .env first, then the parent folder's; accept the key under any capitalisation of GEMINI_API_KEY."""
    for p in (REPO / ".env", REPO.parent / ".env"):
        if p.exists():
            load_dotenv(p, override=False)
    if not os.environ.get("GEMINI_API_KEY"):
        for k, v in list(os.environ.items()):
            if k.strip().lower() == "gemini_api_key" and v.strip():
                os.environ["GEMINI_API_KEY"] = v.strip().strip("'\"")
                break


_load_env()


class ManualPending(Exception):
    """manual backend: the bundle is written; put response.json in it and re-run."""


class ArgonError(Exception):
    pass


class LoopGuardTripped(ArgonError):
    """A runaway loop was detected (too many live calls / repeats / agents in one process)."""


BudgetExceeded = LoopGuardTripped


@dataclass
class CallLog:
    name: str
    model: str
    backend: str
    seconds: float
    ok: bool
    cached: bool = False
    usd: float = 0.0


CALLS: list[CallLog] = []   # process-wide call log for the scorecard ("model calls")
AGENT_LAUNCHES: list[str] = []
_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------------------------------
# Runaway-loop guard + usage ledger
# ----------------------------------------------------------------------------------------------------------------------

def _caps() -> dict:
    return {"max_calls": int(os.environ.get("WORLDGYM_MAX_CALLS", 400)),
            "max_repeat": int(os.environ.get("WORLDGYM_MAX_REPEAT", 4)),
            "max_agents": int(os.environ.get("WORLDGYM_MAX_AGENTS", 16))}


def _ledger_path() -> Path:
    return STATE / "api_ledger.jsonl"


def ledger_today() -> list[dict]:
    p = _ledger_path()
    if not p.exists():
        return []
    today = date.today().isoformat()
    rows = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("day") == today:
            rows.append(r)
    return rows


def spend_summary() -> dict:
    rows = ledger_today()
    live = [c for c in CALLS if c.backend == "gemini" and not c.cached]
    return {"day": date.today().isoformat(), "model_calls_today": sum(r["kind"] == "model" for r in rows),
            "agents_today": sum(r["kind"] == "agent" for r in rows),
            "usd_est_today": round(sum(r.get("usd", 0) for r in rows), 3),
            "process_live_calls": len(live), "process_agents": len(AGENT_LAUNCHES), "guards": _caps()}


def check_budget(kind: str = "model", name: str | None = None) -> None:
    """Trip on runaway loops only: total live calls, repeats of one request name, agent launches (per process)."""
    caps = _caps()
    with _LOCK:
        live = [c for c in CALLS if c.backend == "gemini" and not c.cached]
        if kind == "model" and len(live) >= caps["max_calls"]:
            raise LoopGuardTripped(f"{len(live)} live calls in this process (WORLDGYM_MAX_CALLS={caps['max_calls']})")
        if kind == "model" and name and sum(c.name == name for c in live) >= caps["max_repeat"]:
            raise LoopGuardTripped(f"request {name!r} repeated {caps['max_repeat']}x (WORLDGYM_MAX_REPEAT)")
        if kind == "agent" and len(AGENT_LAUNCHES) >= caps["max_agents"]:
            raise LoopGuardTripped(f"{len(AGENT_LAUNCHES)} agents launched (WORLDGYM_MAX_AGENTS={caps['max_agents']})")
        if kind == "agent":
            AGENT_LAUNCHES.append(name or "agent")


def record_spend(kind: str, name: str, model: str, usage: dict | None, ok: bool, interaction_id: str | None = None) -> float:
    u = usage or {}
    tin = u.get("total_input_tokens") or 0
    tout = (u.get("total_output_tokens") or 0) + (u.get("total_thought_tokens") or 0)
    usd = round(tin / 1e6 * USD_PER_M_IN + tout / 1e6 * USD_PER_M_OUT, 5)
    STATE.mkdir(parents=True, exist_ok=True)
    row = {"day": date.today().isoformat(), "time": datetime.now().strftime("%H:%M:%S"), "kind": kind, "name": name,
           "model": model, "ok": ok, "input_tokens": tin, "output_tokens": tout, "usd": usd, "id": interaction_id}
    with _ledger_path().open("a") as f:
        f.write(json.dumps(row) + "\n")
    return usd


def _usage_dict(it) -> dict:
    u = getattr(it, "usage", None)
    if u is None:
        return {}
    return u.model_dump() if hasattr(u, "model_dump") else dict(u)


# ----------------------------------------------------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith(("{", "[")):
        i = min([k for k in (t.find("{"), t.find("[")) if k >= 0], default=-1)
        if i >= 0:
            t = t[i:]
    return t


def parse_json(text: str):
    t = strip_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # tolerate trailing prose after the object
        depth, end = 0, None
        for k, ch in enumerate(t):
            depth += ch in "{["
            depth -= ch in "}]"
            if depth == 0 and ch in "}]":
                end = k + 1
                break
        if end is None:
            raise
        return json.loads(t[:end])


# ----------------------------------------------------------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------------------------------------------------------

class Argon:
    def __init__(self, bundle_root: str | Path, backend: str | None = None, replay_dir: str | Path | None = None):
        self.root = Path(bundle_root)
        self.backend = backend or os.environ.get("WORLDGYM_LLM", "gemini")
        self.replay_dir = Path(replay_dir) if replay_dir else REPLAY_DIR
        self.use_cache = os.environ.get("WORLDGYM_CACHE", "1") != "0"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            if not os.environ.get("GEMINI_API_KEY"):
                raise ArgonError("GEMINI_API_KEY is not set (put it in .env) or use WORLDGYM_LLM=replay")
            self._client = genai.Client()
        return self._client

    def _bundle(self, name: str, system: str, prompt: str, attachments: list[Path]) -> Path:
        b = self.root / name
        att = b / "attachments"
        if att.exists():
            shutil.rmtree(att)  # stale keyframes must never leak into a new request
        att.mkdir(parents=True, exist_ok=True)
        (b / "system.md").write_text(system)
        (b / "prompt.md").write_text(prompt)
        for p in attachments:
            shutil.copy2(p, att / Path(p).name)
        return b

    def _parts(self, prompt: str, attachments: list[Path]) -> list[dict]:
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for p in attachments:
            p = Path(p)
            mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            kind = "video" if mime.startswith("video/") else "image"
            parts.append({"type": "text", "text": f"[attachment: {p.name}]"})
            if kind == "video" and p.stat().st_size > INLINE_VIDEO_MAX:
                parts.append({"type": "video", "uri": self._upload(p, mime), "mime_type": mime})
            else:
                parts.append({"type": kind, "data": base64.b64encode(p.read_bytes()).decode(), "mime_type": mime})
        return parts

    def _upload(self, p: Path, mime: str) -> str:
        f = self.client.files.upload(file=str(p), config={"mime_type": mime})
        t0 = time.time()
        while f.state is not None and f.state.name == "PROCESSING":
            if time.time() - t0 > 300:
                raise ArgonError(f"upload of {p.name} stuck in PROCESSING")
            time.sleep(2)
            f = self.client.files.get(name=f.name)
        if f.state is not None and f.state.name == "FAILED":
            raise ArgonError(f"upload of {p.name} failed: {f.error}")
        return f.uri

    def _create(self, name: str, model: str, kw: dict):
        try:
            return self.client.interactions.create(**kw)
        except Exception:
            record_spend("model", name, model, None, False)
            raise

    def _gemini(self, name: str, model: str, system: str, parts: list[dict], thinking: str, json_mode: bool,
                previous_id: str | None, max_output_tokens: int, background: bool) -> tuple[str, str, dict]:
        check_budget("model", name)
        kw = dict(model=model, input=parts, system_instruction=system, background=background,
                  generation_config={"thinking_level": thinking, "max_output_tokens": max_output_tokens})
        if not background:
            kw["timeout"] = TIMEOUT_S
        if previous_id:
            kw["previous_interaction_id"] = previous_id
        # JSON is enforced by the prompt contract + pydantic validation (+ one re-ask); the server requires a full
        # response_format schema whenever response_mime_type is set, which pydantic schemas do not always satisfy.
        for attempt in range(3):
            try:
                it = self._create(name, model, kw)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if attempt < 2 and any(k in msg.lower() for k in DROPPED):
                    continue        # the connection dropped mid-call: send the same request again
                if attempt == 2 or not any(k in msg.lower() for k in RETRYABLE):
                    raise
                # the server asks for the error to be included in the retry prompt
                kw["input"] = [*kw["input"], {"type": "text", "text": f"\n\nThe previous attempt failed with: "
                                              f"{msg[:400]}\nReply with ONE valid JSON object as plain text."}]
                if attempt == 1:
                    kw.pop("previous_interaction_id", None)  # last resort: a fresh turn
        t0 = time.time()
        while it.status in ("in_progress", "queued"):
            if time.time() - t0 > TIMEOUT_S:
                self.client.interactions.cancel(it.id)
                record_spend("model", name, model, None, False, it.id)
                raise ArgonError(f"interaction {it.id} timed out")
            time.sleep(POLL_S)
            it = self.client.interactions.get(it.id)
        usage = _usage_dict(it)
        record_spend("model", name, model, usage, it.status == "completed", it.id)
        if it.status == "incomplete" and not getattr(self, "_retrying", False):
            # out of output budget: background calls can go past 65,536 tokens; synchronous ones think less instead
            self._retrying = True
            try:
                more = min(max_output_tokens * 2, 262144) if background else max_output_tokens
                return self._gemini(name, model, system, parts, thinking if background else "medium", json_mode,
                                    previous_id, more, background)
            finally:
                self._retrying = False
        if it.status != "completed":
            raise ArgonError(f"interaction {it.id} ended {it.status}: {getattr(it, 'errors', None)}")
        return it.output_text or "", it.id, usage

    @staticmethod
    def _cache_key(model, system, prompt, attachments, thinking, json_mode, previous_id, max_output_tokens) -> str:
        h = hashlib.sha256()
        h.update(json.dumps([model, system, prompt, thinking, json_mode, previous_id, max_output_tokens]).encode())
        for p in attachments:
            h.update(hashlib.sha256(Path(p).read_bytes()).digest())
        return h.hexdigest()[:32]

    def ask_text(self, name: str, system: str, prompt: str, attachments: list[str | Path] = (), model: str = ARGON,
                 thinking: str = "high", json_mode: bool = False, previous_id: str | None = None,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT, background: bool | None = None) -> tuple[str, dict]:
        attachments = [Path(a) for a in attachments]
        if background is None:  # background serving rejects image/video input; media calls run synchronously
            background = not attachments
        b = self._bundle(name, system, prompt, attachments)
        resp_path = b / "response.json"
        t0 = time.time()
        cached = False
        if self.backend == "replay":
            src = self.replay_dir / f"{name}.json"
            if not src.exists():
                raise ArgonError(f"replay backend: no cached response {src}")
            resp = json.loads(src.read_text())
        elif self.backend == "manual":
            if resp_path.exists():
                resp = json.loads(resp_path.read_text())
            else:
                raise ManualPending(str(b))
        elif self.backend == "gemini":
            key = self._cache_key(model, system, prompt, attachments, thinking, json_mode, previous_id,
                                  max_output_tokens)
            cpath = STATE / "cache" / f"{key}.json"
            if self.use_cache and cpath.exists():
                resp, cached = json.loads(cpath.read_text()), True
            else:
                try:
                    text, iid, usage = self._gemini(name, model, system, self._parts(prompt, attachments), thinking,
                                                    json_mode, previous_id, max_output_tokens, background)
                except Exception:
                    with _LOCK:
                        CALLS.append(CallLog(name, model, self.backend, time.time() - t0, False))
                    raise
                resp = {"text": text, "interaction_id": iid, "model": model, "usage": usage,
                        "seconds": round(time.time() - t0, 1), "thinking_level": thinking}
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps(resp, indent=1))
                REPLAY_DIR.mkdir(parents=True, exist_ok=True)
                (REPLAY_DIR / f"{name}.json").write_text(json.dumps(resp, indent=1))
        else:
            raise ArgonError(f"unknown backend {self.backend!r}")
        resp.setdefault("backend", self.backend)
        resp["cached"] = cached
        resp_path.write_text(json.dumps(resp, indent=1))
        u = resp.get("usage") or {}
        usd = ((u.get("total_input_tokens") or 0) * USD_PER_M_IN
               + ((u.get("total_output_tokens") or 0) + (u.get("total_thought_tokens") or 0)) * USD_PER_M_OUT) / 1e6
        with _LOCK:
            CALLS.append(CallLog(name, resp.get("model", model), self.backend, time.time() - t0, True, cached,
                                 0.0 if cached else usd))
        return resp["text"], resp

    def ask_json(self, name: str, system: str, prompt: str, schema: type[T], attachments: list[str | Path] = (),
                 model: str = ARGON, thinking: str = "high",
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT, background: bool | None = None,
                 previous_id: str | None = None) -> tuple[T, dict]:
        rule = ("\n\nReturn exactly ONE JSON object and nothing else (no prose, no code fences). It must validate "
                f"against this JSON schema:\n{json.dumps(schema.model_json_schema())}")
        text, resp = self.ask_text(name, system, prompt + rule, attachments, model, thinking, json_mode=True,
                                   previous_id=previous_id, max_output_tokens=max_output_tokens, background=background)
        try:
            return schema.model_validate(parse_json(text)), resp
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            err = str(e)[:4000]
        fix = f"Your previous answer failed validation:\n{err}\n\nReturn the corrected single JSON object only."
        if resp.get("interaction_id") and self.backend == "gemini":  # multi-turn: no need to resend attachments
            # the chained turn still carries the images, so it must stay synchronous like the original call
            text2, resp2 = self.ask_text(f"{name}__fix", system, fix, (), model, thinking, json_mode=True,
                                         previous_id=resp["interaction_id"], max_output_tokens=max_output_tokens,
                                         background=False if attachments else background)
        else:
            text2, resp2 = self.ask_text(f"{name}__fix", system, f"{prompt}{rule}\n\nPrevious answer:\n{text[:20000]}"
                                         f"\n\n{fix}", attachments, model, thinking, json_mode=True,
                                         max_output_tokens=max_output_tokens)
        return schema.model_validate(parse_json(text2)), resp2


def call_summary() -> dict:
    return {"calls": len(CALLS), "ok": sum(c.ok for c in CALLS), "cached": sum(c.cached for c in CALLS),
            "seconds": round(sum(c.seconds for c in CALLS), 1), "usd_est": round(sum(c.usd for c in CALLS), 4),
            "by_model": {m: sum(c.model == m for c in CALLS) for m in {c.model for c in CALLS}}}


def run_parallel(jobs: list[Callable[[], object]], max_workers: int = 8) -> list[object]:
    """Run independent model calls concurrently (e.g. one Argon critic per object or per view).
    Returns results in order; a failed job yields its exception instead of raising."""
    def wrap(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return e

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(wrap, jobs))


def ask_with_tools(argon: Argon, name: str, system: str, prompt: str, schema: type[T], tools: dict,
                   attachments: list[str | Path] = (), model: str = ARGON, thinking: str = "high",
                   max_rounds: int = 6, max_output_tokens: int = DEFAULT_MAX_OUTPUT) -> tuple[T, dict]:
    """Function-calling loop on the Interactions API: tools = {name: (declaration, fn(args) -> result)}.
    The model calls tools (status requires_action), we answer with function_result steps on the same interaction
    chain, until it returns the final JSON. Offline backends fall back to ask_json without tools."""
    if argon.backend != "gemini":
        return argon.ask_json(name, system, prompt, schema, attachments, model, thinking, max_output_tokens)
    attachments = [Path(a) for a in attachments]
    rule = ("\n\nWhen you are done using tools, return exactly ONE JSON object and nothing else. It must validate "
            f"against this JSON schema:\n{json.dumps(schema.model_json_schema())}")
    b = argon._bundle(name, system, prompt + rule, attachments)
    key = argon._cache_key(model, system, prompt + rule + json.dumps(sorted(tools)), attachments, thinking, True,
                           None, max_output_tokens)
    cpath = STATE / "cache" / f"{key}.json"
    if argon.use_cache and cpath.exists():
        resp = json.loads(cpath.read_text())
    else:
        decls = [d for d, _ in tools.values()]
        gen = {"thinking_level": thinking, "max_output_tokens": max_output_tokens}
        check_budget("model", name)
        t0 = time.time()
        it = argon._create(name, model, dict(model=model, input=argon._parts(prompt + rule, attachments),
                                              system_instruction=system, tools=decls, generation_config=gen,
                                              timeout=TIMEOUT_S))
        record_spend("model", name, model, _usage_dict(it), True, it.id)
        log = []
        for _ in range(max_rounds):
            calls = [s for s in (it.steps or []) if getattr(s, "type", None) == "function_call"]
            if it.status != "requires_action" or not calls:
                break
            results = []
            for c in calls:
                try:
                    out, err = tools[c.name][1](dict(c.arguments or {})), False
                except Exception as e:  # noqa: BLE001 - the model sees the error and can adapt
                    out, err = f"error: {e}", True
                log.append({"tool": c.name, "args": dict(c.arguments or {}),
                            "result": out if isinstance(out, str) else [p for p in out if p.get("type") == "text"]})
                results.append({"type": "function_result", "call_id": c.id, "name": c.name, "result": out,
                                "is_error": err})
            check_budget("model", f"{name}__tools")
            it = argon._create(name, model, dict(model=model, input=results, previous_interaction_id=it.id,
                                                  system_instruction=system, tools=decls, generation_config=gen,
                                                  timeout=TIMEOUT_S))
            record_spend("model", f"{name}__tools", model, _usage_dict(it), True, it.id)
        resp = {"text": it.output_text or "", "interaction_id": it.id, "model": model, "tool_calls": log,
                "seconds": round(time.time() - t0, 1)}
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(resp, indent=1))
        with _LOCK:
            CALLS.append(CallLog(name, model, argon.backend, time.time() - t0, True))
    (b / "response.json").write_text(json.dumps(resp, indent=1))
    try:
        return schema.model_validate(parse_json(resp["text"])), resp
    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        fix = f"Your answer failed validation:\n{str(e)[:3000]}\n\nReturn the corrected single JSON object only."
        return argon.ask_json(f"{name}__fix", system, fix, schema, (), model, thinking, max_output_tokens,
                              background=False if attachments else None,   # the chain still holds the media
                              previous_id=resp.get("interaction_id"))
