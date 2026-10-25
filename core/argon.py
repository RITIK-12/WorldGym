"""One adapter for every model call, with three backends selected by WORLDGYM_LLM:

- gemini : live Interactions API (background=True + polling, so "high" thinking never hits an HTTP timeout)
- replay : read <replay_dir>/<name>.json (stage-outage safety net; record once the demo video works)
- manual : write the request bundle and stop, so a human/other agent can answer it

Every request writes a bundle runs/<id>/argon/<name>/{system.md, prompt.md, attachments/, response.json}.
JSON hygiene: ask for ONE object, strip code fences, validate with pydantic, re-ask once with the validation error.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

ARGON = "gemini-4-argon"
FLASH = "gemini-3.8-flash"
LITE = "gemini-3.5-flash-lite"
REPLAY_DIR = Path(os.environ.get("WORLDGYM_REPLAY_DIR", REPO / "replay_cache"))
INLINE_VIDEO_MAX = 18 * 1024 * 1024
POLL_S = 3.0
TIMEOUT_S = 20 * 60

T = TypeVar("T", bound=BaseModel)


class ManualPending(Exception):
    """manual backend: the bundle is written; put response.json in it and re-run."""


class ArgonError(Exception):
    pass


@dataclass
class CallLog:
    name: str
    model: str
    backend: str
    seconds: float
    ok: bool


CALLS: list[CallLog] = []   # process-wide call log for the scorecard ("model calls")


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


class Argon:
    def __init__(self, bundle_root: str | Path, backend: str | None = None, replay_dir: str | Path | None = None):
        self.root = Path(bundle_root)
        self.backend = backend or os.environ.get("WORLDGYM_LLM", "gemini")
        self.replay_dir = Path(replay_dir) if replay_dir else REPLAY_DIR
        self._client = None

    # ------------------------------------------------------------------------------------------------------------------
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

    def _gemini(self, model: str, system: str, parts: list[dict], thinking: str, json_mode: bool,
                previous_id: str | None) -> tuple[str, str, dict]:
        kw = dict(model=model, input=parts, system_instruction=system, background=True,
                  generation_config={"thinking_level": thinking, "max_output_tokens": 65536})
        if previous_id:
            kw["previous_interaction_id"] = previous_id
        if json_mode:
            kw["response_mime_type"] = "application/json"
        try:
            it = self.client.interactions.create(**kw)
        except Exception as e:  # noqa: BLE001 - some models reject JSON mode without a schema; retry plain
            if json_mode and "response" in str(e).lower():
                kw.pop("response_mime_type")
                it = self.client.interactions.create(**kw)
            else:
                raise
        t0 = time.time()
        while it.status in ("in_progress", "queued"):
            if time.time() - t0 > TIMEOUT_S:
                self.client.interactions.cancel(it.id)
                raise ArgonError(f"interaction {it.id} timed out")
            time.sleep(POLL_S)
            it = self.client.interactions.get(it.id)
        if it.status != "completed":
            raise ArgonError(f"interaction {it.id} ended {it.status}: {getattr(it, 'errors', None)}")
        usage = it.usage.model_dump() if getattr(it, "usage", None) is not None and hasattr(it.usage, "model_dump") else {}
        return it.output_text or "", it.id, usage

    # ------------------------------------------------------------------------------------------------------------------
    def ask_text(self, name: str, system: str, prompt: str, attachments: list[str | Path] = (), model: str = ARGON,
                 thinking: str = "high", json_mode: bool = False, previous_id: str | None = None) -> tuple[str, dict]:
        attachments = [Path(a) for a in attachments]
        b = self._bundle(name, system, prompt, attachments)
        resp_path = b / "response.json"
        t0 = time.time()
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
            try:
                text, iid, usage = self._gemini(model, system, self._parts(prompt, attachments), thinking,
                                                json_mode, previous_id)
            except Exception:
                CALLS.append(CallLog(name, model, self.backend, time.time() - t0, False))
                raise
            resp = {"text": text, "interaction_id": iid, "model": model, "usage": usage,
                    "seconds": round(time.time() - t0, 1)}
            REPLAY_DIR.mkdir(parents=True, exist_ok=True)
            (REPLAY_DIR / f"{name}.json").write_text(json.dumps(resp, indent=1))
        else:
            raise ArgonError(f"unknown backend {self.backend!r}")
        resp.setdefault("backend", self.backend)
        resp_path.write_text(json.dumps(resp, indent=1))
        CALLS.append(CallLog(name, resp.get("model", model), self.backend, time.time() - t0, True))
        return resp["text"], resp

    def ask_json(self, name: str, system: str, prompt: str, schema: type[T], attachments: list[str | Path] = (),
                 model: str = ARGON, thinking: str = "high") -> tuple[T, dict]:
        rule = ("\n\nReturn exactly ONE JSON object and nothing else (no prose, no code fences). It must validate "
                f"against this JSON schema:\n{json.dumps(schema.model_json_schema())}")
        text, resp = self.ask_text(name, system, prompt + rule, attachments, model, thinking, json_mode=True)
        try:
            return schema.model_validate(parse_json(text)), resp
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            err = str(e)[:4000]
        fix = f"Your previous answer failed validation:\n{err}\n\nReturn the corrected single JSON object only."
        if resp.get("interaction_id") and self.backend == "gemini":  # multi-turn: no need to resend attachments
            text2, resp2 = self.ask_text(f"{name}__fix", system, fix, (), model, thinking, json_mode=True,
                                         previous_id=resp["interaction_id"])
        else:
            text2, resp2 = self.ask_text(f"{name}__fix", system, f"{prompt}{rule}\n\nPrevious answer:\n{text[:20000]}"
                                         f"\n\n{fix}", attachments, model, thinking, json_mode=True)
        return schema.model_validate(parse_json(text2)), resp2


def call_summary() -> dict:
    return {"calls": len(CALLS), "ok": sum(c.ok for c in CALLS), "seconds": round(sum(c.seconds for c in CALLS), 1),
            "by_model": {m: sum(c.model == m for c in CALLS) for m in {c.model for c in CALLS}}}
