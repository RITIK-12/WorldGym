"""The critic proposes θ patches; the harness keeps a patch only if the ladder score improves.

Claude reads the ladder table, fact-only events, the current θ, what was already tried (with the score change and
whether it was kept) and a contact sheet, and returns a ConfigPatch with its hypothesis. It never sees the referee's
GT numbers. A playbook learned across clips (outer loop) is appended to its system prompt.
Two ways to run it: ClaudeCritic through the API, or `harness step`, where the coding agent (Claude) reads the same
evidence printed by the harness and writes the patch itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from humanoid import llm
from humanoid.config import ConfigPatch, HarnessConfig

SYSTEM = """You tune a real-to-sim pipeline that turns a monocular sports clip into a reference motion for a Unitree
G1 humanoid (29 joints, 1.3 m, ~35 kg). The reference must be something an RL tracking policy can learn on the real
robot, so it is judged by a GT-free ladder, cheapest rung first:
L0 perception sanity (is the 3D human input clean and well grounded), L1 kinematic feasibility (can the G1 body make
these poses at these speeds without skating, penetrating or colliding with itself), L2 dynamic feasibility (can foot
contacts supply the forces the motion needs: the ghost force/torque is what nobody can supply; joint torques vs limits).
The ladder score counts passed rungs plus the mean goodness of the first failing rung (0..3).

Pipeline knobs you can change (θ). Perception itself is fixed and cached.
{fields}

Facts are measurements, not diagnoses: form your own hypothesis. Change one to three fields, aimed at the first
failing rung, without wrecking rungs that already pass (a patch is kept only if the total score rises). Do not repeat
a patch that was already tried. Keep hypotheses short and specific: which metric should move and why.
In the contact sheet, cyan is the perceived human and orange the G1 projected into the clip; the two right columns
render the G1 from the side and the front."""


def field_doc() -> str:
    lines = []
    for name, f in HarnessConfig.model_fields.items():
        meta = {k: getattr(m, k) for m in f.metadata for k in ("ge", "le") if hasattr(m, k)}
        rng = f" [{meta.get('ge')}..{meta.get('le')}]" if meta else ""
        ann = getattr(f.annotation, "__args__", None)
        kind = f" one of {list(ann)}" if ann and all(isinstance(a, str) for a in ann) else ""
        lines.append(f"- {name}{rng}{kind} default {f.default}: {f.description or ''}".rstrip(": "))
    return "\n".join(lines)


class ClaudeCritic:
    name = "claude"

    def __init__(self, playbook: str = ""):
        self.playbook = playbook

    def propose(self, clip: str, it: int, cfg: HarnessConfig, ladder_table: str, facts: list[str],
                history: list[dict], sheet: Path | None, bundle_dir: Path) -> ConfigPatch:
        system = SYSTEM.format(fields=field_doc())
        if self.playbook:
            system += "\n\nPlaybook learned on earlier clips (validated):\n" + self.playbook
        tried = "\n".join(f"- iter {h['iter']}: {json.dumps(h['diff'])} -> score {h['score_before']:.3f} -> "
                          f"{h['score_after']:.3f} ({'kept' if h['kept'] else 'reverted'}); hypothesis: {h['hypothesis']}"
                          for h in history) or "(nothing yet)"
        prompt = (f"Clip {clip}, iteration {it}.\n\nCurrent θ:\n{cfg.model_dump_json(indent=1)}\n\n{ladder_table}\n\n"
                  f"Facts:\n" + "\n".join(f"- {f}" for f in facts) + f"\n\nAlready tried on this clip:\n{tried}\n\n"
                  "Return the patch.")
        return llm.ask(f"critic_{clip}_{it}", system, prompt, ConfigPatch, images=[sheet] if sheet else [],
                       bundle_dir=bundle_dir, effort="high")
