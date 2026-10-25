"""500 Hz state + contact recording -> fact-only critical events + metrics.

Events are FACTS (what touched what, when, how hard, where). Never write causal diagnoses here; that is the
critic's job. With several arms every robot contact is attributed to the arm that made it and judged against that
arm's current phase; `phase`/`tcp` arrays follow the primary arm (the first one doing something other than waiting
or holding) and per-arm arrays are added.
"""
from __future__ import annotations

from collections import defaultdict

import mujoco
import numpy as np

from core.locked_evaluator import body_aabb
from core.mjcf_compiler import SceneIndex
from core.policy import PolicyExecutor

MIN_FORCE_N = 0.5
COALESCE_GAP_S = 0.04
FPS = 30
CARRY = ("lift_clear", "transit_arc", "lower_place", "wipe_approach", "wipe_lower", "wipe", "wipe_lift")
WIPING = ("wipe_lower", "wipe", "wipe_lift")
PASSIVE = ("idle", "hold", "wait", "pending")
UNWANTED = {"ILLEGAL_OBSTACLE_COLLISION", "ROBOT_CONTAINER_CONTACT", "ROBOT_TABLE_COLLISION", "OBJECT_ON_FLOOR",
            "CONTAINER_RIM_STRIKE", "CONTAINER_WALL_CONTACT_WHILE_HELD", "ROBOT_DISTURBED_OTHER_OBJECT",
            "HAND_BODY_OBJECT_CONTACT", "OBJECT_OBJECT_COLLISION", "ARM_ARM_COLLISION", "ROBOT_TARGET_CONTACT"}


class Telemetry:
    def __init__(self, ix: SceneIndex, d: mujoco.MjData, ctrls: dict, ex: PolicyExecutor):
        self.ix, self.m, self.d, self.ctrls, self.ex = ix, ix.m, d, ctrls, ex
        self.task = ix.task
        self.arms = list(ctrls)
        self.f6 = np.zeros(6)
        self.t: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.tcp: list[np.ndarray] = []
        self.width: list[float] = []
        self.lag: list[float] = []
        self.phase: list[str] = []
        self.phase_obj: list[str] = []
        self.arm_phase: dict[str, list[str]] = {a: [] for a in self.arms}
        self.arm_tcp: dict[str, list[np.ndarray]] = {a: [] for a in self.arms}
        self.arm_width: dict[str, list[float]] = {a: [] for a in self.arms}
        self.max_obj_speed: list[float] = []
        self.rows: dict[tuple, list[dict]] = defaultdict(list)   # (type, a, b) -> per-step rows
        self.events: list[dict] = []
        self.finger_contact_t: dict[str, float] = {}             # object -> last finger contact time
        self._grasp_ref: dict[str, tuple] = {}                   # arm -> (object, object - tcp offset at grasp)
        self._lag_event: dict[str, dict] = {}
        self.grasp_margin_min = np.inf
        self.carry_slip_max = 0.0
        self.lost_flagged: set = set()
        self.slip_flagged: set = set()
        self.prev_phase: dict[str, tuple] = {}
        self.wipe_points: dict[str, list[list[float]]] = defaultdict(list)   # "wiper|target" -> contact points
        self.free = [o.id for o in self.task.objects if not o.is_fixed]
        self.obstacles = {o.id for o in self.task.objects if o.is_obstacle}
        self.containers = {o.id for o in self.task.objects if o.shape == "container"}
        self.rim: dict[str, float] = {}
        self.geom_arm: dict[int, str] = {}
        for arm in ix.adapter.arms:
            bodies = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in arm.robot_bodies}
            for g in ix.robot_geoms:
                if self.m.geom_bodyid[g] in bodies:
                    self.geom_arm[g] = arm.name

    # ------------------------------------------------------------------------------------------------------------------
    def _owner(self, g: int) -> tuple[str, str]:
        ix = self.ix
        if g in ix.robot_geoms:
            b = self.m.geom_bodyid[g]
            return ("finger" if g in ix.finger_geoms else "robot"), mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
        if g in ix.geom_owner:
            return "object", ix.geom_owner[g]
        if g == ix.table:
            return "table", "table"
        if g == ix.floor:
            return "floor", "floor"
        return "other", ix.geom_name(g)

    def _rim_z(self, cid: str) -> float:
        if cid not in self.rim:
            self.rim[cid] = float(body_aabb(self.m, self.d, self.ix.obj_body[cid])[1][2])
        return self.rim[cid]

    def _held_bottom(self, oid: str | None) -> float | None:
        if not oid:
            return None
        return float(body_aabb(self.m, self.d, self.ix.obj_body[oid])[0][2])

    def _obstacle_top(self) -> float | None:
        if not self.obstacles:
            return None
        return float(max(body_aabb(self.m, self.d, self.ix.obj_body[o])[1][2] for o in self.obstacles))

    def _states(self) -> dict[str, dict]:
        out = {}
        for a, ctrl in self.ctrls.items():
            phase, obj = self.ex.phase(a)
            cur = self.ex.current.get(a)
            p = cur[0].p if cur else None
            tcp, _ = ctrl.tcp_pose()
            cmd, _ = ctrl.cmd_pose()
            out[a] = {"phase": phase, "obj": obj, "target": p.target_id if p else None, "kind": p.kind if p else None,
                      "tcp": tcp, "lag": float(np.linalg.norm(cmd - tcp)), "ctrl": ctrl}
        return out

    # ------------------------------------------------------------------------------------------------------------------
    def record(self) -> None:
        m, d, ix = self.m, self.d, self.ix
        t = float(d.time)
        st = self._states()
        primary = next((a for a in self.arms if st[a]["phase"] not in PASSIVE), self.arms[0])
        ps = st[primary]
        self.t.append(t)
        self.qpos.append(d.qpos.astype(np.float32).copy())
        self.tcp.append(ps["tcp"])
        self.width.append(ps["ctrl"].gripper_width())
        self.lag.append(max(s["lag"] for s in st.values()))
        self.phase.append(ps["phase"])
        self.phase_obj.append(ps["obj"] or "")
        for a, s in st.items():
            self.arm_phase[a].append(s["phase"])
            self.arm_tcp[a].append(s["tcp"])
            self.arm_width[a].append(s["ctrl"].gripper_width())
        speeds = [np.linalg.norm(d.qvel[ix.obj_dadr[o]:ix.obj_dadr[o] + 3]) for o in self.free]
        self.max_obj_speed.append(float(max(speeds, default=0.0)))

        held = {s["obj"] for s in st.values() if s["obj"] and s["phase"] in CARRY}
        step_sum: dict[tuple, dict] = {}
        finger_on: dict[str, float] = defaultdict(float)
        for i in range(d.ncon):
            c = d.contact[i]
            k1, n1 = self._owner(c.geom1)
            k2, n2 = self._owner(c.geom2)
            if {k1, k2} <= {"object", "table"} and "table" in (k1, k2):
                continue  # object resting on the table is not interesting
            if "other" in (k1, k2):
                continue
            mujoco.mj_contactForce(m, d, i, self.f6)
            fn = abs(float(self.f6[0]))
            if fn < MIN_FORCE_N:
                continue
            for (ka, na), (kb, nb) in (((k1, n1), (k2, n2)), ((k2, n2), (k1, n1))):
                if ka == "finger" and kb == "object":
                    finger_on[nb] += fn
            etype, info = self._classify(k1, n1, c.geom1, k2, n2, c.geom2, st, held, float(c.pos[2]))
            if etype is None:
                continue
            if etype == "WIPE_CONTACT":
                self.wipe_points[info].append([round(float(v), 4) for v in c.pos])
                continue
            a, b = sorted([n1, n2])
            key = (etype, a, b)
            row = step_sum.setdefault(key, {"t": t, "fn": 0.0, "fmax": 0.0, "pos": None, "geoms": set(), "arm": info})
            row["fn"] += fn
            row["geoms"].update([ix.geom_name(c.geom1), ix.geom_name(c.geom2)])
            if fn > row["fmax"]:
                row["fmax"], row["pos"], row["normal"] = fn, c.pos.copy(), c.frame[:3].copy()
        if step_sum:
            ob_top = self._obstacle_top()
            for key, row in step_sum.items():
                s = st.get(row["arm"]) if row["arm"] in st else ps
                hb = s["obj"] if s["phase"] in CARRY else None
                row.update(phase=s["phase"], tcp=s["tcp"].copy(), held_bottom_z=self._held_bottom(hb),
                           obstacle_top_z=ob_top)
                self.rows[key].append(row)

        for oid in finger_on:
            self.finger_contact_t[oid] = t
        for a, s in st.items():
            self._phase_facts(t, a, s, finger_on)
            for ev in s["ctrl"].events:
                ev.update(frame=round(ev["t"] * FPS), phase=s["phase"], object=s["obj"], arm=a)
                self.events.append(ev)
            s["ctrl"].events.clear()

    def _classify(self, k1, n1, g1, k2, n2, g2, st, held, z) -> tuple[str | None, str | None]:
        kinds = {k1, k2}
        robotish = {"robot", "finger"}
        names = {n1, n2}
        if kinds <= robotish:
            a1, a2 = self.geom_arm.get(g1), self.geom_arm.get(g2)
            return ("ARM_ARM_COLLISION", a1) if a1 and a2 and a1 != a2 else (None, None)
        if kinds & robotish:
            rg, rk = (g1, k1) if k1 in robotish else (g2, k2)
            other_k, other_n = (k2, n2) if k1 in robotish else (k1, n1)
            arm = self.geom_arm.get(rg)
            s = st.get(arm, {"obj": None, "target": None, "kind": None})
            if other_k in ("table", "floor"):
                return "ROBOT_TABLE_COLLISION", arm
            if other_k == "object":
                if other_n in self.obstacles:
                    return "ILLEGAL_OBSTACLE_COLLISION", arm
                if other_n in self.containers:
                    return "ROBOT_CONTAINER_CONTACT", arm
                if other_n == s["obj"]:
                    return ("HAND_BODY_OBJECT_CONTACT" if rk == "robot" else None), arm
                if other_n == s["target"] and s["kind"] == "wipe":
                    return "ROBOT_TARGET_CONTACT", arm
                return "ROBOT_DISTURBED_OTHER_OBJECT", arm
            return None, None
        if kinds == {"object", "floor"}:
            return "OBJECT_ON_FLOOR", None
        if kinds == {"object"}:
            for a, s in st.items():
                if s["obj"] not in names:
                    continue
                other = (names - {s["obj"]}).pop() if len(names) > 1 else s["obj"]
                if s["kind"] == "wipe" and other == s["target"] and s["phase"] in WIPING:
                    return "WIPE_CONTACT", f"{s['obj']}|{other}"
                if other in self.obstacles:
                    return "ILLEGAL_OBSTACLE_COLLISION", a
                if other in self.containers:
                    if z > self._rim_z(other) - 0.012:
                        return "CONTAINER_RIM_STRIKE", a
                    return ("CONTAINER_WALL_CONTACT_WHILE_HELD", a) if s["obj"] in held else (None, None)
                if s["obj"] in held and any(o["obj"] == other and o["phase"] == "hold" for o in st.values()):
                    return None, None   # contact with the object another arm is holding still (e.g. a wipe target)
                if s["obj"] in held:
                    return "OBJECT_OBJECT_COLLISION", a
            return None, None
        return None, None

    def _phase_facts(self, t: float, arm: str, s: dict, finger_on: dict) -> None:
        phase, pobj, tcp, ctrl, lag = s["phase"], s["obj"], s["tcp"], s["ctrl"], s["lag"]
        prev = self.prev_phase.get(arm)
        frame = round(t * FPS)
        if prev and prev[0] == "close" and phase != "close" and prev[1]:
            last = self.finger_contact_t.get(prev[1], -1.0)
            if last < t - 0.05:
                self.events.append({"type": "GRASP_MISSED", "t": t, "frame": frame, "phase": "close", "arm": arm,
                                    "object": prev[1], "tcp_xyz": tcp.tolist(),
                                    "gripper_width_m": round(ctrl.gripper_width(), 4)})
            else:
                self._grasp_ref[arm] = (prev[1], self.d.xpos[self.ix.obj_body[prev[1]]] - tcp)
        ref = self._grasp_ref.get(arm)
        if phase in CARRY and pobj and ref and ref[0] == pobj:
            last = self.finger_contact_t.get(pobj, -1.0)
            if t - last > 0.05 and pobj not in self.lost_flagged:
                self.lost_flagged.add(pobj)
                self.events.append({"type": "OBJECT_LOST_FROM_GRASP", "t": t, "frame": frame, "phase": phase,
                                    "arm": arm, "object": pobj, "tcp_xyz": tcp.tolist(),
                                    "object_xyz": self.d.xpos[self.ix.obj_body[pobj]].tolist()})
            if pobj not in self.lost_flagged:
                slip = float(np.linalg.norm(self.d.xpos[self.ix.obj_body[pobj]] - tcp - ref[1]))
                self.carry_slip_max = max(self.carry_slip_max, slip)
                if slip > 0.01 and pobj not in self.slip_flagged:
                    self.slip_flagged.add(pobj)
                    self.events.append({"type": "GRASP_SLIP", "t": t, "frame": frame, "phase": phase, "arm": arm,
                                        "object": pobj, "slip_m": round(slip, 4), "tcp_xyz": tcp.tolist()})
                fn = finger_on.get(pobj, 0.0)
                o = self.task.obj(pobj)
                self.grasp_margin_min = min(self.grasp_margin_min, 1.5 * fn / (o.mass * 9.81))
        if prev and prev[0] == "release" and phase != "release" and prev[1]:
            p = self.d.xpos[self.ix.obj_body[prev[1]]]
            self.events.append({"type": "OBJECT_RELEASED", "t": t, "frame": frame, "phase": "release", "arm": arm,
                                "object": prev[1], "object_xyz": p.tolist(),
                                "bottom_z": self._held_bottom(prev[1]), "tcp_xyz": tcp.tolist()})
        if lag > 0.02:
            ev = self._lag_event.get(arm)
            if ev is None or t - ev["t1"] > COALESCE_GAP_S:
                ev = {"type": "IK_TRACKING_ERROR", "t": t, "t1": t, "frame": frame, "phase": phase, "arm": arm,
                      "object": pobj, "peak_err_m": lag, "tcp_xyz": tcp.tolist()}
                self._lag_event[arm] = ev
                self.events.append(ev)
            else:
                ev["t1"] = t
                ev["peak_err_m"] = max(ev["peak_err_m"], lag)
        self.prev_phase[arm] = (phase, pobj)

    # ------------------------------------------------------------------------------------------------------------------
    def coalesce(self) -> list[dict]:
        out = []
        for (etype, a, b), rows in self.rows.items():
            groups: list[list[dict]] = []
            for r in rows:
                if groups and r["t"] - groups[-1][-1]["t"] <= COALESCE_GAP_S:
                    groups[-1].append(r)
                else:
                    groups.append([r])
            for g in groups:
                pk = max(g, key=lambda r: r["fn"])
                ev = {"type": etype, "t": round(g[0]["t"], 3), "t1": round(g[-1]["t"], 3),
                      "frame": round(pk["t"] * FPS), "phase": pk["phase"], "pair": [a, b],
                      "geoms": sorted(set().union(*[r["geoms"] for r in g])),
                      "peak_force_n": round(pk["fn"], 2), "t_peak": round(pk["t"], 3),
                      "contact_xyz": [round(float(v), 4) for v in pk["pos"]],
                      "tcp_xyz": [round(float(v), 4) for v in pk["tcp"]],
                      "held_bottom_z": None if pk["held_bottom_z"] is None else round(pk["held_bottom_z"], 4),
                      "obstacle_top_z": None if pk["obstacle_top_z"] is None else round(pk["obstacle_top_z"], 4)}
                if len(self.arms) > 1 and pk.get("arm") in self.arms:
                    ev["arm"] = pk["arm"]
                out.append(ev)
        return out

    @staticmethod
    def _timeline(ts, phases, objs=None) -> list[dict]:
        timeline: list[dict] = []
        for k, (t, ph) in enumerate(zip(ts, phases)):
            ob = objs[k] if objs else ""
            if not timeline or timeline[-1]["phase"] != ph or timeline[-1]["object"] != (ob or None):
                timeline.append({"phase": ph, "object": ob or None, "t0": round(t, 3), "t1": round(t, 3)})
            else:
                timeline[-1]["t1"] = round(t, 3)
        return timeline

    def finish(self) -> dict:
        contact_events = self.coalesce()
        events = sorted(contact_events + [self._round(e) for e in self.events], key=lambda e: e["t"])
        unwanted = [e for e in events if e["type"] in UNWANTED]
        obst = [e["peak_force_n"] for e in events if e["type"] == "ILLEGAL_OBSTACLE_COLLISION"]
        lag = np.array(self.lag) if self.lag else np.zeros(1)
        metrics = {
            "unwanted_collision_count": len(unwanted),
            "peak_obstacle_force_n": round(max(obst, default=0.0), 2),
            "grasp_margin_min": None if not np.isfinite(self.grasp_margin_min) else round(self.grasp_margin_min, 2),
            "carry_slip_max_m": round(self.carry_slip_max, 4),
            "episode_time_sec": round(self.t[-1] if self.t else 0.0, 3),
            "tcp_lag_p50_m": round(float(np.percentile(lag, 50)), 5),
            "tcp_lag_p99_m": round(float(np.percentile(lag, 99)), 5),
        }
        if self.wipe_points:
            metrics["wipe_contact_steps"] = {k: len(v) for k, v in self.wipe_points.items()}
        out = {"events": events, "metrics": metrics, "phase_timeline": self._timeline(self.t, self.phase,
                                                                                      self.phase_obj)}
        if len(self.arms) > 1:
            out["phase_timeline_per_arm"] = {a: self._timeline(self.t, self.arm_phase[a]) for a in self.arms}
        return out

    def wipe_contacts(self, max_points: int = 4000) -> dict[str, list[list[float]]]:
        out = {}
        for k, pts in self.wipe_points.items():
            step = max(1, len(pts) // max_points)
            out[k] = pts[::step]
        return out

    @staticmethod
    def _round(e: dict) -> dict:
        out = {}
        for k, v in e.items():
            if isinstance(v, float):
                out[k] = round(v, 4)
            elif isinstance(v, list):
                out[k] = [round(x, 4) if isinstance(x, float) else x for x in v]
            else:
                out[k] = v
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        out = {"t": np.array(self.t, np.float32), "qpos": np.array(self.qpos, np.float32),
               "tcp": np.array(self.tcp, np.float32), "width": np.array(self.width, np.float32),
               "lag": np.array(self.lag, np.float32), "phase": np.array(self.phase),
               "phase_obj": np.array(self.phase_obj), "max_obj_speed": np.array(self.max_obj_speed, np.float32)}
        if len(self.arms) > 1:
            for a in self.arms:
                out[f"phase_{a}"] = np.array(self.arm_phase[a])
                out[f"tcp_{a}"] = np.array(self.arm_tcp[a], np.float32)
                out[f"width_{a}"] = np.array(self.arm_width[a], np.float32)
        return out
