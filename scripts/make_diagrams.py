"""Draw the README diagrams (media/architecture.svg, media/pipeline.svg).

    .venv/bin/python scripts/make_diagrams.py
"""
from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "media"
ORANGE, BLUE, GREEN, SLATE = "#E8710A", "#1A73E8", "#1E8E3E", "#5F6368"
INK, INK2, PANEL, LINE = "#202124", "#5F6368", "#F1F3F4", "#202124"
FONT = "Helvetica, Arial, sans-serif"
T = 46          # tile half size
R = 54          # hexagon radius


# ----------------------------------------------------------------------------------------------------------------------
# icons (white, centred on 0,0, about 50 px)
# ----------------------------------------------------------------------------------------------------------------------
def _w(d: str, sw: float = 3.2) -> str:
    return f'<path d="{d}" fill="none" stroke="#fff" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'


ICONS = {
    "video": _w("M-24,-14 h30 a4,4 0 0 1 4,4 v20 a4,4 0 0 1 -4,4 h-30 a4,4 0 0 1 -4,-4 v-20 a4,4 0 0 1 4,-4 z")
    + _w("M10,-5 L24,-13 V13 L10,5 Z"),
    "robot": _w("M-22,22 h24 M-10,22 V8 L-14,-8 L8,-18 L20,-6") + _w("M20,-6 l6,2 M20,-6 l2,6")
    + "".join(f'<circle cx="{x}" cy="{y}" r="3.6" fill="#fff"/>' for x, y in ((-10, 8), (-14, -8), (8, -18))),
    "cube": _w("M0,-22 L20,-11 V11 L0,22 L-20,11 V-11 Z M-20,-11 L0,0 L20,-11 M0,0 V22"),
    "check": _w("M-18,-22 h36 v44 h-36 z") + _w("M-10,-6 l6,6 l12,-12") + _w("M-10,12 h20", 2.6),
    "target": '<circle cx="0" cy="0" r="16" fill="none" stroke="#fff" stroke-width="3.2"/>'
    + '<circle cx="0" cy="0" r="4" fill="#fff"/>' + _w("M0,-26 V-18 M0,18 V26 M-26,0 H-18 M18,0 H26"),
    "motion": _w("M-24,14 C-14,14 -14,-12 -4,-12 S6,14 16,6") + _w("M16,6 l-1,-9 M16,6 l-9,1")
    + '<circle cx="-24" cy="14" r="4" fill="#fff"/>',
    "world": _w("M0,-20 L24,-8 L0,4 L-24,-8 Z") + _w("M-24,2 L0,14 L24,2", 2.6) + _w("M-24,12 L0,24 L24,12", 2.6),
    "shield": _w("M0,-24 L20,-16 V0 C20,12 10,20 0,24 C-10,20 -20,12 -20,0 V-16 Z") + _w("M-8,0 l5,6 l11,-12"),
    "screen": _w("M-26,-18 h52 v32 h-52 z M-8,22 h16 M0,14 v8") + _w("M-6,-8 l12,6 l-12,6 z", 2.4),
    "grid": _w("M-22,-22 h44 v44 h-44 z M-22,0 h44 M0,-22 v44", 2.6),
}


def neural() -> str:
    pts = [(14 * math.cos(a), 14 * math.sin(a)) for a in [math.pi / 6 + k * math.pi / 3 for k in range(6)]]
    lines = "".join(f'<line x1="0" y1="0" x2="{x:.1f}" y2="{y:.1f}" stroke="#fff" stroke-width="2"/>' for x, y in pts)
    ring = "".join(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#fff" '
                   f'stroke-width="2"/>' for a, b in zip(pts, pts[1:] + pts[:1]))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#fff"/>' for x, y in pts)
    return lines + ring + dots + '<circle cx="0" cy="0" r="5" fill="#fff"/>'


# ----------------------------------------------------------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------------------------------------------------------
def label(x: float, y: float, title: str, sub: str = "", size: int = 19) -> str:
    out = []
    for k, line in enumerate(title.split("\n")):
        out.append(f'<text x="{x}" y="{y + k * (size + 4)}" text-anchor="middle" font-size="{size}" '
                   f'font-weight="600" fill="{INK}">{line}</text>')
    y2 = y + len(title.split("\n")) * (size + 4)
    for k, line in enumerate(sub.split("\n") if sub else []):
        out.append(f'<text x="{x}" y="{y2 + k * 19}" text-anchor="middle" font-size="15" fill="{INK2}">{line}</text>')
    return "".join(out)


def side_label(x: float, y: float, title: str, sub: str = "") -> str:
    """Left-aligned label to the right of a shape, vertically centred on y."""
    lines = [(t, 19, "600", INK) for t in title.split("\n")] + [(t, 15, "400", INK2) for t in sub.split("\n") if sub]
    y0 = y - (len(lines) - 1) * 11 + 6
    return "".join(f'<text x="{x}" y="{y0 + k * 22:.0f}" font-size="{sz}" font-weight="{wt}" fill="{c}">{t}</text>'
                   for k, (t, sz, wt, c) in enumerate(lines))


def tile(x: float, y: float, colour: str, icon: str, title: str, sub: str = "", side: bool = False) -> str:
    text = side_label(x + T + 16, y, title, sub) if side else label(x, y + T + 28, title, sub)
    return (f'<rect x="{x - T}" y="{y - T}" width="{2 * T}" height="{2 * T}" rx="10" fill="{colour}"/>'
            f'<g transform="translate({x},{y})">{ICONS[icon]}</g>' + text)


def wide(x: float, y: float, w: float, colour: str, icon: str, title: str, sub: str) -> str:
    """A wide tile with its text inside (for the world, which many arrows attach to)."""
    return (f'<rect x="{x - w / 2}" y="{y - T}" width="{w}" height="{2 * T}" rx="10" fill="{colour}"/>'
            f'<g transform="translate({x - w / 2 + 48},{y})">{ICONS[icon]}</g>'
            f'<text x="{x - w / 2 + 92}" y="{y - 4}" font-size="21" font-weight="600" fill="#fff">{title}</text>'
            f'<text x="{x - w / 2 + 92}" y="{y + 20}" font-size="15" fill="#fff" opacity="0.9">{sub}</text>')


def hexagon(x: float, y: float, title: str, sub: str = "", stack: int = 1, colour: str = BLUE,
            side: bool = False) -> str:
    def poly(cx: float, cy: float, fill: str, op: float = 1.0) -> str:
        pts = " ".join(f"{cx + R * math.cos(math.pi / 6 + k * math.pi / 3):.1f},"
                       f"{cy + R * math.sin(math.pi / 6 + k * math.pi / 3):.1f}" for k in range(6))
        return f'<polygon points="{pts}" fill="{fill}" opacity="{op}"/>'
    back = "".join(poly(x + 9 * k, y - 9 * k, colour, 0.35 + 0.2 * k) for k in range(stack - 1, 0, -1))
    text = side_label(x + R + 18, y, title, sub) if side else label(x, y + R + 30, title, sub)
    return back + poly(x, y, colour) + f'<g transform="translate({x},{y})">{neural()}</g>' + text


def panel(x: float, y: float, w: float, h: float, title: str, bottom: bool = False) -> str:
    ty = y + h - 14 if bottom else y + 26
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PANEL}"/>'
            f'<text x="{x + 14}" y="{ty}" font-size="16" fill="{INK2}">{title}</text>')


def arrow(pts: list[tuple[float, float]], text: str = "", at: float = 0.5, dx: float = 0, dy: float = -10,
          r: float = 14, two_way: bool = False, dashed: bool = False) -> str:
    """Orthogonal polyline with rounded corners and an arrowhead; text sits on the segment at fraction `at`."""
    d = f"M{pts[0][0]},{pts[0][1]}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        a = min(r, math.dist((x0, y0), (x1, y1)) / 2, math.dist((x1, y1), (x2, y2)) / 2)
        ux, uy = (x1 - x0) / max(math.dist((x0, y0), (x1, y1)), 1e-9), (y1 - y0) / max(math.dist((x0, y0), (x1, y1)), 1e-9)
        vx, vy = (x2 - x1) / max(math.dist((x1, y1), (x2, y2)), 1e-9), (y2 - y1) / max(math.dist((x1, y1), (x2, y2)), 1e-9)
        d += f" L{x1 - ux * a:.1f},{y1 - uy * a:.1f} Q{x1},{y1} {x1 + vx * a:.1f},{y1 + vy * a:.1f}"
    d += f" L{pts[-1][0]},{pts[-1][1]}"
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    s = (f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="2"{dash} marker-end="url(#head)"'
         + (' marker-start="url(#tail)"' if two_way else "") + "/>")
    if text:
        segs = list(zip(pts, pts[1:]))
        total = sum(math.dist(a, b) for a, b in segs)
        goal, run = total * at, 0.0
        for a, b in segs:
            L = math.dist(a, b)
            if run + L >= goal:
                f = (goal - run) / max(L, 1e-9)
                tx, ty = a[0] + (b[0] - a[0]) * f + dx, a[1] + (b[1] - a[1]) * f + dy
                break
            run += L
        lines = text.split("\n")
        for k, line in enumerate(lines):
            yy = ty + (k - (len(lines) - 1) / 2) * 19
            w = 8.6 * len(line) + 10
            s += (f'<rect x="{tx - w / 2:.1f}" y="{yy - 14:.1f}" width="{w:.1f}" height="19" fill="#fff"/>'
                  f'<text x="{tx:.1f}" y="{yy:.1f}" text-anchor="middle" font-size="16" fill="{INK}">{line}</text>')
    return s


def svg(w: int, h: int, body: str) -> str:
    defs = (f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
            f'<marker id="tail" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="9" markerHeight="9" '
            f'orient="auto"><path d="M10,0 L0,5 L10,10 z" fill="{LINE}"/></marker>'
            f'<pattern id="hatch" patternUnits="userSpaceOnUse" width="9" height="9" patternTransform="rotate(45)">'
            f'<line x1="0" y1="0" x2="0" y2="9" stroke="#A5D8FF" stroke-width="2.5"/></pattern></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}"><rect width="{w}" height="{h}" fill="#fff"/>{defs}{body}</svg>')


def legend(x: float, y: float) -> str:
    items = [(BLUE, "Gemini agent", "hex"), (ORANGE, "input, output, asset library", "sq"),
             (GREEN, "runnable world", "sq"), (SLATE, "deterministic tool", "sq")]
    out = []
    for k, (c, t, kind) in enumerate(items):
        cx = x + k * 250
        if kind == "hex":
            pts = " ".join(f"{cx + 11 * math.cos(math.pi / 6 + j * math.pi / 3):.1f},"
                           f"{y + 11 * math.sin(math.pi / 6 + j * math.pi / 3):.1f}" for j in range(6))
            out.append(f'<polygon points="{pts}" fill="{c}"/>')
        else:
            out.append(f'<rect x="{cx - 10}" y="{y - 10}" width="20" height="20" rx="4" fill="{c}"/>')
        out.append(f'<text x="{cx + 20}" y="{y + 5}" font-size="15" fill="{INK2}">{t}</text>')
    return "".join(out)


# ----------------------------------------------------------------------------------------------------------------------
# architecture
# ----------------------------------------------------------------------------------------------------------------------
def architecture() -> str:
    W, H = 1860, 1050
    VID, VA, SW = (130, 480), (540, 480), (540, 820)
    CAND, SR, UI = (360, 130), (720, 130), (1270, 130)
    CAM, MOT = (900, 650), (900, 860)
    WX, WY, WW = 1270, 480, 330                     # world: wide tile
    CRIT, EVAL = (1600, 250), (1600, 720)
    hexface = R * math.cos(math.pi / 6)            # half width of a pointy-top hexagon
    b = [panel(330, 300, 1510, 680, "Orchestrator (Python)", bottom=True),
         tile(*CAND, ORANGE, "robot", "Rendered robot models", "13 MuJoCo Menagerie arms"),
         tile(*SR, ORANGE, "cube", "NVIDIA SimReady", "3.5k physics-ready assets"),
         tile(*UI, ORANGE, "screen", "WorldGym UI", "upload, progress, Run simulation"),
         tile(*VID, ORANGE, "video", "Robot video", "any robot, 1 or 2 arms\nfixed or handheld camera"),
         hexagon(*VA, "Video Architect", "Gemini 4 Argon\none call, function calling"),
         hexagon(*SW, "Labelling swarm", "16 x Gemini Robotics-ER 2", stack=3),
         tile(*CAM, SLATE, "target", "Camera solver", "robot outline +\ntable tracking", side=True),
         tile(*MOT, SLATE, "motion", "Motion extraction", "labels to policy parameters"),
         wide(WX, WY, WW, GREEN, "world", "MuJoCo world", "Gen 1, Gen 2, Gen 3"),
         hexagon(*CRIT, "Four-view Critic", "Gemini 4 Argon\n2 rounds with memory", side=True),
         tile(*EVAL, SLATE, "shield", "Locked evaluator", "success, contacts,\nend-pose error", side=True),
         # inputs to the architect
         arrow([(VID[0] + T + 4, VA[1]), (VA[0] - hexface - 4, VA[1])], "video + keyframes", dy=-14),
         arrow([(CAND[0] + 40, 262), (VA[0] - 30, VA[1] - R - 2)], "robot candidates", at=0.45, dx=-78, dy=0),
         arrow([(SR[0] - 40, 262), (VA[0] + 30, VA[1] - R - 2)], "asset search\n(tool call)", at=0.45, dx=84,
               dy=0, two_way=True),
         # architect -> world (Gen 1)
         arrow([(VA[0] + hexface + 4, VA[1]), (WX - WW / 2 - 4, WY)], "robot, objects, layout, assets,\n"
               "phases, motion of each arm", at=0.5, dy=-30),
         # swarm lane
         arrow([(VID[0], VID[1] + T + 84), (VID[0], SW[1]), (SW[0] - hexface - 4, SW[1])], "frames", at=0.8, dy=-12),
         arrow([(VA[0] + hexface + 2, VA[1] + 26), (CAM[0] - T - 4, CAM[1] - 10)], "robot model", at=0.55,
               dx=8, dy=-14),
         arrow([(SW[0] + hexface + 4, MOT[1]), (MOT[0] - T - 4, MOT[1])], "gripper points,\nobject boxes",
               dy=-26),
         arrow([(CAM[0], CAM[1] + T + 4), (CAM[0], MOT[1] - T - 4)], "camera\nper frame", dx=-62, dy=0),
         # tools -> world (Gen 2)
         arrow([(CAM[0] + T + 170, CAM[1]), (WX - 110, CAM[1]), (WX - 110, WY + T + 4)], "measured\npositions",
               at=0.42, dy=-24),
         arrow([(MOT[0] + T + 4, MOT[1]), (WX - 30, MOT[1]), (WX - 30, WY + T + 4)], "motion", at=0.4, dy=-12),
         # critique loop (Gen 3)
         arrow([(WX + WW / 2 + 4, WY - 20), (CRIT[0] - 40, WY - 20), (CRIT[0] - 40, CRIT[1] + R + 2)],
               "original clip +\n4-view video", at=0.33, dy=-26),
         arrow([(CRIT[0] + 30, CRIT[1] + R + 2), (CRIT[0] + 30, EVAL[1] - T - 4)], "scene edits +\npolicy patch",
               dx=84, dy=10),
         arrow([(EVAL[0] - T - 4, EVAL[1]), (WX + 110, EVAL[1]), (WX + 110, WY + T + 4)],
               "kept only if the\nscore improves", at=0.35, dy=-26),
         # world -> UI
         arrow([(WX, WY - T - 4), (WX, UI[1] + T + 74)], "every generation,\nlive physics", dx=96, dy=0),
         legend(520, H - 26)]
    return svg(W, H, "".join(b))


def box(x: float, y: float, w: float, h: float, fill: str, stroke: str, title: str, sub: str = "",
        size: int = 20, dashed: bool = False, hatch: bool = False) -> str:
    """Rounded box with centred text; (x, y) is the centre."""
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    out = f'<rect x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" ' \
          f'stroke-width="2.2"{dash}/>'
    if hatch:
        out += f'<rect x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" rx="16" fill="url(#hatch)" opacity="0.6"/>'
    lines = [(t, size, "600", INK) for t in title.split("\n")] + [(t, 15, "400", "#343A40") for t in sub.split("\n")
                                                                 if sub]
    y0 = y - (len(lines) - 1) * 11 + 6
    out += "".join(f'<text x="{x}" y="{y0 + k * 22:.0f}" text-anchor="middle" font-size="{sz}" font-weight="{wt}" '
                   f'fill="{c}">{t}</text>' for k, (t, sz, wt, c) in enumerate(lines))
    return out


def link(a: tuple[float, float], b: tuple[float, float], text: str = "", dx: float = 0, dy: float = -12) -> str:
    """Two-way straight connector, as in a system diagram."""
    return arrow([a, b], text, at=0.5, dx=dx, dy=dy, two_way=True)


def agents() -> str:
    """Agent hierarchy: one orchestrator, three phase agents, and what each agent fans out to."""
    W, H = 1600, 760
    BLUE_F, BLUE_S = "#A5D8FF", "#1971C2"
    PURPLE_F, PURPLE_S = "#E5DBFF", "#7048E8"
    ORANGE_F, ORANGE_S = "#FFD8A8", "#E8590C"
    OR = (800, 110)
    AY, AH, AW = 370, 136, 400
    VA, SW, FC = (300, AY), (800, AY), (1300, AY)
    CY = 610                                                    # children row
    b = [box(*OR, 600, 120, BLUE_F, BLUE_S, "Orchestrator", "runs the three phases, checks each agent's typed "
             "output,\nkeeps a fix only if the score improves", size=24),
         box(*VA, AW, AH, PURPLE_F, PURPLE_S, "Phase 1: Video Architect", "Gemini 4 Argon, one call\nrobot, objects, "
             "assets, layout,\nwhat each arm does", size=21),
         box(*SW, AW, AH, PURPLE_F, PURPLE_S, "Phase 2: Labelling swarm", "Gemini Robotics-ER 2\ngripper points and "
             "object boxes\nfor every frame, in parallel", size=21),
         box(*FC, AW, AH, PURPLE_F, PURPLE_S, "Phase 3: Four-view Critic", "Gemini 4 Argon\nwatches the clip next to "
             "4 views\nof the simulation, patches it", size=21)]
    # orchestrator -> agents (org-chart connector)
    bus_y = 240
    b.append(f'<path d="M{OR[0]},{OR[1] + 62} V{bus_y} M{VA[0]},{bus_y} H{FC[0]}" fill="none" stroke="{LINE}" '
             f'stroke-width="2"/>')
    for x in (VA[0], SW[0], FC[0]):
        b.append(arrow([(x, bus_y), (x, AY - AH / 2 - 4)]))
    # phase 1: tool
    b += [box(VA[0], CY, 280, 96, ORANGE_F, ORANGE_S, "NVIDIA SimReady search", "tool the architect calls\n"
              "to pick real 3D assets", size=18),
          arrow([(VA[0], AY + AH / 2 + 4), (VA[0], CY - 52)], "tool call", dx=48, dy=0, two_way=True)]
    # phase 2: sixteen workers
    cols, mw, mh, gap = 8, 46, 40, 10
    gx0 = SW[0] - (cols * mw + (cols - 1) * gap) / 2
    rows_y = (CY - 30, CY + 20)
    b.append(f'<path d="M{SW[0]},{AY + AH / 2 + 2} V{CY - 70} M{gx0 + mw / 2},{CY - 70} '
             f'H{gx0 + (cols - 1) * (mw + gap) + mw / 2}" fill="none" stroke="{LINE}" stroke-width="2"/>')
    for k in range(cols):
        x = gx0 + k * (mw + gap)
        b.append(arrow([(x + mw / 2, CY - 70), (x + mw / 2, rows_y[0] - mh / 2 - 3)]))
    for r, yy in enumerate(rows_y):
        for k in range(cols):
            x = gx0 + k * (mw + gap)
            b.append(f'<rect x="{x}" y="{yy - mh / 2}" width="{mw}" height="{mh}" rx="8" fill="{PURPLE_F}" '
                     f'stroke="{PURPLE_S}" stroke-width="1.6"/>'
                     f'<text x="{x + mw / 2}" y="{yy + 5}" text-anchor="middle" font-size="13" fill="{INK}">'
                     f'ER {r * cols + k + 1}</text>')
    b.append(f'<text x="{SW[0]}" y="{CY + 72}" text-anchor="middle" font-size="15" fill="#343A40">16 Robotics-ER '
             f'agents, 8 frames each, all at once</text>')
    # phase 3: two rounds
    R1, R2 = (FC[0] - 105, CY), (FC[0] + 105, CY)
    b += [box(*R1, 160, 96, PURPLE_F, PURPLE_S, "Round 1", "scene edits +\npolicy patch", size=18),
          box(*R2, 160, 96, PURPLE_F, PURPLE_S, "Round 2", "sees why round 1\nwas not kept", size=18),
          f'<path d="M{FC[0]},{AY + AH / 2 + 2} V{CY - 76} M{R1[0]},{CY - 76} H{R2[0]}" fill="none" '
          f'stroke="{LINE}" stroke-width="2"/>',
          arrow([(R1[0], CY - 76), (R1[0], CY - 52)]), arrow([(R2[0], CY - 76), (R2[0], CY - 52)]),
          arrow([(R1[0] + 82, CY), (R2[0] - 84, CY)])]
    # phase order note
    b.append(f'<text x="{W / 2}" y="{H - 28}" text-anchor="middle" font-size="16" fill="{INK2}">Phase 2 starts as '
             f'soon as Phase 1 returns. Every result is checked against its contract and scored before the next '
             f'phase uses it.</text>')
    return svg(W, H, "".join(b))


# ----------------------------------------------------------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------------------------------------------------------
def pipeline() -> str:
    W, H = 1900, 800
    Y, Y2 = 330, 620
    xs = [90, 290, 500, 710, 920, 1120, 1330, 1550, 1790]
    EV = (1660, Y2)
    b = [panel(830, 40, 1040, 120, ""),
         tile(890, 100, ORANGE, "screen", "", ""),
         f'<text x="{890 + T + 18}" y="94" font-size="19" font-weight="600" fill="{INK}">WorldGym UI</text>'
         f'<text x="{890 + T + 18}" y="118" font-size="15" fill="{INK2}">shows each generation as it lands; '
         f'Run simulation plays it live in MuJoCo</text>',
         tile(xs[0], Y, ORANGE, "video", "Input video", "any robot"),
         tile(xs[1], Y, SLATE, "check", "Clip QA", "fixed or handheld\ncamera mode"),
         hexagon(xs[2], Y, "Video Architect", "Gemini 4 Argon, 1 call\nrobot, objects, assets,\nlayout, "
                                              "arm motions"),
         tile(xs[3], Y, SLATE, "robot", "Robot onboarding", "Menagerie model\n6-check self-test"),
         tile(xs[4], Y, GREEN, "world", "Gen 1", "first world, from the\narchitect alone"),
         tile(xs[5], Y, SLATE, "target", "Camera solve", "robot outline +\ntable tracking"),
         tile(xs[6], Y, GREEN, "world", "Gen 2", "measured positions,\nswarm motion"),
         hexagon(xs[7], Y, "Four-view Critic", "Argon: original clip +\n4 views of the Gen 2 run"),
         tile(xs[8], Y, GREEN, "world", "Gen 3", "refined world"),
         hexagon(1010, Y2, "Labelling swarm", "16 x Robotics-ER 2, in parallel", stack=3),
         tile(*EV, SLATE, "shield", "Locked evaluator", "success, contacts, end error"),
         tile(xs[2], 100, ORANGE, "cube", "", ""),
         f'<text x="{xs[2] + T + 18}" y="96" font-size="19" font-weight="600" fill="{INK}">NVIDIA SimReady</text>'
         f'<text x="{xs[2] + T + 18}" y="118" font-size="15" fill="{INK2}">assets by tool call</text>']
    for a, c in zip(xs, xs[1:]):
        half_a = R if a in (xs[2], xs[7]) else T
        half_c = R if c in (xs[2], xs[7]) else T
        b.append(arrow([(a + half_a + 4, Y), (c - half_c - 4, Y)]))
    b += [arrow([(xs[2], 100 + T + 4), (xs[2], Y - R - 4)], two_way=True),
          arrow([(xs[2], Y + R + 102), (xs[2], Y2), (1010 - R - 4, Y2)], "starts right after the architect",
                at=0.62, dy=-12),
          arrow([(1010 + R + 4, Y2), (xs[6], Y2), (xs[6], Y + T + 80)], "gripper points, boxes", at=0.36, dy=-12),
          arrow([(xs[7], Y + R + 82), (xs[7], Y2), (EV[0] - T - 4, Y2)], "patch, up to\n2 rounds", at=0.42,
                dx=-62, dy=0),
          arrow([(EV[0] + T + 4, Y2), (xs[8], Y2), (xs[8], Y + T + 60)], "kept if better", at=0.42, dy=-12)]
    for x in (xs[4], xs[6], xs[8]):
        b.append(arrow([(x, Y - T - 4), (x, 164)], dashed=True))
    b.append(legend(560, H - 22))
    return svg(W, H, "".join(b))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "architecture.svg").write_text(agents())
    (OUT / "pipeline.svg").write_text(pipeline())
    print("wrote", OUT / "architecture.svg", OUT / "pipeline.svg")
