"""Studio panel content: media loading, big-number / timeline HTML and the two plotly charts."""
from __future__ import annotations

import html
import re
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go

from app.story import Chip, fmt_s, timeline
from core.schemas import RunManifest

MEDIA_W = 960
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#8b8a84"
SERIES = ("#3987e5", "#d95926")


def load_image(path: str | Path, max_w: int = MEDIA_W) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, round(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def big_number(number: str, label: str) -> str:
    size = 56 if len(number) <= 8 else 34
    return (f'<div style="padding:2px 0 8px"><div style="font-size:{size}px;font-weight:700;line-height:1.05;color:{INK};'
            f'font-variant-numeric:tabular-nums">{html.escape(number)}</div>'
            f'<div style="font-size:13px;color:{INK2}">{html.escape(label)}</div></div>')


def _pill(c: Chip, text: str) -> str:
    ink = INK if c.status == "pending" else "#111"
    return (f'<span style="display:inline-block;margin:2px 3px 2px 0;padding:1px 8px;border-radius:9px;'
            f'background:{c.colour};color:{ink};font-size:11px;white-space:nowrap">{html.escape(text)}</span>')


def strip_html(chips: list[Chip]) -> str:
    return "<div style='line-height:1.9'>" + "".join(
        _pill(c, c.name + (f" · {fmt_s(c.seconds)}" if c.status == "running" else "")) for c in chips) + "</div>"


def timeline_html(m: RunManifest, chips: list[Chip]) -> str:
    head = (f"<div style='color:{INK};font-weight:600'>{html.escape(m.run_id)}</div>"
            f"<div style='color:{MUTED};font-size:12px;margin-bottom:6px'>"
            f"{html.escape(m.kind)} run · {html.escape(m.created)}"
            + (f" · {html.escape(Path(m.video).name)}" if m.video else "") + "</div>")
    rows = []
    for c in chips:
        line = (f"<div style='margin:6px 0'>{_pill(c, f'{c.name} · {c.status}')} "
                f"<span style='color:{MUTED};font-size:11px'>{fmt_s(c.seconds)}</span>")
        if c.headline:
            line += f"<div style='color:{INK2};font-size:12px'>{html.escape(c.headline)}</div>"
        if c.error:
            line += f"<div style='color:#e66767;font-size:12px;word-break:break-word'>{html.escape(c.error[:600])}</div>"
        rows.append(line + "</div>")
    return head + "".join(rows)


def runs_html(ms: list[RunManifest]) -> str:
    out = [f"<div style='color:{INK};font-weight:600;margin-top:10px'>All runs</div>"]
    for m in ms:
        dots = "".join(f"<span title='{html.escape(c.name)}: {c.status}' style='display:inline-block;width:9px;height:9px;"
                       f"border-radius:5px;margin-right:2px;background:{c.colour}'></span>" for c in timeline(m))
        out.append(f"<div style='font-size:12px;color:{INK2};margin:3px 0'>{dots} {html.escape(m.run_id)}</div>")
    return "".join(out)


def _style(fig: go.Figure, **axes) -> go.Figure:
    grid = "rgba(255,255,255,0.08)"
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK2, size=12), margin=dict(l=10, r=10, t=30, b=10), barcornerradius=4,
                      legend=dict(orientation="h", x=0, y=1.12), hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False, zeroline=False, **axes.get("x", {}))
    fig.update_yaxes(gridcolor=grid, zeroline=False, **axes.get("y", {}))
    return fig


def ratchet_figure(rows: list[dict]) -> go.Figure:
    x = [f"Draft {re.sub(r'\D', '', r.get('draft', '')) or k + 1}<br>"
         f"{re.sub(r'\s*\(.*?\)', '', r.get('label') or '')}" for k, r in enumerate(rows)]
    fig = go.Figure()
    for name, key, col in (("dev suite", "dev", SERIES[0]), ("held-out (locked)", "held_out", SERIES[1])):
        y = [100 * float(r.get(key) or 0.0) for r in rows]
        fig.add_bar(name=name, x=x, y=y, marker_color=col, text=[f"{v:.0f} %" for v in y], textposition="outside",
                    textfont=dict(color=INK), cliponaxis=False,
                    hovertemplate="%{x}<br>" + name + ": %{y:.0f} %<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    return _style(fig, y=dict(range=[0, 112], ticksuffix=" %"))


def clusters_figure(clusters: dict[str, int]) -> go.Figure:
    items = sorted(clusters.items(), key=lambda kv: kv[1])
    labels = [k.replace("_", " ").lower() for k, _ in items]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=labels, orientation="h", marker_color=SERIES[0],
                           text=[str(v) for _, v in items], textposition="outside", textfont=dict(color=INK),
                           cliponaxis=False, hovertemplate="%{y}: %{x} dev scenes<extra></extra>"))
    fig.update_layout(title=dict(text="Draft 1 failure clusters (dev suite)", font=dict(size=13, color=INK2), x=0),
                      bargap=0.35)
    return _style(fig, x=dict(showticklabels=False))
