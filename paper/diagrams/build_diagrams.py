#!/usr/bin/env python3
"""
Generates the paper's draw.io diagrams (*.drawio, editable in diagrams.net /
draw.io desktop). Export to PDF with draw.io's own renderer:

    drawio -x -f pdf --crop -o ../figures/<name>.pdf <name>.drawio

(make figures does both steps.)
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

FONT = "fontFamily=Helvetica;fontSize=12;"
BOX = FONT + "rounded=1;arcSize=8;whiteSpace=wrap;html=1;strokeColor=#333333;fillColor=#FFFFFF;"
KEY = FONT + "rounded=1;arcSize=8;whiteSpace=wrap;html=1;strokeColor=#1F4E79;fillColor=#E8F0FA;strokeWidth=1.5;"
GROUP = ("fontFamily=Helvetica;fontSize=12;fontStyle=1;rounded=1;arcSize=3;whiteSpace=wrap;html=1;"
         "strokeColor=#9E9E9E;dashed=1;fillColor=#FAFAFA;verticalAlign=top;align=left;spacingLeft=8;spacingTop=4;")
NOTE = FONT + "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontSize=11;fontColor=#333333;"
EDGE = "endArrow=block;endFill=1;html=1;strokeColor=#333333;fontFamily=Helvetica;fontSize=11;labelBackgroundColor=#FFFFFF;"
DASH = EDGE + "dashed=1;"


class Diagram:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cells: list[str] = []
        self.n = 1

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def box(self, label, x, y, w, h, style=BOX):
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def edge(self, s, t, label="", style=EDGE, pts=()):
        i = self._id()
        wp = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts)
        arr = f'<Array as="points">{wp}</Array>' if pts else ""
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label.replace(chr(10), "<br>"), {chr(34): "&quot;"})}" style="{style}" edge="1" parent="1" source="{s}" target="{t}">'
            f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')
        return i

    def arrow(self, x1, y1, x2, y2, label="", style=EDGE):
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
            f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
        return i

    def save(self, folder: Path) -> Path:
        xml = ('<mxfile host="drawio"><diagram name="{n}" id="{n}"><mxGraphModel dx="1200" dy="800" grid="1" '
               'gridSize="10" guides="1" page="0" math="0" shadow="0"><root><mxCell id="0"/>'
               '<mxCell id="1" parent="0"/>{c}</root></mxGraphModel></diagram></mxfile>').format(
            n=self.name, c="".join(self.cells))
        p = folder / f"{self.name}.drawio"
        p.write_text(xml)
        return p


def architecture() -> Diagram:
    d = Diagram("architecture")
    # tier 1: agent fleet
    d.box("Agent fleet (12 roles)", 0, 0, 900, 90, GROUP)
    a1 = d.box("Coding agent", 20, 32, 150, 42)
    a2 = d.box("Security agent", 190, 32, 150, 42)
    d.box("…", 360, 32, 60, 42, NOTE + "align=center;fontSize=16;")
    a3 = d.box("Deployment agent", 440, 32, 150, 42)
    a4 = d.box("IDE assistant<br><i>(MCP client)</i>", 700, 32, 180, 42)
    # tier 2: interfaces
    d.box("Interfaces — identity from API key (SHA-256 hashed, per agent)", 0, 120, 900, 90, GROUP)
    rest = d.box("<b>REST API</b> (FastAPI)<br>/context · /experiences · /reuse", 120, 152, 330, 46, KEY)
    mcp = d.box("<b>MCP stdio server</b><br>bound to one agent key", 620, 152, 260, 46, KEY)
    for a in (a1, a2, a3):
        d.edge(a, rest, "", EDGE)
    d.edge(a4, mcp, "", EDGE)
    # tier 3: service
    d.box("MemoryService", 0, 240, 900, 150, GROUP)
    ctx = d.box("<b>Context builder</b><br>relevance ≥ τ · confidence floor<br>trust rank · dedupe · token budget",
                20, 275, 270, 95, KEY)
    judge = d.box("<b>Memory Judge</b><br>five-tier ladder<br>Beta-posterior confidence<br>quarantine",
                  315, 275, 270, 95, KEY)
    ingest = d.box("<b>Ingest pipeline</b><br>initial tier → extract steps<br>→ embed → one transaction",
                   610, 275, 270, 95, KEY)
    d.edge(rest, ctx, "retrieve", EDGE)
    d.edge(rest, judge, "outcome report", EDGE)
    d.edge(mcp, ingest, "", EDGE)
    d.edge(mcp, judge, "", EDGE)
    # tier 4: storage
    d.box("Storage and messaging", 0, 420, 900, 150, GROUP)
    pg = d.box("<b>PostgreSQL 16 + pgvector</b><br>experiences (vector, tsvector, tier, confidence)<br>"
               "injections · reuse_events · tier_transitions · agents",
               20, 455, 540, 95,
               FONT + "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=12;"
                      "strokeColor=#333333;fillColor=#FFFFFF;")
    sqs = d.box("<b>Amazon SQS</b><br>queued writes,<br>at-least-once", 580, 465, 130, 70, BOX)
    worker = d.box("<b>Worker</b><br>poison-message cap", 750, 470, 130, 60, BOX)
    d.edge(ctx, pg, "hybrid search\n(visibility in SQL)", EDGE)
    d.edge(judge, pg, "SELECT … FOR UPDATE", EDGE)
    d.edge(ingest, pg, "idempotent insert", EDGE + "exitX=0.25;exitY=1;exitDx=0;exitDy=0;", pts=[(677, 410), (420, 410)])
    d.edge(sqs, worker, "", EDGE)
    d.edge(rest, sqs, "", DASH + "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0.2;entryY=0;entryDx=0;entryDy=0;",
           pts=[(597, 175), (597, 460), (606, 460)])
    d.edge(worker, ingest, "", DASH + "entryX=0.7;entryY=1;entryDx=0;entryDy=0;")
    return d


def lifecycle() -> Diagram:
    d = Diagram("lifecycle")
    cols = {"Agent": 60, "REST API": 300, "MemoryService": 500, "Memory Judge": 680, "PostgreSQL": 860}
    top, bottom = 0, 700
    for name, x in cols.items():
        d.box(f"<b>{name}</b>", x - 70, top, 140, 40, KEY)
        d.arrow(x, top + 40, x, bottom, "", "endArrow=none;dashed=1;strokeColor=#9E9E9E;html=1;")
    y = 70

    def msg(a, b, text, dashed=False):
        nonlocal y
        above = "verticalAlign=bottom;labelBackgroundColor=none;spacingBottom=2;"
        d.arrow(cols[a], y, cols[b], y, text, (DASH if dashed else EDGE) + above)
        y += 36

    def phase(text):
        nonlocal y
        d.box(text, 0, y - 12, 940, 22,
              FONT + "text;html=1;align=left;fontStyle=3;fontSize=11;fontColor=#1F4E79;")
        y += 22

    phase("1 · Retrieve")
    msg("Agent", "REST API", "POST /context  (X-API-Key)")
    msg("REST API", "MemoryService", "prepare_context(identity from key)")
    msg("MemoryService", "PostgreSQL", "hybrid search: visibility + scope")
    msg("PostgreSQL", "MemoryService", "k_c candidates", dashed=True)
    msg("MemoryService", "PostgreSQL", "record injections (agent, memory)")
    msg("MemoryService", "Agent", "compact procedure + experience ids", dashed=True)
    phase("2 · Act — the agent prepends the block to its prompt and runs the task")
    y += 6
    phase("3 · Store")
    msg("Agent", "REST API", "POST /experiences")
    msg("REST API", "MemoryService", "ingest: tier → extract → embed")
    msg("MemoryService", "PostgreSQL", "insert row + audit (one transaction)")
    phase("4 · Report")
    msg("Agent", "REST API", "POST /experiences/{id}/reuse {success}")
    msg("REST API", "MemoryService", "record_reuse(reporter from key)")
    msg("MemoryService", "PostgreSQL", "lock row; consume unreported injection")
    msg("MemoryService", "Memory Judge", "decide(tier, n, s)")
    msg("Memory Judge", "MemoryService", "transition or none", dashed=True)
    msg("MemoryService", "PostgreSQL", "update confidence / tier + audit")
    return d


def governance() -> Diagram:
    d = Diagram("governance")
    st = FONT + "rounded=1;arcSize=30;whiteSpace=wrap;html=1;strokeColor=#1F4E79;fillColor=#E8F0FA;strokeWidth=1.5;fontStyle=1;"
    y = 120
    sub = '<br><font style="font-size:10px;font-weight:normal;color:#555555">{}</font>'
    priv = d.box("PRIVATE" + sub.format("author only"), 0, y, 130, 56,
                 FONT + "rounded=1;arcSize=30;whiteSpace=wrap;html=1;strokeColor=#8B0000;fillColor=#FBEAEA;strokeWidth=1.5;fontStyle=1;")
    agent = d.box("AGENT" + sub.format("author + same role"), 220, y, 130, 56, st)
    shared = d.box("SHARED" + sub.format("all agents"), 440, y, 130, 56, st)
    val = d.box("VALIDATED" + sub.format("all agents"), 660, y, 130, 56, st)
    org = d.box("ORGANIZATIONAL" + sub.format("all agents"), 880, y, 150, 56, st)
    s1 = d.box("", 270, 20, 30, 30, "ellipse;fillColor=#333333;strokeColor=none;html=1;")
    s2 = d.box("", 50, 20, 30, 30, "ellipse;fillColor=#333333;strokeColor=none;html=1;")
    d.edge(s1, agent, "self-assessed success", EDGE)
    d.edge(s2, priv, "self-assessed failure", EDGE)
    d.edge(agent, shared, "≥ 1 counted\nsuccess", EDGE)
    d.edge(shared, val, "n ≥ 3,\nrate ≥ 0.8", EDGE)
    d.edge(val, org, "n ≥ 10,\nrate ≥ 0.9", EDGE)
    q = EDGE + "strokeColor=#8B0000;fontColor=#8B0000;exitX=0.5;exitY=1;entryX=0.5;entryY=1;"
    for src, x in ((agent, 285), (shared, 505), (val, 725), (org, 955)):
        d.edge(src, priv, "", q, pts=[(x, 230), (65, 230)])
    d.box("quarantine: n ≥ 3 and rate &lt; 0.4", 380, 220, 260, 20,
          NOTE + "align=center;fontColor=#8B0000;fontSize=11;labelBackgroundColor=#FFFFFF;")
    d.box("<b>Counted report</b>: consumes an unreported injection of the memory to a non-author, "
          "or is the author's first negative report. Positive self-reports never count.<br>"
          "<b>Confidence</b> after each counted report: c = (s + 1.2) / (n + 2); memories with c &lt; 0.45 are not injected.",
          0, 262, 1030, 50,
          FONT + "rounded=1;whiteSpace=wrap;html=1;strokeColor=#9E9E9E;fillColor=#FAFAFA;align=left;spacingLeft=8;fontSize=11;")
    return d


def pipeline() -> Diagram:
    d = Diagram("pipeline")
    steps = [
        ("<b>Query</b><br>task + tech tags", BOX),
        ("<b>Embed</b><br>1536-d", BOX),
        ("<b>Hybrid search</b><br>k<sub>c</sub> = 10, visible,<br>in scope", KEY),
        ("<b>Relevance</b><br>r ≥ τ = 0.30", KEY),
        ("<b>Confidence</b><br>c ≥ 0.45", KEY),
        ("<b>Rank</b><br>r + 0.1·c", KEY),
        ("<b>Dedupe</b><br>same family or<br>sim ≥ 0.85", KEY),
        ("<b>Budget</b><br>k = 1, ≤ 1000 tok", KEY),
    ]
    prev = None
    for i, (label, style) in enumerate(steps):
        b = d.box(label, i * 128, 0, 108, 64, style)
        if prev:
            d.edge(prev, b, "", EDGE)
        prev = b
    out = d.box("<b>Context block</b><br>+ injection ledger", 8 * 128, 0, 120, 64, BOX)
    d.edge(prev, out, "", EDGE)
    d.box("If no candidate survives, nothing is injected.", 3 * 128, 72, 500, 20, NOTE + "fontSize=10;fontColor=#555555;")
    return d


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for fn in (architecture, lifecycle, governance, pipeline):
        print(fn().save(here))
