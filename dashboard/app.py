"""
MiniEval Memory Gate — live dashboard.

Reads the audit log and serves it as a page a person can actually read, plus
a panel for checking a fact against a source in real time.

This lives outside the installable package on purpose. `pip install
minieval-pro` should give you a scorer and a gate, not a web server. The
dashboard is a local tool and a demo surface, so it stays in the repo.

Run:
    pip install fastapi uvicorn jinja2
    python dashboard/app.py

    then open http://localhost:8000

The gate loads three models on first check (~680MB). Startup is fast; the
first live check is slow. Everything read from the log is instant.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Make the library importable when running this file directly from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn

from minieval_pro.persistence import AuditLog


DEFAULT_LOG = os.environ.get("MINIEVAL_AUDIT_PATH", "demo_audit.jsonl")

app = FastAPI(title="MiniEval Memory Gate", docs_url=None, redoc_url=None)

# The gate is expensive to construct — three models. Build it once, lazily,
# on the first live check rather than at startup, so the page loads instantly
# even if nobody uses the check panel.
_gate = None


def get_gate():
    global _gate
    if _gate is None:
        from minieval_pro.gate import MemoryGate
        _gate = MemoryGate()
    return _gate


def get_log(path: Optional[str] = None) -> AuditLog:
    return AuditLog(path or DEFAULT_LOG)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

class CheckRequest(BaseModel):
    source: str
    fact: str
    record: bool = True


@app.get("/api/summary")
def api_summary():
    """Counts, rates and policy breakdown."""
    return JSONResponse(get_log().summary())


@app.get("/api/entries")
def api_entries(limit: int = 200, verdict: Optional[str] = None):
    """
    Decisions, newest first.

    Newest first because an operator checking on a running system cares about
    what just happened, not what happened when the log was created.
    """
    entries = get_log().entries()
    if verdict:
        entries = [e for e in entries if e.get("verdict") == verdict.upper()]
    entries.reverse()
    return JSONResponse(entries[:limit])


@app.post("/api/check")
def api_check(req: CheckRequest):
    """Run one fact through the gate. Optionally append it to the log."""
    if not req.source.strip() or not req.fact.strip():
        return JSONResponse(
            {"error": "Both a source and a fact are required."}, status_code=400
        )

    decision = get_gate().check(source=req.source, fact=req.fact)
    entry = decision.to_dict()

    if req.record:
        get_log().record(decision)

    return JSONResponse(entry)


@app.get("/api/report", response_class=PlainTextResponse)
def api_report(fmt: str = "text"):
    """The text or markdown report, for copying into a ticket or email."""
    from minieval_pro.persistence import generate_report
    return generate_report(get_log(), fmt=fmt)


@app.get("/api/export.csv", response_class=PlainTextResponse)
def api_export_csv():
    return PlainTextResponse(
        get_log().to_csv(),
        headers={"Content-Disposition": "attachment; filename=minieval_audit.csv"},
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE)


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniEval Memory Gate</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background:#f4f6fa; color:#1a2233; padding:32px 36px 60px;
  }
  .head { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:26px; gap:20px; flex-wrap:wrap; }
  .head h1 { font-size:25px; font-weight:600; letter-spacing:-0.2px; }
  .head p { color:#6b7280; font-size:14px; margin-top:5px; }
  .actions { display:flex; gap:10px; }
  .btn {
    background:#7c6ee6; color:#fff; border:none; padding:9px 17px;
    border-radius:8px; font-size:13.5px; cursor:pointer; font-weight:500;
    text-decoration:none; display:inline-block;
  }
  .btn:hover { background:#6a5bd0; }
  .btn.ghost { background:#fff; color:#4b5563; border:1px solid #e5e9f0; }
  .btn.ghost:hover { background:#f9fafc; }

  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:15px; margin-bottom:22px; }
  .card { background:#fff; border:1px solid #e5e9f0; border-radius:12px; padding:20px 22px; }
  .card .label { color:#6b7280; font-size:12.5px; margin-bottom:7px; }
  .card .value { font-size:32px; font-weight:700; letter-spacing:-0.5px; }
  .card .sub { color:#9ca3af; font-size:12px; margin-top:4px; }
  .v-blue { color:#5b8def; } .v-green { color:#4caf82; }
  .v-purple { color:#7c6ee6; } .v-amber { color:#e0a04d; }

  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:22px; }
  .panel { background:#fff; border:1px solid #e5e9f0; border-radius:12px; padding:20px 22px; margin-bottom:22px; }
  .panel h2 { font-size:15.5px; font-weight:600; margin-bottom:4px; }
  .panel .hint { color:#8b93a3; font-size:12.5px; margin-bottom:16px; }
  .chartbox { height:210px; }

  .try { display:grid; grid-template-columns:1fr 1fr auto; gap:12px; align-items:end; }
  label { display:block; font-size:12px; color:#6b7280; margin-bottom:5px; }
  textarea {
    width:100%; border:1px solid #dfe4ec; border-radius:8px; padding:10px 12px;
    font-size:13.5px; font-family:inherit; resize:vertical; min-height:64px; color:#1a2233;
  }
  textarea:focus { outline:none; border-color:#b9aef2; box-shadow:0 0 0 3px rgba(124,110,230,.12); }

  .result { margin-top:16px; padding:16px 18px; border-radius:10px; border:1px solid #e5e9f0; display:none; }
  .result.show { display:block; }
  .result .top { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  .result .reason { font-size:13.5px; color:#4b5563; line-height:1.5; }
  .result .evidence { margin-top:10px; font-size:12px; color:#8b93a3; font-family:ui-monospace, Menlo, Consolas, monospace; }

  table { width:100%; border-collapse:collapse; }
  th {
    text-align:left; font-size:11px; color:#8b93a3; text-transform:uppercase;
    letter-spacing:.6px; padding:9px 12px; border-bottom:2px solid #eef1f6; font-weight:600;
  }
  td { padding:11px 12px; font-size:13px; border-bottom:1px solid #f2f4f8; vertical-align:top; }
  tr:last-child td { border-bottom:none; }
  .badge { font-size:10.5px; font-weight:700; padding:4px 10px; border-radius:6px; display:inline-block; letter-spacing:.3px; }
  .b-STORE  { background:#e6f4ee; color:#3f9d73; }
  .b-REJECT { background:#efe9fc; color:#7c6ee6; }
  .b-REVIEW { background:#fdf3e2; color:#c98a2e; }
  .fact { font-weight:500; }
  .src  { color:#8b93a3; }
  .num  { font-family:ui-monospace, Menlo, Consolas, monospace; color:#6b7280; font-size:12.5px; }
  .rsn  { color:#8b93a3; font-size:12px; }

  .filters { display:flex; gap:7px; margin-bottom:14px; }
  .chip {
    border:1px solid #e5e9f0; background:#fff; color:#4b5563; padding:6px 13px;
    border-radius:20px; font-size:12.5px; cursor:pointer;
  }
  .chip.on { background:#7c6ee6; color:#fff; border-color:#7c6ee6; }

  .empty { text-align:center; color:#9ca3af; padding:34px; font-size:13.5px; }
  .foot { text-align:center; color:#9ca3af; font-size:12.5px; margin-top:26px; }
  .live { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:#4caf82; }
  .dot { width:7px; height:7px; border-radius:50%; background:#4caf82; }
</style>
</head>
<body>

<div class="head">
  <div>
    <h1>MiniEval Memory Gate</h1>
    <p>Every fact checked for faithfulness before it enters your AI's memory.
       <span class="live"><span class="dot"></span> live</span></p>
  </div>
  <div class="actions">
    <a class="btn ghost" href="/api/report" target="_blank">Report</a>
    <a class="btn" href="/api/export.csv">Export CSV</a>
  </div>
</div>

<div class="stats">
  <div class="card"><div class="label">Facts checked</div><div class="value v-blue" id="s-total">—</div></div>
  <div class="card"><div class="label">Stored</div><div class="value v-green" id="s-store">—</div><div class="sub" id="s-store-rate"></div></div>
  <div class="card"><div class="label">Blocked</div><div class="value v-purple" id="s-reject">—</div><div class="sub" id="s-reject-rate"></div></div>
  <div class="card"><div class="label">Awaiting review</div><div class="value v-amber" id="s-review">—</div><div class="sub" id="s-review-rate"></div></div>
</div>

<div class="panel">
  <h2>Check a fact</h2>
  <div class="hint">Paste what the user said and the fact a memory system extracted from it.
    The first check loads the models and takes a few seconds.</div>
  <div class="try">
    <div>
      <label>Source — what the user actually said</label>
      <textarea id="in-source" placeholder="I am allergic to peanuts."></textarea>
    </div>
    <div>
      <label>Fact — what the AI wants to remember</label>
      <textarea id="in-fact" placeholder="The user loves eating peanuts."></textarea>
    </div>
    <div><button class="btn" id="btn-check" onclick="runCheck()">Check</button></div>
  </div>
  <div class="result" id="result">
    <div class="top"><span class="badge" id="r-badge"></span><span class="num" id="r-score"></span></div>
    <div class="reason" id="r-reason"></div>
    <div class="evidence" id="r-evidence"></div>
  </div>
</div>

<div class="grid2">
  <div class="panel">
    <h2>Outcomes</h2>
    <div class="hint">How every checked fact was decided.</div>
    <div class="chartbox"><canvas id="chart-verdicts"></canvas></div>
  </div>
  <div class="panel">
    <h2>Faithfulness distribution</h2>
    <div class="hint">Where scores fall. Clusters at the extremes are expected —
      the model is usually confident.</div>
    <div class="chartbox"><canvas id="chart-scores"></canvas></div>
  </div>
</div>

<div class="panel">
  <h2>Decision log</h2>
  <div class="hint">Newest first. Every row is a permanent, append-only record.</div>
  <div class="filters">
    <div class="chip on" data-f="ALL" onclick="setFilter('ALL')">All</div>
    <div class="chip" data-f="STORE" onclick="setFilter('STORE')">Stored</div>
    <div class="chip" data-f="REJECT" onclick="setFilter('REJECT')">Blocked</div>
    <div class="chip" data-f="REVIEW" onclick="setFilter('REVIEW')">Review</div>
  </div>
  <table>
    <thead><tr>
      <th style="width:100px">Decision</th>
      <th>Fact</th>
      <th>Source</th>
      <th style="width:70px">Faith</th>
      <th style="width:280px">Reason</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
  <div class="empty" id="empty" style="display:none">No decisions recorded yet. Check a fact above.</div>
</div>

<div class="panel" id="policy-panel" style="display:none">
  <h2>Policies in force</h2>
  <div class="hint" id="policy-hint">The rules that produced these decisions.</div>
  <table>
    <thead><tr><th>Policy</th><th>Version</th><th>Fingerprint</th><th>Decisions</th></tr></thead>
    <tbody id="policy-body"></tbody>
  </table>
</div>

<div class="foot">MiniEval Memory Gate · every decision recorded with its evidence and the policy that produced it</div>

<script>
let filter = 'ALL';
let entries = [];
let chartVerdicts = null, chartScores = null;

async function load() {
  const [sum, ent] = await Promise.all([
    fetch('/api/summary').then(r => r.json()),
    fetch('/api/entries?limit=500').then(r => r.json()),
  ]);
  entries = ent;
  renderStats(sum);
  renderPolicies(sum);
  renderTable();
  renderCharts(sum);
}

function renderStats(s) {
  document.getElementById('s-total').textContent  = s.total;
  document.getElementById('s-store').textContent  = s.stored;
  document.getElementById('s-reject').textContent = s.rejected;
  document.getElementById('s-review').textContent = s.flagged_for_review;
  document.getElementById('s-store-rate').textContent  = s.total ? s.store_rate  + '% of checks' : '';
  document.getElementById('s-reject-rate').textContent = s.total ? s.reject_rate + '% of checks' : '';
  document.getElementById('s-review-rate').textContent = s.total ? s.review_rate + '% of checks' : '';
}

function renderPolicies(s) {
  const keys = Object.keys(s.policies || {});
  const panel = document.getElementById('policy-panel');
  if (!keys.length) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';
  document.getElementById('policy-hint').textContent = keys.length > 1
    ? 'More than one policy appears in this log. Overall rates combine periods governed by different rules.'
    : 'The rules that produced these decisions.';
  document.getElementById('policy-body').innerHTML = keys.map(k => {
    const p = s.policies[k];
    return `<tr><td class="fact">${esc(p.name)}</td><td>${esc(p.version)}</td>
            <td class="num">${esc(k)}</td><td class="num">${p.count}</td></tr>`;
  }).join('');
}

function setFilter(f) {
  filter = f;
  document.querySelectorAll('.chip').forEach(c =>
    c.classList.toggle('on', c.dataset.f === f));
  renderTable();
}

function renderTable() {
  const rows = filter === 'ALL' ? entries : entries.filter(e => e.verdict === filter);
  const tbody = document.getElementById('tbody');
  document.getElementById('empty').style.display = rows.length ? 'none' : 'block';
  tbody.innerHTML = rows.map(e => `
    <tr>
      <td><span class="badge b-${esc(e.verdict)}">${esc(e.verdict)}</span></td>
      <td class="fact">${esc(e.fact)}</td>
      <td class="src">${esc(e.source)}</td>
      <td class="num">${(e.faithfulness ?? 0).toFixed(2)}</td>
      <td class="rsn">${esc(e.reason || '')}</td>
    </tr>`).join('');
}

function renderCharts(s) {
  const vc = document.getElementById('chart-verdicts').getContext('2d');
  const labels = ['Stored', 'Blocked', 'Review'];
  const data = [s.stored, s.rejected, s.flagged_for_review];
  const colors = ['#a9dcc4', '#c3b1e1', '#f7d9a8'];

  if (chartVerdicts) chartVerdicts.destroy();
  chartVerdicts = new Chart(vc, {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
    options: { responsive:true, maintainAspectRatio:false, cutout:'62%',
      plugins:{ legend:{ position:'right', labels:{ color:'#6b7280', boxWidth:12, padding:14 } } } }
  });

  // Faithfulness histogram in ten buckets.
  const buckets = new Array(10).fill(0);
  entries.forEach(e => {
    const v = Math.max(0, Math.min(0.999, e.faithfulness ?? 0));
    buckets[Math.floor(v * 10)]++;
  });
  const bLabels = buckets.map((_, i) => (i / 10).toFixed(1));

  const sc = document.getElementById('chart-scores').getContext('2d');
  if (chartScores) chartScores.destroy();
  chartScores = new Chart(sc, {
    type: 'bar',
    data: { labels: bLabels, datasets: [{ data: buckets, backgroundColor: '#a7c7e7', borderRadius: 5 }] },
    options: { responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ display:false } },
      scales:{ y:{ beginAtZero:true, ticks:{ color:'#9ca3af', precision:0 }, grid:{ color:'#eef1f6' } },
               x:{ ticks:{ color:'#9ca3af' }, grid:{ display:false } } } }
  });
}

async function runCheck() {
  const source = document.getElementById('in-source').value.trim();
  const fact   = document.getElementById('in-fact').value.trim();
  const btn    = document.getElementById('btn-check');
  const box    = document.getElementById('result');

  if (!source || !fact) { alert('Enter both a source and a fact.'); return; }

  btn.disabled = true; btn.textContent = 'Checking…';
  try {
    const res = await fetch('/api/check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, fact, record: true }),
    });
    const d = await res.json();
    if (d.error) { alert(d.error); return; }

    document.getElementById('r-badge').textContent = d.verdict;
    document.getElementById('r-badge').className = 'badge b-' + d.verdict;
    document.getElementById('r-score').textContent = 'faithfulness ' + (d.faithfulness ?? 0).toFixed(3);
    document.getElementById('r-reason').textContent = d.reason || '';
    const bits = [];
    if (d.entailment != null)    bits.push('entailment ' + d.entailment.toFixed(3));
    if (d.contradiction != null) bits.push('contradiction ' + d.contradiction.toFixed(3));
    if (d.neutral != null)       bits.push('neutral ' + d.neutral.toFixed(3));
    if (d.relatedness != null)   bits.push('relatedness ' + d.relatedness.toFixed(3));
    document.getElementById('r-evidence').textContent = bits.join('   ');
    box.classList.add('show');

    await load();
  } catch (err) {
    alert('Check failed: ' + err);
  } finally {
    btn.disabled = false; btn.textContent = 'Check';
  }
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

load();
setInterval(load, 5000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"[MiniEval] Dashboard reading: {DEFAULT_LOG}")
    print("[MiniEval] http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")