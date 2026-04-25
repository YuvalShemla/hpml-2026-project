#!/usr/bin/env python3
"""Build an HTML dashboard visualizing Gemma 4 end-to-end execution results.

Shows the full LLM call sequence for each scenario:
  DISCOVER → PLAN (LLM) → [ARG RESOLVE (LLM)] → TOOL CALL (MCP) → SUMMARIZE (LLM)

Compares Gemma 4 plans against Gemini 2.5 Flash gold plans and evaluates
execution success and answer quality.
"""

import json
import os
import html
import re
from datetime import datetime

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "gemma4_e2e_results.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "gemma4_e2e_dashboard.html")

AGENT_COLORS = {
    "IoTAgent": ("#3b82f6", "#1e3a5f"),
    "FMSRAgent": ("#8b5cf6", "#3b1f6e"),
    "TSFMAgent": ("#f59e0b", "#5f3a0a"),
    "Utilities": ("#6b7280", "#374151"),
}


def parse_gold_plan(gold_response):
    steps = []
    if not gold_response:
        return steps
    blocks = re.split(r'(?=#Task\d+:)', gold_response)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        step_m = re.search(r'#Task(\d+):\s*(.*)', block)
        agent_m = re.search(r'#Agent\d+:\s*(.*)', block)
        tool_m = re.search(r'#Tool\d+:\s*(.*)', block)
        args_m = re.search(r'#Args\d+:\s*(.*)', block)
        dep_m = re.search(r'#Dependency\d+:\s*(.*)', block)
        if step_m:
            steps.append({
                "step": int(step_m.group(1)),
                "task": step_m.group(2).strip(),
                "agent": (agent_m.group(1).strip() if agent_m else ""),
                "tool": (tool_m.group(1).strip() if tool_m else ""),
                "args": (args_m.group(1).strip() if args_m else "{}"),
                "dependency": (dep_m.group(1).strip() if dep_m else "None"),
            })
    return steps


def esc(text):
    return html.escape(str(text)) if text else ""


def truncate(text, max_len=800):
    if not text:
        return ""
    text = str(text)
    if len(text) > max_len:
        return text[:max_len] + f"\n... [{len(text)} chars total]"
    return text


def fmt_json(resp):
    if not resp:
        return ""
    try:
        return json.dumps(json.loads(resp), indent=2)
    except (json.JSONDecodeError, TypeError):
        return resp


def plan_match_score(gold_steps, model_plan):
    """Simple plan comparison: check agent-tool pair overlap."""
    if not gold_steps or not model_plan:
        return 0, []
    gold_pairs = set()
    for s in gold_steps:
        gold_pairs.add((s.get("agent", ""), s.get("tool", "")))
    model_pairs = set()
    for s in model_plan:
        model_pairs.add((s.get("agent", ""), s.get("tool", "")))
    matched = gold_pairs & model_pairs
    total = gold_pairs | model_pairs
    score = len(matched) / len(total) if total else 0

    notes = []
    missing = gold_pairs - model_pairs
    extra = model_pairs - gold_pairs
    if missing:
        notes.append(f"Missing: {', '.join(f'{a}.{t}' for a,t in missing)}")
    if extra:
        notes.append(f"Extra: {', '.join(f'{a}.{t}' for a,t in extra)}")
    if len(model_plan) > len(gold_steps):
        notes.append(f"Over-decomposed: {len(model_plan)} vs {len(gold_steps)} gold steps")
    elif len(model_plan) < len(gold_steps):
        notes.append(f"Under-decomposed: {len(model_plan)} vs {len(gold_steps)} gold steps")
    else:
        notes.append(f"Step count matches gold ({len(gold_steps)})")

    return score, notes


def build_html(results):
    ran = [r for r in results if r.get("plan_steps", 0) > 0]
    ok = [r for r in ran if r.get("all_succeeded")]
    failed_exec = [r for r in ran if not r.get("all_succeeded")]
    rate_limited = [r for r in results if r.get("error") and "429" in str(r.get("error", ""))]
    timed_out = [r for r in results if not r.get("all_succeeded") and r.get("plan_steps", 0) == 0 and r.get("error") and "429" not in str(r.get("error", ""))]

    total_steps = sum(r.get("plan_steps", 0) for r in ran)
    steps_ok = sum(r.get("steps_succeeded", 0) for r in ran)

    # Count LLM calls and token usage per scenario
    for r in ran:
        llm_calls = r.get("llm_calls", [])
        if llm_calls:
            r["_llm_calls"] = len(llm_calls)
        else:
            n_llm = 1  # planning call
            for s in r.get("plan", []):
                args = s.get("tool_args", {})
                if any(isinstance(v, str) and "{step_" in v for v in args.values()):
                    n_llm += 1
            n_llm += 1  # summarize call
            r["_llm_calls"] = n_llm
        r["_tool_calls"] = r.get("plan_steps", 0)

    avg_llm = sum(r.get("_llm_calls", 0) for r in ran) / len(ran) if ran else 0
    avg_tool = sum(r.get("_tool_calls", 0) for r in ran) / len(ran) if ran else 0

    # Token stats
    total_tokens_all = sum(r.get("token_summary", {}).get("total", 0) for r in ran)
    total_input_all = sum(r.get("token_summary", {}).get("total_input", 0) for r in ran)
    total_output_all = sum(r.get("token_summary", {}).get("total_output", 0) for r in ran)
    total_tool_desc = sum(r.get("token_summary", {}).get("tool_desc_tokens_est", 0) for r in ran)
    avg_tokens = total_tokens_all / len(ran) if ran else 0
    tool_desc_pct = (total_tool_desc / total_input_all * 100) if total_input_all else 0
    has_tokens = total_tokens_all > 0

    # Median tokens per scenario
    scenario_totals = sorted(r.get("token_summary", {}).get("total", 0) for r in ran)
    if scenario_totals:
        mid = len(scenario_totals) // 2
        if len(scenario_totals) % 2 == 0:
            median_tokens = (scenario_totals[mid - 1] + scenario_totals[mid]) / 2
        else:
            median_tokens = scenario_totals[mid]
    else:
        median_tokens = 0

    # Compute plan quality for ran scenarios
    plan_scores = []
    for r in ran:
        gold = parse_gold_plan(r.get("gold_response", ""))
        score, _ = plan_match_score(gold, r.get("plan", []))
        plan_scores.append(score)
    avg_plan_score = sum(plan_scores) / len(plan_scores) if plan_scores else 0

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemma 4 26B-A4B — End-to-End Execution Dashboard</title>
<style>
:root {{
    --bg-0: #0f172a; --bg-1: #1e293b; --bg-2: #263548; --bg-3: #0a1628;
    --border: #334155; --text: #e2e8f0; --text-dim: #94a3b8; --text-muted: #64748b;
    --green: #4ade80; --red: #f87171; --blue: #60a5fa; --amber: #fbbf24; --purple: #a78bfa;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: var(--bg-0); color: var(--text); line-height: 1.6; }}
.container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 1.8rem; color: #f8fafc; }}
.subtitle {{ color: var(--text-dim); font-size: 0.95rem; margin-bottom: 28px; }}
code, .mono {{ font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Fira Code', monospace; }}

/* Summary */
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 32px; }}
.card {{ background: var(--bg-1); border-radius: 12px; padding: 18px; border: 1px solid var(--border); }}
.card-value {{ font-size: 1.8rem; font-weight: 700; }}
.card-label {{ font-size: 0.82rem; color: var(--text-dim); margin-top: 2px; }}
.green {{ color: var(--green); }} .red {{ color: var(--red); }} .blue {{ color: var(--blue); }} .amber {{ color: var(--amber); }}

/* Legend */
.legend {{ display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.82rem; color: var(--text-dim); }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
.legend-dot.llm {{ background: var(--purple); }}
.legend-dot.mcp {{ background: var(--blue); }}
.legend-dot.ok {{ background: var(--green); }}
.legend-dot.err {{ background: var(--red); }}

/* Scenarios */
.scenario {{ background: var(--bg-1); border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--border); overflow: hidden; }}
.scenario-header {{ padding: 14px 20px; cursor: pointer; display: flex; align-items: center; gap: 12px; }}
.scenario-header:hover {{ background: var(--bg-2); }}
.badge {{ padding: 3px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; }}
.badge-ok {{ background: #166534; color: var(--green); }}
.badge-fail {{ background: #7f1d1d; color: #fca5a5; }}
.badge-timeout {{ background: #78350f; color: #fcd34d; }}
.badge-rate {{ background: #44403c; color: #d6d3d1; }}
.scenario-q {{ flex: 1; font-weight: 500; font-size: 0.95rem; }}
.scenario-meta {{ display: flex; gap: 14px; font-size: 0.78rem; color: var(--text-muted); }}
.meta-tag {{ background: var(--bg-0); padding: 2px 8px; border-radius: 4px; }}
.chevron {{ transition: transform 0.2s; color: var(--text-muted); font-size: 0.8rem; }}
.scenario.open .chevron {{ transform: rotate(90deg); }}
.scenario-body {{ display: none; padding: 0 20px 20px; }}
.scenario.open .scenario-body {{ display: block; }}

/* Pipeline visualization */
.pipeline {{ margin-top: 16px; }}
.pipeline-label {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }}
.pipeline-label.llm-label {{ color: var(--purple); }}
.pipeline-label.tool-label {{ color: var(--blue); }}

.call-block {{ border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; position: relative; }}
.call-block.llm-call {{ background: #1a1030; border: 1px solid #3b2068; }}
.call-block.tool-call {{ background: #0f1a2e; border: 1px solid #1e3a5f; }}
.call-block.tool-call.success {{ border-left: 3px solid var(--green); }}
.call-block.tool-call.failure {{ border-left: 3px solid var(--red); }}

.call-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
.call-type {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; padding: 2px 8px; border-radius: 4px; }}
.call-type.llm {{ background: #2d1a4e; color: var(--purple); }}
.call-type.mcp {{ background: #0f2a4e; color: var(--blue); }}
.call-type.result {{ background: #0f2e1a; color: var(--green); }}
.call-step {{ font-weight: 600; font-size: 0.85rem; }}
.agent-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; color: white; }}
.tool-name {{ font-family: monospace; color: #93c5fd; font-size: 0.85rem; }}

.call-desc {{ font-size: 0.88rem; color: #cbd5e1; margin-bottom: 8px; }}
.call-io {{ margin-top: 8px; }}
.io-label {{ font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 4px; }}
.io-content {{ font-family: monospace; font-size: 0.78rem; padding: 10px 12px; border-radius: 6px; white-space: pre-wrap; word-break: break-all; max-height: 250px; overflow-y: auto; }}
.io-input {{ background: #1a1a2e; color: #a5b4fc; border: 1px solid #2d2d5e; }}
.io-output {{ background: #0c1a0c; color: #86efac; border: 1px solid #1e3a1e; }}
.io-error {{ background: #1c0f0f; color: #fca5a5; border: 1px solid #3a1e1e; }}

/* Arrow connector */
.connector {{ display: flex; align-items: center; justify-content: center; padding: 4px 0; color: var(--text-muted); font-size: 0.8rem; }}

/* Evaluation panel */
.eval-panel {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px; }}
.eval-section {{ padding: 14px; border-radius: 8px; }}
.eval-gold {{ background: var(--bg-3); border: 1px solid #1e3a5f; }}
.eval-score {{ background: #0f1a0f; border: 1px solid #1e3a1e; }}
.eval-label {{ font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }}
.eval-gold .eval-label {{ color: var(--blue); }}
.eval-score .eval-label {{ color: var(--green); }}
.eval-plan {{ font-family: monospace; font-size: 0.76rem; white-space: pre-wrap; color: var(--text-dim); line-height: 1.5; }}
.score-item {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1e3a1e; }}
.score-label {{ color: var(--text-dim); font-size: 0.85rem; }}
.score-value {{ font-weight: 700; font-size: 0.85rem; }}
.score-bar {{ height: 6px; border-radius: 3px; background: #1e293b; margin-top: 4px; }}
.score-fill {{ height: 100%; border-radius: 3px; }}
.score-note {{ font-size: 0.78rem; color: var(--text-muted); margin-top: 8px; padding: 8px; background: var(--bg-0); border-radius: 6px; }}

/* Answer */
.answer-block {{ margin-top: 16px; padding: 16px; background: #0a0f1e; border: 1px solid #1e2a4f; border-radius: 8px; }}
.answer-label {{ font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--amber); margin-bottom: 6px; }}
.answer-text {{ color: var(--text); font-size: 0.92rem; }}

@media (max-width: 768px) {{
    .eval-panel {{ grid-template-columns: 1fr; }}
    .scenario-meta {{ flex-direction: column; gap: 4px; }}
}}
</style>
</head>
<body>
<div class="container">
<h1>Gemma 4 26B-A4B — End-to-End Execution</h1>
<p class="subtitle">
    AssetOpsBench: Question → <span style="color:var(--purple)">LLM Plan</span> →
    <span style="color:var(--blue)">MCP Tool Calls</span> →
    <span style="color:var(--purple)">LLM Summarize</span> →
    Answer &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>

<div class="summary">
    <div class="card"><div class="card-value blue">{len(ran)}</div><div class="card-label">Scenarios Ran</div></div>
    <div class="card"><div class="card-value green">{len(ok)}/{len(ran)}</div><div class="card-label">Full E2E Success</div></div>
    <div class="card"><div class="card-value {'green' if steps_ok==total_steps else 'amber'}">{steps_ok}/{total_steps}</div><div class="card-label">Tool Steps OK</div></div>
    <div class="card"><div class="card-value" style="color:var(--purple)">{avg_llm:.1f}</div><div class="card-label">Avg LLM Calls/Scenario</div></div>
    <div class="card"><div class="card-value blue">{avg_tool:.1f}</div><div class="card-label">Avg Tool Calls/Scenario</div></div>
    <div class="card"><div class="card-value amber">{avg_plan_score*100:.0f}%</div><div class="card-label">Avg Plan Match (vs Ref)</div></div>
    {f'<div class="card"><div class="card-value">{median_tokens:,.0f}</div><div class="card-label">Median Tokens/Scenario</div></div>' if has_tokens else ''}
    {f'<div class="card"><div class="card-value">{total_input_all:,}</div><div class="card-label">Total Input Tokens</div></div>' if has_tokens else ''}
    {f'<div class="card"><div class="card-value red">{tool_desc_pct:.0f}%</div><div class="card-label">Tool Desc % of Input</div></div>' if has_tokens else ''}
    <div class="card"><div class="card-value">3.8B</div><div class="card-label">Active Params (MoE 26B)</div></div>
</div>

{f"""<div style="background:var(--bg-1);border-radius:12px;padding:16px 20px;margin-bottom:24px;border:1px solid var(--border);display:flex;gap:32px;flex-wrap:wrap;font-size:0.85rem">
    <div><span style="color:var(--text-muted)">Total tokens:</span> <strong>{total_tokens_all:,}</strong></div>
    <div><span style="color:var(--text-muted)">Input:</span> <strong>{total_input_all:,}</strong></div>
    <div><span style="color:var(--text-muted)">Output:</span> <strong>{total_output_all:,}</strong></div>
    <div><span style="color:var(--text-muted)">Tool descriptions (est):</span> <strong style="color:var(--red)">{total_tool_desc:,}</strong> <span style="color:var(--text-muted)">({tool_desc_pct:.0f}% of input)</span></div>
    <div><span style="color:var(--text-muted)">Tool desc = prompt overhead that fine-tuning can eliminate</span></div>
</div>""" if has_tokens else ''}

<div class="legend">
    <div class="legend-item"><div class="legend-dot llm"></div> LLM Call (Gemma 4)</div>
    <div class="legend-item"><div class="legend-dot mcp"></div> MCP Tool Call</div>
    <div class="legend-item"><div class="legend-dot ok"></div> Success</div>
    <div class="legend-item"><div class="legend-dot err"></div> Error</div>
</div>
""")

    for r in results:
        is_success = r.get("all_succeeded", False)
        plan_steps = r.get("plan_steps", 0)
        is_rate = r.get("error") and "429" in str(r.get("error", ""))
        is_timeout = not is_success and plan_steps == 0 and not is_rate and r.get("error")

        if is_success:
            badge_cls, badge_txt = "badge-ok", "E2E SUCCESS"
        elif is_rate:
            badge_cls, badge_txt = "badge-rate", "RATE LIMITED"
        elif is_timeout:
            badge_cls, badge_txt = "badge-timeout", "TIMEOUT"
        else:
            badge_cls, badge_txt = "badge-fail", "FAILED"

        gold_steps = parse_gold_plan(r.get("gold_response", ""))
        score, notes = plan_match_score(gold_steps, r.get("plan", []))

        open_cls = "open" if is_success else ""

        parts.append(f"""
<div class="scenario {open_cls}" id="s-{r['id']}">
  <div class="scenario-header" onclick="this.parentElement.classList.toggle('open')">
    <span class="chevron">&#9654;</span>
    <span class="badge {badge_cls}">{badge_txt}</span>
    <span class="scenario-q">{esc(r['prompt'])}</span>
    <div class="scenario-meta">
      <span class="meta-tag">ID #{r['id']}</span>
      <span class="meta-tag">{plan_steps} steps (gold: {r.get('gold_steps', '?')})</span>
      <span class="meta-tag">{r.get('elapsed_seconds', 0):.1f}s</span>
      {f'<span class="meta-tag" style="color:var(--green)">{score*100:.0f}% match</span>' if plan_steps > 0 else ''}
    </div>
  </div>
  <div class="scenario-body">
""")

        if is_rate:
            parts.append('<div style="padding:20px;color:#fcd34d;text-align:center">Rate limited by Gemini API (16K tokens/min for Gemma 4). Will retry.</div>')
        elif is_timeout:
            parts.append('<div style="padding:20px;color:#fcd34d;text-align:center">Timed out (>600s). Query may require very large tool responses.</div>')
        elif plan_steps > 0:
            # === PIPELINE VISUALIZATION ===
            parts.append('<div class="pipeline">')

            # Get per-call token data if available
            llm_calls_data = r.get("llm_calls", [])
            tok_summary = r.get("token_summary", {})

            def tok_badge(call_data):
                if not call_data:
                    return ""
                inp = call_data.get("input_tokens", 0)
                out = call_data.get("output_tokens", 0)
                if inp == 0 and out == 0:
                    return ""
                return f'<span style="margin-left:auto;font-size:0.72rem;color:var(--text-muted);font-family:monospace">{inp:,} in / {out:,} out = {inp+out:,} tok</span>'

            plan_call_data = llm_calls_data[0] if llm_calls_data else {}
            tool_desc_est = tok_summary.get("tool_desc_tokens_est", 0)
            plan_input = plan_call_data.get("input_tokens", 0)

            # Step 1: DISCOVER (implicit)
            parts.append(f"""
    <div class="call-block llm-call">
      <div class="call-header">
        <span class="call-type llm">LLM CALL #1</span>
        <span class="call-step">PLAN GENERATION</span>
        {tok_badge(plan_call_data)}
      </div>
      <div class="call-desc">Gemma 4 receives the question + tool descriptions and produces a structured plan.</div>
      <div class="call-io">
        <div class="io-label">Input (question)</div>
        <div class="io-content io-input">{esc(r['prompt'])}</div>
      </div>
      <div class="call-io">
        <div class="io-label">Output (plan — {plan_steps} steps)</div>
        <div class="io-content io-output">""")

            for s in r.get("plan", []):
                deps = ", ".join(f"#S{d}" for d in s.get("dependencies", [])) or "None"
                a = s.get("agent", "")
                t = s.get("tool", "")
                args_str = json.dumps(s.get("tool_args", {}))
                parts.append(f"Step {s['step']}: [{a}] {t}({args_str})  deps={deps}\n")
                parts.append(f"  Task: {esc(s.get('task',''))}\n")

            parts.append("""</div>
      </div>
    </div>
    <div class="connector">&#8595;</div>
""")

            # Steps 2+: EXECUTE each tool call
            history = r.get("history", [])
            history_by_step = {h["step"]: h for h in history}
            plan = r.get("plan", [])
            llm_call_num = 1
            llm_call_idx = 1  # index into llm_calls_data (0 = plan, 1+ = arg/summarize)

            for s in plan:
                step_num = s["step"]
                h = history_by_step.get(step_num, {})
                success = h.get("success", False)
                agent = s.get("agent", "Unknown")
                ac, _ = AGENT_COLORS.get(agent, ("#6b7280", "#374151"))
                tool = s.get("tool", "")
                orig_args = s.get("tool_args", {})
                resolved_args = h.get("tool_args", orig_args)
                has_placeholder = any(isinstance(v, str) and "{step_" in v for v in orig_args.values())

                # If args had placeholders, show the LLM arg resolution call
                if has_placeholder:
                    llm_call_num += 1
                    arg_call_data = llm_calls_data[llm_call_idx] if llm_call_idx < len(llm_calls_data) else {}
                    llm_call_idx += 1
                    parts.append(f"""
    <div class="call-block llm-call">
      <div class="call-header">
        <span class="call-type llm">LLM CALL #{llm_call_num}</span>
        <span class="call-step">ARG RESOLVE (Step {step_num})</span>
        {tok_badge(arg_call_data)}
      </div>
      <div class="call-desc">Resolve <code>{esc(json.dumps(orig_args))}</code> using prior step results</div>
      <div class="call-io">
        <div class="io-label">Resolved args</div>
        <div class="io-content io-output">{esc(json.dumps(resolved_args, indent=2))}</div>
      </div>
    </div>
    <div class="connector">&#8595;</div>
""")

                # Tool call
                status_cls = "success" if success else "failure"
                resp = truncate(fmt_json(h.get("response", "")), 1200)
                error = h.get("error", "")

                parts.append(f"""
    <div class="call-block tool-call {status_cls}">
      <div class="call-header">
        <span class="call-type mcp">MCP TOOL</span>
        <span class="agent-badge" style="background:{ac}">{esc(agent)}</span>
        <span class="tool-name">{esc(tool)}</span>
        <span style="margin-left:auto;font-size:0.75rem;color:{'var(--green)' if success else 'var(--red)'}">{'SUCCESS' if success else 'FAILED'}</span>
      </div>
      <div class="call-desc">Step {step_num}: {esc(s.get('task', ''))}</div>
      <div class="call-io">
        <div class="io-label">Arguments</div>
        <div class="io-content io-input">{esc(json.dumps(resolved_args, indent=2))}</div>
      </div>
      <div class="call-io">
        <div class="io-label">{'Response' if success else 'Error'}</div>
        <div class="io-content {'io-output' if success else 'io-error'}">{esc(resp if success else error)}</div>
      </div>
    </div>
    <div class="connector">&#8595;</div>
""")

            # Final: SUMMARIZE call
            llm_call_num += 1
            summarize_data = llm_calls_data[-1] if llm_calls_data and len(llm_calls_data) >= 2 else {}
            parts.append(f"""
    <div class="call-block llm-call">
      <div class="call-header">
        <span class="call-type llm">LLM CALL #{llm_call_num}</span>
        <span class="call-step">SUMMARIZE</span>
        {tok_badge(summarize_data)}
      </div>
      <div class="call-desc">Gemma 4 synthesizes tool results into a final answer.</div>
    </div>
""")
            parts.append('</div>')  # end pipeline

            # === ANSWER ===
            answer = r.get("answer", "")
            if answer:
                parts.append(f"""
    <div class="answer-block">
      <div class="answer-label">Final Answer</div>
      <div class="answer-text">{esc(truncate(answer, 1500))}</div>
    </div>
""")

            # === SIDE-BY-SIDE PLAN COMPARISON ===
            gold_text = r.get("gold_response", "")
            model_text = ""
            for s in r.get("plan", []):
                deps = ", ".join(f"#S{d}" for d in s.get("dependencies", [])) or "None"
                model_text += f"#Task{s['step']}: {s.get('task','')}\n"
                model_text += f"#Agent{s['step']}: {s.get('agent','')}\n"
                model_text += f"#Tool{s['step']}: {s.get('tool','')}\n"
                model_text += f"#Args{s['step']}: {json.dumps(s.get('tool_args',{}))}\n"
                model_text += f"#Dependency{s['step']}: {deps}\n\n"

            score_color = "var(--green)" if score >= 0.8 else ("var(--amber)" if score >= 0.5 else "var(--red)")

            # Score summary bar
            parts.append(f"""
    <div style="margin-top:20px;padding:14px 18px;background:var(--bg-3);border:1px solid var(--border);border-radius:8px;display:flex;gap:24px;flex-wrap:wrap;align-items:center">
      <div>
        <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted)">Agent-Tool Match</span>
        <span style="font-size:1.2rem;font-weight:700;color:{score_color};margin-left:8px">{score*100:.0f}%</span>
      </div>
      <div style="font-size:0.82rem;color:var(--text-dim)">Steps: {plan_steps} vs {len(gold_steps)} ref</div>
      <div style="font-size:0.82rem;color:{'var(--green)' if is_success else 'var(--red)'}">{'All steps passed' if is_success else 'Some steps failed'}</div>
      <div style="font-size:0.82rem;color:var(--text-dim)">{r.get('_llm_calls', '?')} LLM calls</div>
      <div style="font-size:0.82rem;color:var(--text-dim)">{tok_summary.get('total', 0):,} tokens</div>
      <div style="font-size:0.82rem;color:var(--red)">{tok_summary.get('tool_desc_tokens_est', 0):,} tool desc overhead</div>
      <div style="font-size:0.82rem;color:var(--text-dim)">{r.get('elapsed_seconds', 0):.1f}s</div>
      {''.join(f'<div style="font-size:0.78rem;color:var(--amber);background:var(--bg-0);padding:2px 8px;border-radius:4px">{esc(n)}</div>' for n in notes)}
    </div>
""")

            # Side-by-side plans
            parts.append(f"""
    <div class="eval-panel" style="margin-top:12px">
      <div class="eval-section eval-gold">
        <div class="eval-label">Reference Plan (Gemini 2.5 Flash) — {len(gold_steps)} steps</div>
        <div class="eval-plan">{esc(gold_text or 'N/A')}</div>
        <div style="margin-top:8px;font-size:0.72rem;color:var(--text-muted);font-style:italic">
          Note: AssetOpsBench does not provide structured gold plans. This reference was generated by Gemini 2.5 Flash as our baseline.
        </div>
      </div>
      <div class="eval-section" style="background:#120a20;border:1px solid #3b2068">
        <div class="eval-label" style="color:var(--purple)">Gemma 4 Plan — {plan_steps} steps</div>
        <div class="eval-plan">{esc(model_text or 'N/A')}</div>
      </div>
    </div>
""")

        parts.append("  </div>\n</div>\n")

    parts.append("""
</div>
<script>
// Keyboard nav: press 'n' for next, 'p' for previous
document.addEventListener('keydown', e => {
    const scenarios = [...document.querySelectorAll('.scenario')];
    const openIdx = scenarios.findIndex(s => s.classList.contains('open'));
    if (e.key === 'n' || e.key === 'ArrowDown') {
        if (openIdx >= 0) scenarios[openIdx].classList.remove('open');
        const next = Math.min(openIdx + 1, scenarios.length - 1);
        scenarios[next].classList.add('open');
        scenarios[next].scrollIntoView({behavior: 'smooth', block: 'start'});
    } else if (e.key === 'p' || e.key === 'ArrowUp') {
        if (openIdx >= 0) scenarios[openIdx].classList.remove('open');
        const prev = Math.max(openIdx - 1, 0);
        scenarios[prev].classList.add('open');
        scenarios[prev].scrollIntoView({behavior: 'smooth', block: 'start'});
    }
});
</script>
</body>
</html>
""")

    return "".join(parts)


def main():
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print(f"Building dashboard from {len(results)} results...")
    with open(OUTPUT_FILE, "w") as f:
        f.write(build_html(results))
    ran = [r for r in results if r.get("plan_steps", 0) > 0]
    ok = [r for r in ran if r.get("all_succeeded")]
    print(f"Dashboard: {OUTPUT_FILE}")
    print(f"  Ran: {len(ran)}/{len(results)} | Success: {len(ok)}/{len(ran)}")


if __name__ == "__main__":
    main()
