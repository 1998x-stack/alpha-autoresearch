# -*- coding: utf-8 -*-
"""Autonomous experiment loop runner for alpha_autoresearch.
Runs N iterations of: modify factors.py -> evaluate -> log to results.tsv -> keep/discard.
"""
import sys, os, json, subprocess, time, hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
FACTORS_FILE = ROOT / "factors.py"
RESULTS_FILE = ROOT / "results.tsv"
PROGRESS_FILE = ROOT / "progress.log"
PARETO_FILE = ROOT / "pareto_frontier.json"
MAX_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 30

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(PROGRESS_FILE, "a") as f:
        f.write(line + "\n")

def get_git_hash():
    r = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip()[:7]

def run_eval():
    r = subprocess.run(["uv", "run", "python", "prepare.py"], capture_output=True, text=True, cwd=ROOT)
    return r.stdout, r.stderr

def parse_results(output):
    results = []
    current = None
    for line in output.split("\n"):
        line = line.strip()
        if line == "---":
            if current:
                results.append(current)
            current = {}
        elif line.startswith("factor:"):
            current["factor_name"] = line.split(":", 1)[1].strip()
        elif line.startswith("rank_ic:"):
            current["rank_ic"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("ic_ir:"):
            current["ic_ir"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("turnover_stability:"):
            current["turnover"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("status:"):
            current["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("dominates:"):
            current["dominates"] = line.split(":", 1)[1].strip()
        elif line.startswith("dominated_by:"):
            current["dominated_by"] = line.split(":", 1)[1].strip()
    if current:
        results.append(current)
    return results

def append_tsv(commit, results):
    with open(RESULTS_FILE, "a") as f:
        for r in results:
            f.write(f"{commit}\t{r.get('factor_name','?')}\t{r.get('rank_ic',0):.4f}\t"
                    f"{r.get('ic_ir',0):.2f}\t{r.get('turnover',0):.4f}\t"
                    f"{r.get('dominates','(none)')}\t{r.get('dominated_by','(none)')}\t"
                    f"{r.get('status','?')}\tgenerated\n")

def count_frontier():
    if PARETO_FILE.exists():
        with open(PARETO_FILE) as f:
            data = json.load(f)
            return len(data.get("frontier", []))
    return 0

# Factor generation ideas — exploring different Alpha101 categories
FACTOR_IDEAS = [
    # Momentum variants
    ('momentum_v1', 'ops.cs_rank(m["close"] - ops.delay(m["close"], 5))'),
    ('momentum_v2', 'ops.cs_rank(m["close"] - ops.delay(m["close"], 10))'),
    ('momentum_v3', 'ops.cs_rank(m["close"] - ops.delay(m["close"], 20))'),
    ('momentum_long', 'ops.cs_rank(m["close"] - ops.delay(m["close"], 60))'),
    ('momentum_decay', 'ops.decay_linear(m["close"] - ops.delay(m["close"], 1), 5)'),
    # Reversal
    ('reversal_5d', '-ops.cs_rank(m["close"] - ops.delay(m["close"], 5))'),
    ('reversal_10d', '-ops.cs_rank(m["close"] - ops.delay(m["close"], 10))'),
    # Volume-price correlation
    ('vol_corr', 'ops.rolling_corr(m["close"], m["volume"], 10)'),
    ('vol_corr_20', 'ops.rolling_corr(m["close"], m["volume"], 20)'),
    ('vol_rank_corr', 'ops.rolling_corr(ops.cs_rank(m["open"]), ops.cs_rank(m["volume"]), 10)'),
    # Volatility
    ('volatility', 'ops.rolling_std(m["close"], 20)'),
    ('volume_vol', 'ops.rolling_std(m["volume"], 20)'),
    # Price range
    ('hl_range', '(m["high"] - m["low"]) / m["close"]'),
    ('open_hl', '(m["close"] - m["open"]) / (m["high"] - m["low"] + 0.001)'),
    # VWAP
    ('vwap_diff', 'm["close"] - m["vwap"]'),
    ('vwap_sq', '-(m["vwap"] - ops.rolling_min(m["vwap"], 14)) ** 2'),
    # Volume ratio
    ('vol_adv20', 'm["volume"] / m["adv20"]'),
    ('vol_adv5_20', 'ops.cs_rank(m["volume"] / m["adv5"]) * ops.cs_rank(m["volume"] / m["adv20"])'),
    # Returns
    ('ret_mean', 'ops.rolling_sum(m["returns"], 20) / 20'),
    ('ret_vol', 'ops.rolling_std(m["returns"], 20)'),
    # Combo
    ('momentum_vol', 'ops.cs_rank(m["close"] - ops.delay(m["close"], 5)) * ops.cs_rank(m["volume"])'),
    ('momentum_turn', '-ops.cs_rank(m["close"] - ops.delay(m["close"], 5)) * ops.cs_rank(m["volume"] / m["adv20"])'),
    # Classic Alpha101
    ('alpha003_style', '-ops.rolling_corr(ops.cs_rank(m["open"]), ops.cs_rank(m["volume"]), 10)'),
    ('alpha006_style', '-ops.rolling_corr(m["open"], m["volume"], 10)'),
    ('alpha054_style', '-((m["low"] - m["close"]) * (m["open"] ** 2)) / ((m["low"] - m["high"]) * (m["close"] ** 2) + 0.001)'),
    ('alpha101_style', '(m["close"] - m["open"]) / (m["high"] - m["low"] + 0.001)'),
    # TS rank
    ('ts_rank_mom', 'ops.ts_rank(ops.delta(m["close"], 5), 10)'),
    ('ts_rank_vol', '-ops.ts_rank(m["volume"], 5)'),
    # Decay linear
    ('decay_mom', 'ops.decay_linear(ops.delta(m["close"], 1), 5)'),
    ('decay_range', 'ops.decay_linear((m["high"] - m["low"]) / m["close"], 10)'),
]

def generate_factors(iteration):
    """Generate factor variants based on iteration number."""
    idea_idx = iteration % len(FACTOR_IDEAS)
    name, formula = FACTOR_IDEAS[idea_idx]
    
    code = """# factors.py — Agent edit surface for alpha_autoresearch
# Each Factor* class is auto-discovered and evaluated.
# Available operators: ops.ts_rank, ops.rolling_corr, ops.rolling_cov,
#   ops.cs_rank, ops.delta, ops.delay, ops.decay_linear, ops.rolling_std,
#   ops.rolling_min, ops.rolling_max, ops.rolling_sum, ops.cs_zscore
# Available columns: open, high, low, close, volume, vwap, returns,
#   adv5, adv10, adv20, adv30, adv40, adv60, adv120, adv150, adv180

from prepare import Factor, ops


"""
    code += f"""class Factor001(Factor):
    name = "{name}"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        val = {formula}
        return Factor.as_cs_series(df, val)
"""
    
    # Add a bonus factor every 3rd iteration
    if iteration % 3 == 0 and iteration > 0:
        bonus_idx = (iteration * 7) % len(FACTOR_IDEAS)
        bname, bformula = FACTOR_IDEAS[bonus_idx]
        if bname != name:
            code += f"""

class Factor002(Factor):
    name = "{bname}_v2"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        val = {bformula}
        return Factor.as_cs_series(df, val)
"""
    
    FACTORS_FILE.write_text(code)

# Main loop
log("=" * 60)
log(f"STARTING AUTONOMOUS EXPERIMENT LOOP — max iterations: {MAX_ITER}")
log(f"Initial frontier size: {count_frontier()}")
log("=" * 60)

best_ic = -999
best_ir = -999
total_keeps = 0
total_discards = 0
total_crashes = 0

for iteration in range(MAX_ITER):
    log(f"\n--- Iteration {iteration + 1}/{MAX_ITER} ---")
    
    generate_factors(iteration)
    
    # Git commit
    subprocess.run(["git", "add", "factors.py"], capture_output=True, cwd=ROOT)
    commit_msg = f"exp: iteration {iteration + 1} — factor generation"
    subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, cwd=ROOT)
    
    # Run evaluation
    t0 = time.time()
    stdout, stderr = run_eval()
    elapsed = time.time() - t0
    
    if stderr.strip():
        log(f"  WARN: {stderr.strip()[:200]}")
    
    results = parse_results(stdout)
    commit = get_git_hash()
    
    if not results:
        log(f"  No results parsed! stdout preview: {stdout[:200]}")
        continue
    
    # Log results
    append_tsv(commit, results)
    
    for r in results:
        status = r.get("status", "?")
        name = r.get("factor_name", "?")
        ic = r.get("rank_ic", 0)
        ir = r.get("ic_ir", 0)
        to = r.get("turnover", 0)
        
        if status == "keep":
            total_keeps += 1
            if abs(ic) > best_ic:
                best_ic = abs(ic)
            if abs(ir) > best_ir:
                best_ir = abs(ir)
            log(f"  KEEP  | {name:25s} | IC={ic:+.4f} IR={ir:+.2f} TO={to:.4f} | frontier={count_frontier()}")
        elif status == "discard":
            total_discards += 1
            log(f"  DISC  | {name:25s} | IC={ic:+.4f} IR={ir:+.2f} TO={to:.4f}")
        elif status == "crash":
            total_crashes += 1
            log(f"  CRASH | {name:25s}")
    
    log(f"  Elapsed: {elapsed:.1f}s | Keeps: {total_keeps} | Discards: {total_discards} | Crashes: {total_crashes}")

log("\n" + "=" * 60)
log(f"LOOP COMPLETE — {MAX_ITER} iterations")
log(f"Total: {total_keeps} keeps, {total_discards} discards, {total_crashes} crashes")
log(f"Best IC: {best_ic:.4f} | Best IR: {best_ir:.2f}")
log(f"Final frontier size: {count_frontier()}")
log("=" * 60)
