"""
Auto-tuner for Cage Walk animation quality.

Usage:
    python auto_tune.py                                  # Tune across default animations
    python auto_tune.py "Capoeira wild"                  # Tune on single animation
    python auto_tune.py --resume best_config.json        # Resume from saved config
    python auto_tune.py --sweeps 3 --values 5            # Custom sweep/value counts
    python auto_tune.py --headless                       # Run Chrome headless
    python auto_tune.py --param KBAND_LO                 # Tune only one param
    python auto_tune.py --animations "Walking,Capoeira Idle,Capoeira wild"

Runs coordinate descent over 20 animation parameters, optimizing a weighted
loss function computed from diagnostic metrics. Evaluates across multiple
animations to prevent overfitting.
"""

import sys
import os
import re
import json
import csv
import time
import argparse
import urllib.parse
import numpy as np
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from auto_test import parse_metrics, format_summary, extract_msg, start_server

# ---------------------------------------------------------------------------
# Parameter definitions: (key, default, min, max, type)
# ---------------------------------------------------------------------------
PARAMS = [
    # Blend bands
    ("KBAND_LO",               0.70,   0.50,  1.00,  float),
    ("KBAND_HI",               1.30,   1.00,  1.50,  float),
    ("HBAND_FRAC",             0.35,   0.10,  0.50,  float),
    ("SBAND_SPINE",            0.05,   0.01,  0.08,  float),
    ("ABAND_FRAC",             0.20,   0.05,  0.30,  float),
    ("EBAND_FRAC",             0.18,   0.05,  0.25,  float),
    # PBD stretch
    ("MAX_STRETCH",            1.50,   1.10,  2.00,  float),
    ("MAX_STRETCH_SHORT",      1.20,   1.05,  1.50,  float),
    ("PBD_ITERATIONS",         12,     4,     24,    int),
    ("MAX_CB_STRETCH",         1.30,   1.05,  1.80,  float),
    # Smoothing
    ("CAGE_SMOOTHING_VAL",     0.25,   0.05,  0.50,  float),
    ("SHIN_SMOOTH",            0.92,   0.50,  1.00,  float),
    ("FOOT_SMOOTH",            0.65,   0.30,  1.00,  float),
    ("ANKLE_ANGLE_SMOOTH_VAL", 0.80,   0.30,  1.00,  float),
    # Jitter damping
    ("ACCEL_THRESH_VAL",       0.028,  0.005, 0.050, float),
    ("MAX_DAMP_VAL",           0.82,   0.30,  0.95,  float),
    # Contact
    ("PLANT_STRENGTH",         0.60,   0.00,  1.00,  float),
    ("ENTER_THRESH_FRAC",      0.03,   0.01,  0.08,  float),
    ("EXIT_THRESH_FRAC",       0.08,   0.03,  0.15,  float),
    ("FOOT_ENTER_FRAMES_VAL",  5,      1,     8,     int),
]

PARAM_KEYS = [p[0] for p in PARAMS]
PARAM_DEFAULTS = {p[0]: p[1] for p in PARAMS}
PARAM_RANGES = {p[0]: (p[2], p[3]) for p in PARAMS}
PARAM_TYPES = {p[0]: p[4] for p in PARAMS}

# Default animation set for multi-animation evaluation
DEFAULT_ANIMATIONS = [
    "Walking",
    "Capoeira Idle",
    "Capoeira wild",
    "Capoeira High Kick",
    "Running Left Arc In A Sad Disposition",
]

# Catastrophe thresholds — abort evaluation early if any exceeded
CATASTROPHE = {
    'mf_worst_stretch': 50.0,     # 50x stretch = completely broken
    'inverted_pct': 50.0,         # 50% inverted triangles
    'mf_avg_jitter_mm': 500.0,    # 500mm/f^2 jitter
}

# Large penalty for structural failures or catastrophes
PENALTY_LOSS = 9999.0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CAGE_WALK_DIR = Path(__file__).parent.resolve()
MIXAMO_DIR = CAGE_WALK_DIR.parent / "Mixamo"
PORT = 8765
MAX_WAIT = 240  # max seconds to wait for multi-frame diagnostic


# ---------------------------------------------------------------------------
# Structural preflight checks
# ---------------------------------------------------------------------------
def parse_structural_status(logs):
    """Parse structural gate logs from console output.
    Returns dict with gate results, or None if logs not found.
    """
    status = {
        'cfg_dump_confirmed': False,
        'cfg_dump_data': None,
        'override_p_from_fbx': False,
        'euler_fallback': False,
        'position_mode': False,
        'spine_split': {},
        'transform_missing': [],
    }

    for entry in logs:
        msg = entry.get("message", "")
        txt = extract_msg(msg)

        # CFG_DUMP confirmation
        if 'CFG_DUMP:' in txt:
            status['cfg_dump_confirmed'] = True
            try:
                json_str = txt.split('CFG_DUMP:', 1)[1].strip()
                status['cfg_dump_data'] = json.loads(json_str)
            except Exception:
                pass

        # [CONFIG] Loaded from URL
        if '[CONFIG] Loaded from URL:' in txt:
            status['cfg_dump_confirmed'] = True

        # STRUCTURAL_STATUS line
        if 'STRUCTURAL_STATUS:' in txt:
            m = re.search(r'overridePFromFBX=(\w+)', txt)
            if m:
                status['override_p_from_fbx'] = m.group(1) == 'true'
            m = re.search(r'eulerFallback=(\w+)', txt)
            if m:
                status['euler_fallback'] = m.group(1) == 'true'
            m = re.search(r'positionMode=(\w+)', txt)
            if m:
                status['position_mode'] = m.group(1) == 'true'

        # [SPINE SPLIT] pelvis=X spine=X spine1=X spine2=X chest=X
        if '[SPINE SPLIT]' in txt:
            for key in ['pelvis', 'spine', 'spine1', 'spine2', 'chest']:
                m = re.search(rf'{key}=(\d+)', txt)
                if m:
                    status['spine_split'][key] = int(m.group(1))

        # REGION_TRANSFORM_MISSING: [list]
        if 'REGION_TRANSFORM_MISSING:' in txt:
            m = re.search(r'REGION_TRANSFORM_MISSING: \[([^\]]*)\]', txt)
            if m:
                content = m.group(1).strip()
                if content:
                    status['transform_missing'] = [x.strip() for x in content.split(',')]
                else:
                    status['transform_missing'] = []

    return status


def check_preflight(status, config):
    """Check structural preflight gates. Returns (ok, reasons) tuple."""
    reasons = []

    # Gate 1: Config injection confirmed (only if config has overrides)
    non_default = {k: v for k, v in config.items() if v != PARAM_DEFAULTS.get(k)}
    if non_default and not status['cfg_dump_confirmed']:
        reasons.append("CONFIG injection not confirmed in logs")

    # Gate 2: overridePFromFBX must have run
    if not status['override_p_from_fbx']:
        reasons.append("overridePFromFBX did not run (no FBX skeleton data)")

    # Gate 3: Euler fallback must NOT be active
    if status['euler_fallback']:
        reasons.append("Euler fallback active (position mode should be used)")

    # Gate 4: Spine subregions must have vertices
    spine = status['spine_split']
    if spine:
        for key in ['spine1', 'spine2', 'chest']:
            if spine.get(key, 0) == 0:
                reasons.append(f"Spine subregion '{key}' has 0 verts")
    else:
        reasons.append("No [SPINE SPLIT] log found")

    # Gate 5: No regions with verts but missing transforms
    if status['transform_missing']:
        reasons.append(f"Regions with verts but no transform: {status['transform_missing']}")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
def max_seam_gap(metrics):
    """Return the maximum avg seam gap (mm) across all boundary seams."""
    seam_gaps = metrics.get('seam_gaps', {})
    if not seam_gaps:
        return 50.0  # pessimistic default
    return max(sg['avg_mm'] for sg in seam_gaps.values())


def max_contact_drift(metrics):
    """Return the maximum avg contact drift (mm) across all limbs."""
    cd = metrics.get('contact_drift', {})
    if not cd:
        return 50.0  # pessimistic default
    drifts = [v['avg_drift'] for v in cd.values() if v['locked_frames'] > 0]
    return max(drifts) if drifts else 0.0


def check_catastrophe(metrics):
    """Check for catastrophic metrics. Returns list of triggered thresholds."""
    triggered = []
    for key, thresh in CATASTROPHE.items():
        val = metrics.get(key)
        if val is not None and val > thresh:
            triggered.append(f"{key}={val:.1f} > {thresh}")
    # Check for NaN in key metrics
    for key in ['mf_worst_stretch', 'mf_avg_strain_pct', 'inverted_pct']:
        val = metrics.get(key)
        if val is not None and (val != val):  # NaN check
            triggered.append(f"{key} is NaN")
    return triggered


def compute_loss(metrics):
    """Weighted loss from diagnostic metrics. Lower = better."""
    stretch = metrics.get('mf_worst_stretch', 10.0)
    strain = metrics.get('mf_avg_strain_pct', 50.0)
    jitter = metrics.get('mf_avg_jitter_mm', 100.0)
    invert = metrics.get('inverted_pct', 10.0)
    seam = max_seam_gap(metrics)
    contact = max_contact_drift(metrics)
    shape = metrics.get('shape_fidelity_avg', 90.0)

    loss = (
        2.0 * stretch   +  # Stretch heavily weighted (24.8x is catastrophic)
        1.0 * strain    +  # Avg strain percentage
        0.5 * jitter    +  # Jitter in mm/f^2
        3.0 * invert    +  # Inverted tris percentage (very visible)
        0.1 * seam      +  # Seam gap in mm
        0.3 * contact   +  # Contact drift in mm (raised from 0.05 per feedback)
        0.5 * shape        # Shape fidelity in degrees
    )

    breakdown = {
        'stretch': stretch, 'strain': strain, 'jitter': jitter,
        'invert': invert, 'seam': seam, 'contact': contact, 'shape': shape,
    }
    return loss, breakdown


def aggregate_losses(per_anim_losses):
    """Combine per-animation losses: 0.7*mean + 0.3*max (prevents sacrificing one anim)."""
    if not per_anim_losses:
        return PENALTY_LOSS
    mean_loss = sum(per_anim_losses) / len(per_anim_losses)
    max_loss = max(per_anim_losses)
    return 0.7 * mean_loss + 0.3 * max_loss


# ---------------------------------------------------------------------------
# Candidate value generation
# ---------------------------------------------------------------------------
def generate_candidates(key, n_values):
    """Generate n_values candidate values for a parameter, linearly spaced."""
    lo, hi = PARAM_RANGES[key]
    typ = PARAM_TYPES[key]
    if typ == int:
        vals = sorted(set(int(round(v)) for v in np.linspace(lo, hi, n_values)))
        if len(vals) < n_values:
            all_ints = list(range(lo, hi + 1))
            vals = sorted(set(vals + all_ints))[:n_values]
        return vals
    else:
        return list(np.linspace(lo, hi, n_values))


# ---------------------------------------------------------------------------
# Browser interaction
# ---------------------------------------------------------------------------
def run_one_evaluation(driver, fbx_path, config, port=PORT):
    """
    Reload page with config in URL hash, load FBX, capture diagnostics.
    Returns (metrics_dict, raw_logs, structural_status).
    """
    # 1. Build URL with config in hash fragment
    config_json = json.dumps(config)
    encoded = urllib.parse.quote(config_json)
    url = f"http://localhost:{port}/index.html#{encoded}"

    # 2. Navigate (forces full page reload)
    driver.get(url)

    # 3. Wait for mesh to load (status shows "Ready" or "Drop")
    try:
        WebDriverWait(driver, 60).until(
            lambda d: any(kw in d.find_element(By.ID, "status").text.lower()
                         for kw in ["ready", "drop"])
        )
    except Exception:
        return None, [], None

    time.sleep(0.5)

    # 4. Set plant strength slider if in config
    if 'PLANT_STRENGTH' in config:
        driver.execute_script(
            f"document.getElementById('plant').value = {config['PLANT_STRENGTH']};"
            f"document.getElementById('plant').dispatchEvent(new Event('input'));"
        )

    # 5. Enable diagnostics
    for cb_id in ["showStrain", "showBoneError"]:
        try:
            cb = driver.find_element(By.ID, cb_id)
            if not cb.is_selected():
                cb.click()
        except Exception:
            pass

    time.sleep(0.3)

    # 6. Load FBX
    try:
        driver.find_element(By.ID, "fbxFile").send_keys(str(fbx_path))
    except Exception:
        return None, [], None

    time.sleep(2)

    # 7. Poll for multi-frame diagnostic
    mf_found = False
    elapsed = 0
    all_logs = []
    while elapsed < MAX_WAIT:
        time.sleep(2)
        elapsed += 2
        try:
            new_logs = driver.get_log("browser")
        except Exception:
            new_logs = []
        all_logs.extend(new_logs)

        # Early catastrophe check: parse metrics periodically to fail fast
        if elapsed >= 30 and elapsed % 10 == 0 and not mf_found:
            partial_metrics = parse_metrics(all_logs)
            cats = check_catastrophe(partial_metrics)
            if cats:
                print(f"  CATASTROPHE detected at {elapsed}s: {cats}")
                return partial_metrics, all_logs, parse_structural_status(all_logs)

        for entry in new_logs:
            if "MULTI-FRAME DIAGNOSTIC SUMMARY" in entry.get("message", ""):
                mf_found = True
                break
        if mf_found:
            time.sleep(1)
            try:
                all_logs.extend(driver.get_log("browser"))
            except Exception:
                pass
            break

    # 8. Parse metrics and structural status
    metrics = parse_metrics(all_logs)
    structural = parse_structural_status(all_logs)

    return metrics, all_logs, structural


def evaluate_config(driver, fbx_paths, config, port=PORT):
    """
    Evaluate a config across multiple animations.
    Returns (aggregate_loss, per_anim_losses, per_anim_breakdowns, preflight_ok).
    """
    per_anim_losses = []
    per_anim_breakdowns = []
    preflight_ok = True

    for fbx_path in fbx_paths:
        metrics, logs, structural = run_one_evaluation(driver, fbx_path, config, port)

        if metrics is None:
            per_anim_losses.append(PENALTY_LOSS)
            per_anim_breakdowns.append(None)
            preflight_ok = False
            continue

        # Structural preflight
        if structural:
            ok, reasons = check_preflight(structural, config)
            if not ok:
                # Only warn on first failure (don't spam)
                if preflight_ok:
                    for r in reasons:
                        print(f"    [preflight] {r}")
                preflight_ok = False
                per_anim_losses.append(PENALTY_LOSS)
                per_anim_breakdowns.append(None)
                continue

        # Catastrophe check
        cats = check_catastrophe(metrics)
        if cats:
            per_anim_losses.append(PENALTY_LOSS)
            per_anim_breakdowns.append(None)
            continue

        loss, breakdown = compute_loss(metrics)
        per_anim_losses.append(loss)
        per_anim_breakdowns.append(breakdown)

    agg_loss = aggregate_losses(per_anim_losses)
    return agg_loss, per_anim_losses, per_anim_breakdowns, preflight_ok


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------
def init_csv(csv_path):
    """Create the tuning log CSV with headers."""
    headers = [
        'iteration', 'sweep', 'param_name', 'param_value',
        'agg_loss', 'per_anim_losses',
        'stretch', 'strain', 'jitter', 'invert',
        'seam', 'contact', 'shape',
        'preflight_ok', 'improved',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_csv(csv_path, row):
    """Append a row to the tuning log CSV."""
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def avg_breakdown(breakdowns):
    """Average the per-animation breakdowns (skipping None entries)."""
    valid = [b for b in breakdowns if b is not None]
    if not valid:
        return {'stretch': 0, 'strain': 0, 'jitter': 0, 'invert': 0,
                'seam': 0, 'contact': 0, 'shape': 0}
    keys = valid[0].keys()
    return {k: sum(b[k] for b in valid) / len(valid) for k in keys}


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Auto-tune Cage Walk animation parameters")
    parser.add_argument('fbx_name', nargs='?', default=None,
                        help='Single FBX name (without .fbx). Overrides --animations.')
    parser.add_argument('--animations', type=str, default=None,
                        help='Comma-separated list of FBX names to tune across')
    parser.add_argument('--sweeps', type=int, default=3,
                        help='Number of coordinate descent sweeps (default: 3)')
    parser.add_argument('--values', type=int, default=5,
                        help='Number of candidate values per parameter (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to best_config.json to resume from')
    parser.add_argument('--headless', action='store_true',
                        help='Run Chrome headless (no visible window)')
    parser.add_argument('--param', type=str, default=None,
                        help='Tune only this single parameter (e.g., KBAND_LO)')
    args = parser.parse_args()

    # Resolve FBX paths
    if args.fbx_name:
        anim_names = [args.fbx_name]
    elif args.animations:
        anim_names = [n.strip() for n in args.animations.split(',')]
    else:
        anim_names = list(DEFAULT_ANIMATIONS)

    fbx_paths = []
    for name in anim_names:
        if not name.endswith('.fbx'):
            name += '.fbx'
        path = MIXAMO_DIR / name
        if not path.exists():
            print(f"[warn] FBX not found: {path} — skipping")
            continue
        fbx_paths.append(path)

    if not fbx_paths:
        print(f"[error] No valid FBX files found in {MIXAMO_DIR}")
        print(f"[info] Available:")
        for f in sorted(MIXAMO_DIR.glob("*.fbx")):
            print(f"  {f.name}")
        sys.exit(1)

    # Initialize config
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = CAGE_WALK_DIR / resume_path
        with open(resume_path, 'r') as f:
            current_config = json.load(f)
        print(f"[config] Resumed from {resume_path}: {len(current_config)} params")
    else:
        current_config = dict(PARAM_DEFAULTS)

    # Determine which params to tune
    if args.param:
        if args.param not in PARAM_KEYS:
            print(f"[error] Unknown param '{args.param}'. Available: {', '.join(PARAM_KEYS)}")
            sys.exit(1)
        tune_params = [args.param]
    else:
        tune_params = list(PARAM_KEYS)

    n_sweeps = args.sweeps
    n_values = args.values
    n_anims = len(fbx_paths)

    total_evals = len(tune_params) * n_values * n_sweeps * n_anims
    print(f"[config] Animations ({n_anims}): {[p.stem for p in fbx_paths]}")
    print(f"[config] Params to tune: {len(tune_params)}")
    print(f"[config] Sweeps: {n_sweeps}, Values/param: {n_values}")
    print(f"[config] Max evaluations: {total_evals} ({total_evals // n_anims} configs x {n_anims} anims)")
    print()

    # Output paths
    csv_path = CAGE_WALK_DIR / "tuning_log.csv"
    best_json_path = CAGE_WALK_DIR / "best_config.json"
    summary_path = CAGE_WALK_DIR / "tuning_summary.txt"
    init_csv(csv_path)

    # Start server
    httpd = start_server()

    # Set up Chrome
    chrome_options = Options()
    chrome_options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    if args.headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,900")

    print("[browser] Launching Chrome...")
    driver = webdriver.Chrome(options=chrome_options)

    best_config = dict(current_config)
    best_loss = PENALTY_LOSS

    try:
        # Baseline evaluation
        print("=" * 60)
        print("  BASELINE EVALUATION")
        print("=" * 60)
        agg_loss, per_losses, per_breakdowns, pf_ok = evaluate_config(
            driver, fbx_paths, current_config)

        if agg_loss >= PENALTY_LOSS:
            print("[error] Baseline evaluation failed structural preflight")
            print("  Fix structural issues before tuning.")
            return

        best_loss = agg_loss
        best_config = dict(current_config)
        bd = avg_breakdown(per_breakdowns)
        print(f"  Baseline aggregate loss: {best_loss:.2f}")
        print(f"    Per-anim losses: {[f'{l:.1f}' for l in per_losses]}")
        print(f"    Avg: stretch={bd['stretch']:.1f}  strain={bd['strain']:.1f}  "
              f"jitter={bd['jitter']:.1f}  invert={bd['invert']:.1f}")
        print(f"    Avg: seam={bd['seam']:.1f}  contact={bd['contact']:.1f}  "
              f"shape={bd['shape']:.1f}")
        print()

        iteration = 0
        summary_lines = []
        summary_lines.append(f"Auto-Tune Summary")
        summary_lines.append(f"Animations: {[p.stem for p in fbx_paths]}")
        summary_lines.append(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Baseline loss: {best_loss:.2f}")
        summary_lines.append("")

        for sweep in range(n_sweeps):
            print("=" * 60)
            print(f"  SWEEP {sweep + 1} / {n_sweeps}")
            print("=" * 60)
            sweep_improved = False
            sweep_start_loss = best_loss

            for param_key in tune_params:
                candidates = generate_candidates(param_key, n_values)
                param_type = PARAM_TYPES[param_key]
                current_val = best_config.get(param_key, PARAM_DEFAULTS[param_key])

                print(f"\n  [{param_key}] current={current_val}  "
                      f"trying {len(candidates)} values: "
                      f"{[round(v, 4) if param_type == float else v for v in candidates]}")

                param_best_loss = best_loss
                param_best_val = current_val

                for cand_val in candidates:
                    iteration += 1
                    trial_config = dict(best_config)
                    trial_config[param_key] = cand_val

                    print(f"    [{iteration}] {param_key}={cand_val}", end="", flush=True)

                    agg_loss, per_losses, per_breakdowns, pf_ok = evaluate_config(
                        driver, fbx_paths, trial_config)

                    if agg_loss >= PENALTY_LOSS:
                        print(f"  PENALTY (structural/catastrophe)")
                        append_csv(csv_path, [
                            iteration, sweep + 1, param_key, cand_val,
                            'PENALTY', str(per_losses),
                            '', '', '', '', '', '', '',
                            pf_ok, False,
                        ])
                        continue

                    improved = agg_loss < param_best_loss
                    if improved:
                        param_best_loss = agg_loss
                        param_best_val = cand_val

                    marker = " ***" if improved else ""
                    per_str = "/".join(f"{l:.0f}" for l in per_losses)
                    print(f"  agg={agg_loss:.2f} [{per_str}]{marker}")

                    bd = avg_breakdown(per_breakdowns)
                    append_csv(csv_path, [
                        iteration, sweep + 1, param_key, cand_val,
                        f"{agg_loss:.4f}", str([round(l, 2) for l in per_losses]),
                        f"{bd['stretch']:.2f}", f"{bd['strain']:.2f}",
                        f"{bd['jitter']:.2f}", f"{bd['invert']:.2f}",
                        f"{bd['seam']:.2f}", f"{bd['contact']:.2f}",
                        f"{bd['shape']:.2f}",
                        pf_ok, improved,
                    ])

                # Update best if this param improved
                if param_best_loss < best_loss:
                    old_val = best_config.get(param_key, PARAM_DEFAULTS[param_key])
                    best_config[param_key] = param_best_val
                    best_loss = param_best_loss
                    sweep_improved = True
                    improvement = f"{param_key}: {old_val} -> {param_best_val} (loss {best_loss:.2f})"
                    print(f"  >> IMPROVED: {improvement}")
                    summary_lines.append(f"  Sweep {sweep+1}: {improvement}")

                    with open(best_json_path, 'w') as f:
                        json.dump(best_config, f, indent=2)
                else:
                    print(f"  >> No improvement for {param_key} (keeping {current_val})")

            sweep_delta = sweep_start_loss - best_loss
            print(f"\n  Sweep {sweep + 1} complete: loss {sweep_start_loss:.2f} -> {best_loss:.2f} "
                  f"(delta={sweep_delta:.2f})")
            summary_lines.append(f"Sweep {sweep+1} total: {sweep_start_loss:.2f} -> {best_loss:.2f} "
                                 f"(delta={sweep_delta:.2f})")
            summary_lines.append("")

            if not sweep_improved:
                print(f"\n  No parameters improved in sweep {sweep + 1} -- converged.")
                summary_lines.append(f"Converged after sweep {sweep+1}")
                break

        # Final evaluation with best config
        print("\n" + "=" * 60)
        print("  FINAL EVALUATION WITH BEST CONFIG")
        print("=" * 60)
        for fbx_path in fbx_paths:
            metrics, logs, structural = run_one_evaluation(driver, fbx_path, best_config)
            if metrics:
                loss, breakdown = compute_loss(metrics)
                print(f"  {fbx_path.stem}: loss={loss:.2f}  "
                      f"stretch={breakdown['stretch']:.1f}  invert={breakdown['invert']:.1f}  "
                      f"jitter={breakdown['jitter']:.1f}")

                diag_summary = format_summary(metrics, fbx_path.name)
                diag_path = CAGE_WALK_DIR / f"diagnostic_summary_tuned_{fbx_path.stem}.txt"
                with open(diag_path, 'w', encoding='utf-8') as f:
                    f.write(diag_summary)

        # Save outputs
        with open(best_json_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\n  Best config -> {best_json_path}")

        summary_lines.append(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Total iterations: {iteration}")
        summary_lines.append(f"Final loss: {best_loss:.2f}")
        summary_lines.append("")
        summary_lines.append("Best config:")
        for key in PARAM_KEYS:
            default = PARAM_DEFAULTS[key]
            val = best_config.get(key, default)
            changed = " *" if val != default else ""
            summary_lines.append(f"  {key:30s} = {val}{changed}")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        print(f"  Tuning summary -> {summary_path}")
        print(f"  Tuning log -> {csv_path}")

        print("\n" + "=" * 60)
        print("  DONE")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n[interrupted] Saving current best config...")
        with open(best_json_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"  Saved -> {best_json_path}")

    except Exception as e:
        print(f"\n[error] {e}")
        import traceback
        traceback.print_exc()
        try:
            with open(best_json_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            print(f"  Saved partial best -> {best_json_path}")
        except Exception:
            pass

    finally:
        print("\n[browser] Closing Chrome...")
        driver.quit()
        httpd.shutdown()
        print("[done] Finished.")


if __name__ == "__main__":
    main()
