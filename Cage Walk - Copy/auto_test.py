"""
Automated diagnostic capture for Cage Walk animation.

Usage:
    python auto_test.py [fbx_name]

Examples:
    python auto_test.py                          # Uses "Capoeira Idle.fbx"
    python auto_test.py "Capoeira High Kick"     # Uses "Capoeira High Kick.fbx"
    python auto_test.py "Walking Forward"         # Uses "Walking Forward.fbx"

Serves the page, loads the FBX, enables strain + bone-vs-cage diagnostics,
captures all console logs after the multi-frame diagnostic completes (~180 frames),
parses key metrics, and saves both raw logs and a parsed summary.
"""

import sys
import os
import re
import time
import threading
import http.server
import socketserver
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# -- Config --
CAGE_WALK_DIR = Path(__file__).parent.resolve()
MIXAMO_DIR = CAGE_WALK_DIR.parent / "Mixamo"
PORT = 8765
MAX_WAIT = 240           # max seconds to wait for multi-frame diagnostic (360 frames)
OUTPUT_FILE = CAGE_WALK_DIR / "captured_logs.txt"
SUMMARY_FILE = CAGE_WALK_DIR / "diagnostic_summary.txt"
SCREENSHOT_DIR = CAGE_WALK_DIR / "screenshots"

def get_fbx_name():
    """Get FBX filename from CLI arg or default."""
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if not name.endswith(".fbx"):
            name += ".fbx"
        return name
    return "Capoeira wild.fbx"

def start_server():
    """Start a simple HTTP server in a background thread."""
    os.chdir(str(CAGE_WALK_DIR))
    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({
        '.glb': 'model/gltf-binary',
        '.json': 'application/json',
        '.fbx': 'application/octet-stream',
    })
    # Suppress request logging
    handler.log_message = lambda self, fmt, *args: None
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", PORT), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"[server] Serving on http://localhost:{PORT}")
    return httpd


def extract_msg(raw):
    """Extract the actual console message from Chrome's log format.
    Chrome wraps messages like: 'http://...index.html 1234:56 "actual message"'
    """
    # Try to extract the quoted message content
    # Pattern: URL LINE:COL "message" or URL LINE:COL message
    m = re.match(r'^https?://\S+\s+\d+:\d+\s+(.*)', raw)
    if m:
        return m.group(1).strip('"')
    return raw


def parse_metrics(logs):
    """Parse console logs and extract key metrics into a dict."""
    metrics = {}
    msgs = [entry.get("message", "") for entry in logs]

    for msg in msgs:
        txt = extract_msg(msg)

        # Strain
        m = re.search(r'Avg deviation from rest length: ([\d.]+)%', txt)
        if m: metrics['strain_avg_pct'] = float(m.group(1))

        m = re.search(r'Verts with >30% strain: (\d+) / (\d+) \(([\d.]+)%\)', txt)
        if m: metrics['strain_bad_pct'] = float(m.group(3)); metrics['strain_bad_count'] = int(m.group(1))

        # Cross-thigh blend
        m = re.search(r'Verts with both L\+R thigh >5%: (\d+)', txt)
        if m: metrics['cross_thigh_blend_verts'] = int(m.group(1))

        m = re.search(r'Sharp boundary verts near midline.*: (\d+)', txt)
        if m: metrics['sharp_boundary_verts'] = int(m.group(1))

        # Cross-boundary edges
        m = re.search(r'crossBoundaryEdges: (\d+)', txt)
        if m: metrics['cross_boundary_edges'] = int(m.group(1))

        # Crotch stretch
        m = re.search(r'WORST STRETCH: ([\d.]+)x', txt)
        if m:
            val = float(m.group(1))
            # Keep the worst one seen
            if val > metrics.get('crotch_worst_stretch', 0):
                metrics['crotch_worst_stretch'] = val

        # Post-collapse
        m = re.search(r'Edges with >5x stretch \(AFTER collapse\): (\d+)', txt)
        if m: metrics['post_collapse_5x_edges'] = int(m.group(1))

        m = re.search(r'WORST OVERALL: v\d+.v\d+ ([\d.]+)x', txt)
        if m: metrics['post_collapse_worst'] = float(m.group(1))

        # Knee boundary blend
        m = re.search(r'Verts in knee band.*: (\d+) \| blended\(thigh\+shin\): (\d+)', txt)
        if m: metrics['knee_band_total'] = int(m.group(1)); metrics['knee_band_blended'] = int(m.group(2))

        # Bone vs cage
        m = re.search(r'Overall: avg ([\d.]+)mm', txt)
        if m: metrics['bone_cage_avg_mm'] = float(m.group(1))

        m = re.search(r'Worst region: (\w+) \(([\d.]+)mm', txt)
        if m and 'Overall:' in txt: metrics['bone_cage_worst_region'] = m.group(1); metrics['bone_cage_worst_mm'] = float(m.group(2))

        # Per-region bone vs cage
        m = re.search(r'Region \d+ \((\w+)\): avg ([\d.]+)mm, max ([\d.]+)mm', txt)
        if m:
            key = f'bone_cage_{m.group(1)}'
            metrics[key + '_avg'] = float(m.group(2))
            metrics[key + '_max'] = float(m.group(3))

        # Weight smoothness
        m = re.search(r'Abrupt transition edges.*: (\d+) / (\d+)', txt)
        if m: metrics['abrupt_weight_edges'] = int(m.group(1)); metrics['total_edges'] = int(m.group(2))

        # Multi-frame summary
        m = re.search(r'Worst edge stretch: ([\d.]+)x at frame (\d+).*?v\d+\[(\w+)\].*?v\d+\[(\w+)\]', txt)
        if m:
            metrics['mf_worst_stretch'] = float(m.group(1))
            metrics['mf_worst_stretch_frame'] = int(m.group(2))
            metrics['mf_worst_stretch_regions'] = m.group(3) + '/' + m.group(4)
        else:
            m = re.search(r'Worst edge stretch: ([\d.]+)x at frame (\d+)', txt)
            if m: metrics['mf_worst_stretch'] = float(m.group(1)); metrics['mf_worst_stretch_frame'] = int(m.group(2))

        m = re.search(r'Avg strain per frame: ([\d.]+)% \| Worst frame avg: ([\d.]+)%', txt)
        if m: metrics['mf_avg_strain_pct'] = float(m.group(1)); metrics['mf_worst_frame_strain_pct'] = float(m.group(2))

        m = re.search(r'Worst vertex jitter.*: ([\d.]+)mm/f.*at v(\d+)\[(\w+)\]', txt)
        if m: metrics['mf_worst_jitter_mm'] = float(m.group(1)); metrics['mf_worst_jitter_vert'] = int(m.group(2)); metrics['mf_worst_jitter_region'] = m.group(3)

        m = re.search(r'Avg vertex jitter.*: ([\d.]+)mm/f', txt)
        if m: metrics['mf_avg_jitter_mm'] = float(m.group(1))

        m = re.search(r'Floor penetration: L=(\S+) R=(\S+)', txt)
        if m:
            metrics['ground_penetration_L'] = m.group(1)
            metrics['ground_penetration_R'] = m.group(2)

        m = re.search(r'Floor gap \(float\): L=(\S+) R=(\S+)', txt)
        if m:
            metrics['ground_gap_L'] = m.group(1)
            metrics['ground_gap_R'] = m.group(2)

        # Surface health: triangle inversions
        m = re.search(r'Total inverted: (\d+)/(\d+) \(([\d.]+)%\) (\w+)', txt)
        if m:
            metrics['inverted_tris'] = int(m.group(1))
            metrics['total_tris'] = int(m.group(2))
            metrics['inverted_pct'] = float(m.group(3))
            metrics['inverted_status'] = m.group(4)

        # Surface health: per-region inversion (appears after "TRIANGLE INVERSIONS")
        # Format: "    l_uarm: 197/4726 (4.2%)"
        m = re.search(r'^\s+(l_\w+|r_\w+|torso|head|pelvis): (\d+)/(\d+) \(([\d.]+)%\)', txt)
        if m:
            if 'inverted_per_region' not in metrics:
                metrics['inverted_per_region'] = {}
            metrics['inverted_per_region'][m.group(1)] = {
                'count': int(m.group(2)),
                'total': int(m.group(3)),
                'pct': float(m.group(4))
            }

        # Surface health: normal displacement per region
        # Format: "    torso         bulge: avg=  15.2mm max=  41.3mm (v47292) WARN  cave: avg=  -16.2mm max=  -39.7mm (v13804) WARN"
        m = re.search(r'(\w+)\s+bulge: avg=\s*([\d.]+)mm max=\s*([\d.]+)mm \(v\d+\) (\w+)\s+cave: avg=\s*([\-\d.]+)mm max=\s*([\-\d.]+)mm \(v\d+\) (\w+)', txt)
        if m:
            if 'surface_health' not in metrics:
                metrics['surface_health'] = {}
            metrics['surface_health'][m.group(1)] = {
                'bulge_avg': float(m.group(2)),
                'bulge_max': float(m.group(3)),
                'bulge_status': m.group(4),
                'cave_avg': float(m.group(5)),
                'cave_max': float(m.group(6)),
                'cave_status': m.group(7)
            }

        # Boundary seam gap
        m = re.search(r'(torso↔\w+|pelvis↔\w+|\w+↔\w+) \(([^)]+)\): (\d+) boundary verts, avg=([\d.]+)mm, max=([\d.]+)mm \(v(\d+)\) (\w+)', txt)
        if m:
            key = m.group(2).replace(' ', '_')
            if 'seam_gaps' not in metrics:
                metrics['seam_gaps'] = {}
            metrics['seam_gaps'][key] = {
                'label': m.group(1) + ' (' + m.group(2) + ')',
                'count': int(m.group(3)),
                'avg_mm': float(m.group(4)),
                'max_mm': float(m.group(5)),
                'vert': int(m.group(6)),
                'status': m.group(7)
            }

        # Proportion comparison
        m = re.search(r'Arm total:\s+FBX=([\d.]+)\s+Cage=([\d.]+)\s+ratio=([\d.]+)\s+(\w+)', txt)
        if m:
            metrics['prop_arm_fbx'] = float(m.group(1))
            metrics['prop_arm_cage'] = float(m.group(2))
            metrics['prop_arm_ratio'] = float(m.group(3))
            metrics['prop_arm_status'] = m.group(4)

        m = re.search(r'Shoulder spread:\s+FBX=([\d.]+)\s+Cage=([\d.]+)\s+ratio=([\d.]+)\s+(\w+)', txt)
        if m:
            metrics['prop_shoulder_fbx'] = float(m.group(1))
            metrics['prop_shoulder_cage'] = float(m.group(2))
            metrics['prop_shoulder_ratio'] = float(m.group(3))
            metrics['prop_shoulder_status'] = m.group(4)

        m = re.search(r'Hip spread:\s+FBX=([\d.]+)\s+Cage=([\d.]+)\s+ratio=([\d.]+)\s+(\w+)', txt)
        if m:
            metrics['prop_hip_fbx'] = float(m.group(1))
            metrics['prop_hip_cage'] = float(m.group(2))
            metrics['prop_hip_ratio'] = float(m.group(3))
            metrics['prop_hip_status'] = m.group(4)

        m = re.search(r'Total reach.*FBX=([\d.]+)\s+Cage=([\d.]+)\s+diff=([\-\d.]+)mm\s+(\w+)', txt)
        if m:
            metrics['prop_reach_fbx'] = float(m.group(1))
            metrics['prop_reach_cage'] = float(m.group(2))
            metrics['prop_reach_diff_mm'] = float(m.group(3))
            metrics['prop_reach_status'] = m.group(4)

        # Arm reach diagnostic
        m = re.search(r'Wrist-to-wrist: cage=([\d.]+)mm fbx=([\d.]+)mm diff=([\-\d.]+)mm', txt)
        if m:
            metrics['wrist_dist_cage_mm'] = float(m.group(1))
            metrics['wrist_dist_fbx_mm'] = float(m.group(2))
            metrics['wrist_dist_diff_mm'] = float(m.group(3))

        m = re.search(r'(L|R) arm: cageReach=([\d.]+)mm fbxReach=([\d.]+)mm diff=([\-\d.]+)mm', txt)
        if m:
            side = m.group(1).lower()
            metrics[f'arm_reach_{side}_cage_mm'] = float(m.group(2))
            metrics[f'arm_reach_{side}_fbx_mm'] = float(m.group(3))
            metrics[f'arm_reach_{side}_diff_mm'] = float(m.group(4))

        m = re.search(r'wrist posErr=([\d.]+)mm\s+elbow posErr=([\d.]+)mm', txt)
        if m:
            if 'arm_wrist_err' not in metrics:
                metrics['arm_wrist_err'] = float(m.group(1))
                metrics['arm_elbow_err'] = float(m.group(2))

        # Bone direction comparison per-bone
        # Format: "  l_thigh       dir err: avg=X.X° max=X.X° (fNN)  STATUS  |  ang vel: avg=X.X max=X.X°/f  STATUS  |  ang accel: avg=X.XX max=X.X°/f²  STATUS"
        m = re.search(r'(\w+)\s+dir err: avg=([\d.]+).*?max=([\d.]+).*?\(f(\d+)\)\s+(\w+)\s+\|\s+ang vel: avg=([\d.]+)\s+max=([\d.]+).*?(\w+)\s+\|\s+ang accel: avg=([\d.]+)\s+max=([\d.]+).*?(\w+)', txt)
        if m:
            if 'bone_dirs' not in metrics:
                metrics['bone_dirs'] = {}
            metrics['bone_dirs'][m.group(1)] = {
                'err_avg': float(m.group(2)),
                'err_max': float(m.group(3)),
                'err_worst_frame': int(m.group(4)),
                'err_status': m.group(5),
                'vel_avg': float(m.group(6)),
                'vel_max': float(m.group(7)),
                'vel_status': m.group(8),
                'accel_avg': float(m.group(9)),
                'accel_max': float(m.group(10)),
                'accel_status': m.group(11),
            }

        # Worst direction error summary
        m = re.search(r'Worst direction error: (\w+) \(([\d.]+)', txt)
        if m:
            metrics['bone_dir_worst_bone'] = m.group(1)
            metrics['bone_dir_worst_err'] = float(m.group(2))

        # Joint angle comparison per-joint (motion fidelity format)
        # Format: "  l_knee        motion err: avg=X.X° max=X.X° (fNN) STATUS  rest offset=X.X°  vel diff: avg=X.XX max=X.X°/f"
        m = re.search(r'(\w+)\s+motion err: avg=([\d.]+).*?max=([\d.]+).*?\(f(\d+)\)\s+(\w+)\s+rest offset=([-\d.]+).*?vel diff: avg=([\d.]+)\s+max=([\d.]+)', txt)
        if m:
            if 'joint_angles' not in metrics:
                metrics['joint_angles'] = {}
            metrics['joint_angles'][m.group(1)] = {
                'err_avg': float(m.group(2)),
                'err_max': float(m.group(3)),
                'worst_frame': int(m.group(4)),
                'status': m.group(5),
                'rest_offset': float(m.group(6)),
                'vel_diff_avg': float(m.group(7)),
                'vel_diff_max': float(m.group(8)),
            }

        # Worst joint motion error
        m = re.search(r'Worst joint motion error: (\w+) \(([\d.]+)', txt)
        if m:
            metrics['joint_angle_worst'] = m.group(1)
            metrics['joint_angle_worst_err'] = float(m.group(2))

        # Bone length comparison per-bone
        # Format: "  l_thigh       ours=XX.Xmm fbx=XX.Xmm ratio=X.XXX range=[X.XXX-X.XXX]  STATUS"
        m = re.search(r'(\w+)\s+ours=([\d.]+)mm\s+fbx=([\d.]+)mm\s+ratio=([\d.]+)\s+range=\[([\d.]+)-([\d.]+)\]\s+(\w+)', txt)
        if m:
            if 'bone_lengths' not in metrics:
                metrics['bone_lengths'] = {}
            metrics['bone_lengths'][m.group(1)] = {
                'ours_mm': float(m.group(2)),
                'fbx_mm': float(m.group(3)),
                'ratio': float(m.group(4)),
                'ratio_min': float(m.group(5)),
                'ratio_max': float(m.group(6)),
                'status': m.group(7),
            }

        # Shape fidelity per-segment
        # Format: "  l_thigh       bend: avg=X.X° max=X.X°  plane: avg=X.X° max=X.X°  combined=X.X° (fNN)  STATUS"
        m = re.search(r'(\w+)\s+bend: avg=([\d.]+).*?max=([\d.]+).*?plane: avg=([\d.]+).*?max=([\d.]+).*?combined=([\d.]+).*?\(f(\d+)\)\s+(\w+)', txt)
        if m:
            if 'shape_fidelity' not in metrics:
                metrics['shape_fidelity'] = {}
            metrics['shape_fidelity'][m.group(1)] = {
                'bend_avg': float(m.group(2)),
                'bend_max': float(m.group(3)),
                'plane_avg': float(m.group(4)),
                'plane_max': float(m.group(5)),
                'combined': float(m.group(6)),
                'worst_frame': int(m.group(7)),
                'status': m.group(8),
            }

        # Shape fidelity overall
        m = re.search(r'Overall shape fidelity: ([\d.]+)', txt)
        if m:
            metrics['shape_fidelity_avg'] = float(m.group(1))

        # Shape fidelity worst segment
        m = re.search(r'Worst segment: (\w+) \(([\d.]+)', txt)
        if m:
            metrics['shape_fidelity_worst_seg'] = m.group(1)
            metrics['shape_fidelity_worst_err'] = float(m.group(2))

        # Contact drift per-limb
        # Format: "  l_foot        locked N frames  avg drift=X.Xmm  max drift=X.Xmm  STATUS"
        m = re.search(r'(\w+)\s+locked (\d+) frames\s+avg drift=([\d.]+)mm\s+max drift=([\d.]+)mm\s+(\w+)', txt)
        if m:
            if 'contact_drift' not in metrics:
                metrics['contact_drift'] = {}
            metrics['contact_drift'][m.group(1)] = {
                'locked_frames': int(m.group(2)),
                'avg_drift': float(m.group(3)),
                'max_drift': float(m.group(4)),
                'status': m.group(5),
            }

        # Contact drift: no locks detected
        m = re.search(r'(\w+)\s+locked 0 frames\s+\(no locks detected\)', txt)
        if m:
            if 'contact_drift' not in metrics:
                metrics['contact_drift'] = {}
            metrics['contact_drift'][m.group(1)] = {
                'locked_frames': 0, 'avg_drift': 0, 'max_drift': 0, 'status': 'N/A',
            }

        # Proportion mismatch per-joint (renamed from relative position)
        # Format: "  l_knee        prop mismatch: avg=X.XXXX max=X.XXXX (fNN)  STATUS"
        m = re.search(r'(\w+)\s+prop mismatch: avg=([\d.]+)\s+max=([\d.]+)\s+\(f(\d+)\)\s+(\w+)', txt)
        if m:
            if 'rel_pos' not in metrics:
                metrics['rel_pos'] = {}
            metrics['rel_pos'][m.group(1)] = {
                'err_avg': float(m.group(2)),
                'err_max': float(m.group(3)),
                'worst_frame': int(m.group(4)),
                'status': m.group(5),
            }

        # Worst proportion mismatch (renamed from relative position error)
        m = re.search(r'Worst proportion mismatch: (\w+) \(([\d.]+)', txt)
        if m:
            metrics['rel_pos_worst'] = m.group(1)
            metrics['rel_pos_worst_err'] = float(m.group(2))

        # Oscillation analysis per region
        # Format: "  l_foot      mean=XX.Xmm std=XX.Xmm range=[XX.X-XX.X] CV=X.XXX  TYPE"
        m = re.search(r'(\w+)\s+mean=([\d.]+)mm\s+std=([\d.]+)mm\s+range=\[([\d.]+)-([\d.]+)\]\s+CV=([\d.]+)\s+(\w+)', txt)
        if m:
            if 'oscillation' not in metrics:
                metrics['oscillation'] = {}
            metrics['oscillation'][m.group(1)] = {
                'mean': float(m.group(2)),
                'std': float(m.group(3)),
                'min': float(m.group(4)),
                'max': float(m.group(5)),
                'cv': float(m.group(6)),
                'type': m.group(7),
            }

        # Twist comparison per bone
        # Format: "  l_uarm    avg_err=X.X° max_err=X.X° (fNN)  cage_avg=X.X° fbx_avg=X.X°  STATUS"
        m = re.search(r'(\w+)\s+avg_err=([\d.]+).*?max_err=([\d.]+).*?\(f(\d+)\)\s+cage_avg=([\d.]+).*?fbx_avg=([\d.]+).*?\s+(\w+)', txt)
        if m:
            if 'twist_comp' not in metrics:
                metrics['twist_comp'] = {}
            metrics['twist_comp'][m.group(1)] = {
                'err_avg': float(m.group(2)),
                'err_max': float(m.group(3)),
                'worst_frame': int(m.group(4)),
                'cage_avg': float(m.group(5)),
                'fbx_avg': float(m.group(6)),
                'status': m.group(7),
            }

        # FBX ground truth comparison
        # Overall: "Overall avg error:          X.Xmm"
        m = re.search(r'Overall avg error:\s+([\d.]+)mm', txt)
        if m:
            metrics['fbx_gt_overall_avg_mm'] = float(m.group(1))

        # Worst region: "Worst region:               region_name (X.Xmm max)"
        m = re.search(r'Worst region:\s+(\w+) \(([\d.]+)mm max\)', txt)
        if m:
            if 'fbx_gt_worst_region' not in metrics:
                metrics['fbx_gt_worst_region'] = m.group(1)
                metrics['fbx_gt_worst_max_mm'] = float(m.group(2))

        # Per-region FBX GT: "    l_thigh       avg   X.Xmm  max   X.Xmm (fNN)  STATUS"
        m = re.search(r'(\w+)\s+avg\s+([\d.]+)mm\s+max\s+([\d.]+)mm\s+\(f(\d+)\)\s+(\w+)', txt)
        if m:
            if 'fbx_gt_regions' not in metrics:
                metrics['fbx_gt_regions'] = {}
            metrics['fbx_gt_regions'][m.group(1)] = {
                'avg_mm': float(m.group(2)),
                'max_mm': float(m.group(3)),
                'worst_frame': int(m.group(4)),
                'status': m.group(5),
            }

        # FBX ref mesh stretch: "FBX ref mesh stretch:       worst=X.Xx  avg strain=X.X%"
        m = re.search(r'FBX ref mesh stretch:\s+worst=([\d.]+)x\s+avg strain=([\d.]+)%', txt)
        if m:
            metrics['fbx_ref_worst_stretch'] = float(m.group(1))
            metrics['fbx_ref_avg_strain_pct'] = float(m.group(2))

        # Cage mesh stretch (from FBX GT section): "Cage mesh stretch:          worst=X.Xx  avg strain=X.X%"
        m = re.search(r'Cage mesh stretch:\s+worst=([\d.]+)x\s+avg strain=([\d.]+)%', txt)
        if m:
            metrics['cage_worst_stretch'] = float(m.group(1))
            metrics['cage_avg_strain_pct'] = float(m.group(2))

        # Correlation analysis
        # Format: "  L shin_dir_err vs foot_jitter: r=X.XXX  STATUS"
        m = re.search(r'([LR]) (\w+) vs (\w+): r=([\-\d.]+)\s+(\w+)', txt)
        if m:
            if 'correlations' not in metrics:
                metrics['correlations'] = []
            metrics['correlations'].append({
                'side': m.group(1),
                'x': m.group(2),
                'y': m.group(3),
                'r': float(m.group(4)),
                'status': m.group(5),
            })

        # Rigidity diagnostic per-segment
        # Format: "  pelvis→spine_joint    mean=X.XX% max=X.XX% blend=X.XXX >2%:X.X% dist=X.X/X.Xmm TAG"
        m = re.search(r'(\w+)\u2192(\w+)\s+mean=([\d.]+)%\s+max=([\d.]+)%\s+blend=([\d.]+)\s+>2%:([\d.]+)%\s+dist=([\d.]+)/([\d.]+)mm\s+(\w+\??)', txt)
        if m:
            if 'rigidity' not in metrics:
                metrics['rigidity'] = {}
            seg = m.group(1) + '→' + m.group(2)
            metrics['rigidity'][seg] = {
                'mean_pct': float(m.group(3)),
                'max_pct': float(m.group(4)),
                'blend': float(m.group(5)),
                'high_err_pct': float(m.group(6)),
                'mean_dist_mm': float(m.group(7)),
                'max_dist_mm': float(m.group(8)),
                'tag': m.group(9),
            }

        # Bend localization per-segment
        # Format: "  pelvis→l_hip          center=X.XX joint=XX% mid=XX% disp=X.XX% TAG"
        m = re.search(r'(\w+)\u2192(\w+)\s+center=([\d.]+)\s+joint=([\d.]+)%\s+mid=([\d.]+)%\s+disp=([\d.]+)%\s+(\w+!?)', txt)
        if m:
            if 'bend_loc' not in metrics:
                metrics['bend_loc'] = {}
            seg = m.group(1) + '→' + m.group(2)
            metrics['bend_loc'][seg] = {
                'center': float(m.group(3)),
                'joint_pct': float(m.group(4)),
                'mid_pct': float(m.group(5)),
                'disp_pct': float(m.group(6)),
                'tag': m.group(7),
            }

    return metrics


def format_summary(metrics, fbx_name):
    """Format metrics into a readable summary."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  DIAGNOSTIC SUMMARY -{fbx_name}")
    lines.append(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'='*60}")
    lines.append("")

    # Grading thresholds
    def grade(val, good, ok, label=""):
        if val <= good: return f"  GOOD"
        elif val <= ok: return f"  OK"
        else: return f"  BAD"

    lines.append("-- STRETCH / DEFORMATION --")
    v = metrics.get('mf_worst_stretch', metrics.get('crotch_worst_stretch', 0))
    g = grade(v, 3.0, 8.0)
    sr = metrics.get('mf_worst_stretch_regions', '')
    lines.append(f"  Multi-frame worst stretch:  {v:.1f}x{g}" + (f" [{sr}]" if sr else ""))
    v = metrics.get('mf_avg_strain_pct', metrics.get('strain_avg_pct', 0))
    g = grade(v, 5.0, 10.0)
    lines.append(f"  Avg strain per frame:       {v:.1f}%{g}")
    v = metrics.get('mf_worst_frame_strain_pct', 0)
    if v: lines.append(f"  Worst frame strain:         {v:.1f}%")
    v = metrics.get('post_collapse_worst', 0)
    if v: lines.append(f"  Post-collapse worst:        {v:.1f}x" + grade(v, 3.0, 8.0))
    v = metrics.get('post_collapse_5x_edges', 0)
    lines.append(f"  Edges >5x after collapse:   {v}" + ("  GOOD" if v == 0 else f"  BAD ({v} remaining)"))
    lines.append("")

    lines.append("-- CROTCH / THIGH BLEND --")
    v = metrics.get('cross_thigh_blend_verts', 0)
    lines.append(f"  Cross-thigh blend verts:    {v}" + ("  GOOD" if v > 1000 else "  BAD (need >1000)"))
    v = metrics.get('sharp_boundary_verts', 0)
    lines.append(f"  Sharp boundary verts:       {v}" + ("  GOOD" if v < 500 else "  OK" if v < 2000 else "  BAD"))
    v = metrics.get('cross_boundary_edges', 0)
    lines.append(f"  Cross-boundary edges:       {v}" + ("  GOOD" if v == 0 else f"  BAD ({v} edges)"))
    lines.append("")

    lines.append("-- KNEE BLEND --")
    t = metrics.get('knee_band_total', 0)
    b = metrics.get('knee_band_blended', 0)
    pct = (b / t * 100) if t > 0 else 0
    lines.append(f"  Knee band verts:            {t} total, {b} blended ({pct:.0f}%)" +
                 ("  GOOD" if pct > 60 else "  BAD"))
    lines.append("")

    lines.append("-- JITTER --")
    v = metrics.get('mf_worst_jitter_mm', 0)
    vert = metrics.get('mf_worst_jitter_vert', '?')
    jr = metrics.get('mf_worst_jitter_region', '?')
    g = grade(v, 30.0, 80.0)  # cage deformation: 30mm/f^2 good, 80mm/f^2 ok (fast animation)
    lines.append(f"  Worst jitter (accel):       {v:.1f}mm/f^2 (v{vert} [{jr}]){g}")
    v = metrics.get('mf_avg_jitter_mm', 0)
    lines.append(f"  Avg jitter (accel):         {v:.2f}mm/f^2" + grade(v, 5.0, 12.0))
    lines.append("")

    lines.append("-- FOOT GROUND CONTACT --")
    lp = metrics.get('ground_penetration_L', 'N/A')
    rp = metrics.get('ground_penetration_R', 'N/A')
    lines.append(f"  L foot penetration:         {lp}")
    lines.append(f"  R foot penetration:         {rp}")
    lg = metrics.get('ground_gap_L', 'N/A')
    rg = metrics.get('ground_gap_R', 'N/A')
    lines.append(f"  L foot float (gap):         {lg}")
    lines.append(f"  R foot float (gap):         {rg}")
    lines.append("")

    lines.append("-- WEIGHT SMOOTHNESS --")
    ae = metrics.get('abrupt_weight_edges', 0)
    te = metrics.get('total_edges', 0)
    pct = (ae / te * 100) if te > 0 else 0
    lines.append(f"  Abrupt weight edges:        {ae} / {te} ({pct:.2f}%)" +
                 ("  GOOD" if pct < 1.0 else "  OK" if pct < 5.0 else "  BAD"))
    lines.append("")

    lines.append("-- BONE vs CAGE COMPARISON --")
    v = metrics.get('bone_cage_avg_mm', 0)
    lines.append(f"  Overall avg error:          {v:.1f}mm")
    wr = metrics.get('bone_cage_worst_region', '?')
    wv = metrics.get('bone_cage_worst_mm', 0)
    lines.append(f"  Worst region:               {wr} ({wv:.1f}mm max)")
    # Show per-region if available
    region_order = ['l_thigh','r_thigh','l_shin','r_shin','l_foot','r_foot',
                    'l_uarm','r_uarm','l_farm','r_farm','head','torso','pelvis']
    for rn in region_order:
        avg = metrics.get(f'bone_cage_{rn}_avg')
        mx = metrics.get(f'bone_cage_{rn}_max')
        if avg is not None:
            lines.append(f"    {rn:12s}  avg {avg:6.1f}mm  max {mx:6.1f}mm" +
                         ("  GOOD" if mx < 20 else "  OK" if mx < 50 else ""))

    lines.append("")

    # Surface health
    inv = metrics.get('inverted_tris')
    if inv is not None:
        lines.append("-- SURFACE HEALTH --")
        total_t = metrics.get('total_tris', 0)
        inv_pct = metrics.get('inverted_pct', 0)
        inv_st = metrics.get('inverted_status', '')
        lines.append(f"  Inverted triangles:         {inv}/{total_t} ({inv_pct:.2f}%)  {inv_st}")
        # Per-region inversions
        ipr = metrics.get('inverted_per_region', {})
        for rn in sorted(ipr.keys(), key=lambda k: ipr[k]['count'], reverse=True):
            r = ipr[rn]
            lines.append(f"    {rn:12s}  {r['count']}/{r['total']} ({r['pct']:.1f}%)")
        # Per-region bulge/cave
        sh = metrics.get('surface_health', {})
        if sh:
            lines.append("  Normal displacement:")
            region_order_sh = ['torso','l_thigh','r_thigh','l_shin','r_shin','l_foot','r_foot',
                               'l_uarm','r_uarm','l_farm','r_farm','head','pelvis']
            for rn in region_order_sh:
                if rn in sh:
                    s = sh[rn]
                    lines.append(f"    {rn:12s}  bulge max={s['bulge_max']:5.1f}mm {s['bulge_status']:4s}  cave max={s['cave_max']:6.1f}mm {s['cave_status']}")
        lines.append("")

    # Boundary seam gaps
    seam_gaps = metrics.get('seam_gaps', {})
    if seam_gaps:
        lines.append("-- BOUNDARY SEAM GAPS --")
        for key in sorted(seam_gaps.keys()):
            sg = seam_gaps[key]
            lines.append(f"  {sg['label']:35s} {sg['count']:4d} verts, avg={sg['avg_mm']:5.1f}mm, max={sg['max_mm']:5.1f}mm  {sg['status']}")
        lines.append("")

    # Proportion comparison
    if metrics.get('prop_arm_ratio') is not None:
        lines.append("-- PROPORTION COMPARISON --")
        ar = metrics.get('prop_arm_ratio', 0)
        lines.append(f"  Arm length ratio:           {ar:.3f} (FBX={metrics.get('prop_arm_fbx',0):.4f} Cage={metrics.get('prop_arm_cage',0):.4f})  {metrics.get('prop_arm_status','')}")
        sr = metrics.get('prop_shoulder_ratio', 0)
        lines.append(f"  Shoulder spread ratio:      {sr:.3f} (FBX={metrics.get('prop_shoulder_fbx',0):.4f} Cage={metrics.get('prop_shoulder_cage',0):.4f})  {metrics.get('prop_shoulder_status','')}")
        hr = metrics.get('prop_hip_ratio', 0)
        lines.append(f"  Hip spread ratio:           {hr:.3f} (FBX={metrics.get('prop_hip_fbx',0):.4f} Cage={metrics.get('prop_hip_cage',0):.4f})  {metrics.get('prop_hip_status','')}")
        rd = metrics.get('prop_reach_diff_mm', 0)
        lines.append(f"  Total reach diff:           {rd:.1f}mm  {metrics.get('prop_reach_status','')}")
        lines.append("")

    # Bone direction comparison
    bone_dirs = metrics.get('bone_dirs', {})
    if bone_dirs:
        lines.append("-- BONE DIRECTION COMPARISON --")
        wb = metrics.get('bone_dir_worst_bone', '?')
        we = metrics.get('bone_dir_worst_err', 0)
        lines.append(f"  Worst direction error:      {wb} ({we:.1f}° max)")
        bone_order = ['l_thigh','r_thigh','l_shin','r_shin','l_foot','r_foot',
                      'l_uarm','r_uarm','l_farm','r_farm','head','torso','neck']
        for bn in bone_order:
            if bn in bone_dirs:
                d = bone_dirs[bn]
                lines.append(f"    {bn:12s}  err: avg={d['err_avg']:5.1f}° max={d['err_max']:5.1f}° (f{d['err_worst_frame']}) {d['err_status']:4s}"
                             f"  vel: avg={d['vel_avg']:4.1f} max={d['vel_max']:5.1f}°/f {d['vel_status']:4s}"
                             f"  accel: avg={d['accel_avg']:5.2f} max={d['accel_max']:5.1f}°/f² {d['accel_status']}")
        lines.append("")

    # Joint angle comparison (motion fidelity)
    joint_angles = metrics.get('joint_angles', {})
    if joint_angles:
        lines.append("-- JOINT ANGLE COMPARISON (motion fidelity) --")
        wj = metrics.get('joint_angle_worst', '?')
        wje = metrics.get('joint_angle_worst_err', 0)
        lines.append(f"  Worst joint motion error:   {wj} ({wje:.1f}° max)")
        ja_order = ['l_hip','r_hip','l_knee','r_knee','l_ankle','r_ankle',
                     'l_shoulder','r_shoulder','l_elbow','r_elbow']
        for jn in ja_order:
            if jn in joint_angles:
                d = joint_angles[jn]
                rest_off = d.get('rest_offset', 0)
                lines.append(f"    {jn:12s}  motion: avg={d['err_avg']:5.1f}° max={d['err_max']:5.1f}° (f{d['worst_frame']}) {d['status']:4s}"
                             f"  rest offset={rest_off:5.1f}°"
                             f"  vel diff: avg={d['vel_diff_avg']:5.2f} max={d['vel_diff_max']:5.1f}°/f")
        lines.append("")

    # Bone length comparison
    bone_lengths = metrics.get('bone_lengths', {})
    if bone_lengths:
        lines.append("-- BONE LENGTH COMPARISON --")
        bl_order = ['l_thigh','r_thigh','l_shin','r_shin','l_foot','r_foot',
                     'l_uarm','r_uarm','l_farm','r_farm']
        for bn in bl_order:
            if bn in bone_lengths:
                d = bone_lengths[bn]
                lines.append(f"    {bn:12s}  ours={d['ours_mm']:6.1f}mm  fbx={d['fbx_mm']:6.1f}mm  ratio={d['ratio']:.3f}"
                             f"  [{d['ratio_min']:.3f}-{d['ratio_max']:.3f}]  {d['status']}")
        lines.append("")

    # Shape fidelity (proportion-invariant)
    shape_fid = metrics.get('shape_fidelity', {})
    if shape_fid:
        lines.append("-- SHAPE FIDELITY (proportion-invariant) --")
        sf_avg = metrics.get('shape_fidelity_avg', 0)
        sf_worst_seg = metrics.get('shape_fidelity_worst_seg', '?')
        sf_worst_err = metrics.get('shape_fidelity_worst_err', 0)
        lines.append(f"  Overall avg:                {sf_avg:.1f}\u00b0")
        lines.append(f"  Worst segment:              {sf_worst_seg} ({sf_worst_err:.1f}\u00b0 combined max)")
        sf_order = ['l_thigh','r_thigh','l_shin','r_shin','l_foot','r_foot',
                     'l_uarm','r_uarm','l_farm','r_farm','torso']
        for seg in sf_order:
            if seg in shape_fid:
                d = shape_fid[seg]
                lines.append(f"    {seg:12s}  bend: avg={d['bend_avg']:5.1f}\u00b0 max={d['bend_max']:5.1f}\u00b0"
                             f"  plane: avg={d['plane_avg']:5.1f}\u00b0 max={d['plane_max']:5.1f}\u00b0"
                             f"  combined={d['combined']:5.1f}\u00b0 (f{d['worst_frame']}) {d['status']}")
        lines.append("")

    # Contact drift (locked limb displacement)
    contact_drift = metrics.get('contact_drift', {})
    if contact_drift:
        lines.append("-- CONTACT DRIFT --")
        cd_order = ['l_foot','r_foot','l_hand','r_hand']
        for limb in cd_order:
            if limb in contact_drift:
                d = contact_drift[limb]
                if d['locked_frames'] == 0:
                    lines.append(f"    {limb:12s}  (no locks detected)")
                else:
                    lines.append(f"    {limb:12s}  locked {d['locked_frames']} frames"
                                 f"  avg={d['avg_drift']:5.1f}mm  max={d['max_drift']:5.1f}mm  {d['status']}")
        lines.append("")

    # Proportion mismatch (pelvis-relative, height-normalized) — diagnostic only
    rel_pos = metrics.get('rel_pos', {})
    if rel_pos:
        lines.append("-- PROPORTION MISMATCH (diagnostic only) --")
        wrp = metrics.get('rel_pos_worst', '?')
        wrpe = metrics.get('rel_pos_worst_err', 0)
        lines.append(f"  Worst proportion mismatch:  {wrp} ({wrpe:.4f} max)")
        rp_order = ['l_knee','r_knee','l_ankle','r_ankle','l_toe','r_toe',
                     'l_elbow','r_elbow','l_wrist','r_wrist','l_shoulder','r_shoulder','head']
        for jn in rp_order:
            if jn in rel_pos:
                d = rel_pos[jn]
                lines.append(f"    {jn:12s}  err: avg={d['err_avg']:.4f} max={d['err_max']:.4f} (f{d['worst_frame']}) {d['status']}")
        lines.append("")

    # Oscillation analysis
    osc = metrics.get('oscillation', {})
    if osc:
        lines.append("-- OSCILLATION ANALYSIS --")
        for rn in ['l_foot','r_foot','l_shin','r_shin']:
            if rn in osc:
                d = osc[rn]
                lines.append(f"    {rn:12s}  mean={d['mean']:6.1f}mm  std={d['std']:5.1f}mm  "
                             f"range=[{d['min']:.1f}-{d['max']:.1f}]  CV={d['cv']:.3f}  {d['type']}")
        lines.append("")

    # Correlation analysis
    corrs = metrics.get('correlations', [])
    if corrs:
        lines.append("-- CORRELATION ANALYSIS --")
        for c in corrs:
            lines.append(f"  {c['side']} {c['x']} vs {c['y']}: r={c['r']:.3f}  {c['status']}")
        lines.append("")

    # Twist comparison (cage vs FBX)
    twist_comp = metrics.get('twist_comp', {})
    if twist_comp:
        lines.append("-- TWIST COMPARISON (cage vs FBX) --")
        tw_order = ['l_uarm','r_uarm','l_farm','r_farm','l_thigh','r_thigh']
        for bn in tw_order:
            if bn in twist_comp:
                d = twist_comp[bn]
                lines.append(f"    {bn:12s}  err: avg={d['err_avg']:5.1f}\u00b0 max={d['err_max']:5.1f}\u00b0 (f{d['worst_frame']})"
                             f"  cage={d['cage_avg']:5.1f}\u00b0 fbx={d['fbx_avg']:5.1f}\u00b0  {d['status']}")
        lines.append("")

    # FBX ground truth comparison
    fbx_gt_regions = metrics.get('fbx_gt_regions', {})
    fbx_gt_avg = metrics.get('fbx_gt_overall_avg_mm')
    if fbx_gt_avg is not None or fbx_gt_regions:
        lines.append("-- FBX GROUND TRUTH COMPARISON --")
        if fbx_gt_avg is not None:
            lines.append(f"  Overall avg error:          {fbx_gt_avg:.1f}mm")
        wr = metrics.get('fbx_gt_worst_region', '?')
        wm = metrics.get('fbx_gt_worst_max_mm', 0)
        lines.append(f"  Worst region:               {wr} ({wm:.1f}mm max)")
        gt_order = ['l_thigh','r_thigh','l_shin','r_shin','l_foot','r_foot',
                     'l_uarm','r_uarm','l_farm','r_farm','head','torso','pelvis',
                     'spine1','spine2','chest','l_toe','r_toe']
        for rn in gt_order:
            if rn in fbx_gt_regions:
                d = fbx_gt_regions[rn]
                lines.append(f"    {rn:12s}  avg {d['avg_mm']:6.1f}mm  max {d['max_mm']:6.1f}mm (f{d['worst_frame']})  {d['status']}")
        # Stretch comparison
        frs = metrics.get('fbx_ref_worst_stretch')
        if frs is not None:
            fra = metrics.get('fbx_ref_avg_strain_pct', 0)
            lines.append(f"  FBX ref mesh stretch:       worst={frs:.1f}x  avg strain={fra:.1f}%")
        cws = metrics.get('cage_worst_stretch')
        if cws is not None:
            cas = metrics.get('cage_avg_strain_pct', 0)
            lines.append(f"  Cage mesh stretch:          worst={cws:.1f}x  avg strain={cas:.1f}%")
        lines.append("")

    # Arm reach
    wdc = metrics.get('wrist_dist_cage_mm')
    if wdc is not None:
        lines.append("-- ARM REACH --")
        wdf = metrics.get('wrist_dist_fbx_mm', 0)
        wdd = metrics.get('wrist_dist_diff_mm', 0)
        lines.append(f"  Wrist-to-wrist distance:    cage={wdc:.1f}mm  fbx={wdf:.1f}mm  diff={wdd:.1f}mm" +
                     ("  MISMATCH" if abs(wdd) > 50 else "  OK"))
        for side in ['l', 'r']:
            rc = metrics.get(f'arm_reach_{side}_cage_mm')
            rf = metrics.get(f'arm_reach_{side}_fbx_mm')
            rd = metrics.get(f'arm_reach_{side}_diff_mm')
            if rc is not None:
                lines.append(f"  {side.upper()} arm reach:            cage={rc:.1f}mm  fbx={rf:.1f}mm  diff={rd:.1f}mm")
        we = metrics.get('arm_wrist_err')
        if we is not None:
            lines.append(f"  Wrist position error:       {we:.1f}mm")
        lines.append("")

    # Rigidity diagnostic
    rigidity = metrics.get('rigidity', {})
    if rigidity:
        lines.append("-- SEGMENT RIGIDITY (domW≥0.8 verts) --")
        for seg in sorted(rigidity.keys()):
            d = rigidity[seg]
            tag_color = 'GOOD' if d['tag'] == 'GOOD' else 'OK' if d['tag'] == 'OK' else 'BAD'
            lines.append(f"    {seg:24s}  mean={d['mean_pct']:5.2f}%  max={d['max_pct']:5.2f}%"
                         f"  blend={d['blend']:.3f}  >2%:{d['high_err_pct']:.1f}%  {d['tag']}")
        lines.append("")

    # Bend localization diagnostic
    bend_loc = metrics.get('bend_loc', {})
    if bend_loc:
        lines.append("-- BEND LOCALIZATION --")
        for seg in sorted(bend_loc.keys()):
            d = bend_loc[seg]
            lines.append(f"    {seg:24s}  center={d['center']:.2f}  joint={d['joint_pct']:.0f}%"
                         f"  mid={d['mid_pct']:.0f}%  disp={d['disp_pct']:.2f}%  {d['tag']}")
        lines.append("")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def main():
    fbx_name = get_fbx_name()
    fbx_path = MIXAMO_DIR / fbx_name

    if not fbx_path.exists():
        print(f"[error] FBX not found: {fbx_path}")
        print(f"[info] Available FBX files in {MIXAMO_DIR}:")
        for f in sorted(MIXAMO_DIR.glob("*.fbx")):
            print(f"  {f.name}")
        sys.exit(1)

    print(f"[config] FBX: {fbx_path.name}")

    # Start local server
    httpd = start_server()

    # Set up Chrome with console log capture
    chrome_options = Options()
    chrome_options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    # Uncomment to run headless (no visible window):
    # chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,900")

    print("[browser] Launching Chrome...")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Load the page
        url = f"http://localhost:{PORT}/index.html"
        print(f"[browser] Loading {url}")
        driver.get(url)

        # Wait for mesh to load
        print("[browser] Waiting for mesh to load...")
        WebDriverWait(driver, 60).until(
            lambda d: "Ready" in d.find_element(By.ID, "status").text
            or "ready" in d.find_element(By.ID, "status").text.lower()
            or "Drop" in d.find_element(By.ID, "status").text
            or "drop" in d.find_element(By.ID, "status").text.lower()
        )
        time.sleep(1)

        # ── MASTER TOGGLE TABLE ──
        # Every checkbox explicitly set. Nothing left to chance.
        TOGGLES_BEFORE_FBX = {
            # Pipeline
            'useCage':           False,  # no cage
            'useFBXTransforms':  True,   # FBX quaternion transforms (bone rotations)
            'useFBXJoints':      False,  # OUR joint placement (placed_joints.json)
            'boneEnvWeights':    False,  # no bone-envelope weights
            'useDQS':            False,  # LBS (not DQS)
            'fbxDeformTransfer': False,  # no deform transfer
            'useBoneLBS':        True,   # per-bone LBS with our proportions
            # Diagnostics
            'showStrain':        False,
            'showBoneError':     False,
            'showFBXGT':         False,
            'showRigidity':      True,
            'showBendLoc':       False,
        }
        print("[browser] Setting toggles BEFORE FBX load:")
        for cb_id, expected in TOGGLES_BEFORE_FBX.items():
            try:
                cb = driver.find_element(By.ID, cb_id)
                actual = cb.is_selected()
                if actual != expected:
                    cb.click()
                    time.sleep(0.3)
                    print(f"  {cb_id:20s} {actual} -> {expected}  FIXED")
                else:
                    print(f"  {cb_id:20s} {actual}  OK")
            except Exception:
                print(f"  {cb_id:20s} NOT FOUND (skipped)")

        # Load the FBX file
        print(f"[browser] Loading FBX: {fbx_path.name}...")
        driver.find_element(By.ID, "fbxFile").send_keys(str(fbx_path))
        time.sleep(3)

        status = driver.find_element(By.ID, "status").text
        print(f"[browser] Status: {status}")

        # Enable FBX weights AFTER FBX load (needs FBX skin data)
        TOGGLES_AFTER_FBX = {
            'useFBXWeights':     True,   # FBX skin weights
        }
        for cb_id, expected in TOGGLES_AFTER_FBX.items():
            try:
                cb = driver.find_element(By.ID, cb_id)
                actual = cb.is_selected()
                if actual != expected:
                    cb.click()
                    time.sleep(1)
                    print(f"  {cb_id:20s} {actual} -> {expected}  FIXED")
                else:
                    print(f"  {cb_id:20s} {actual}  OK")
            except Exception:
                print(f"  {cb_id:20s} NOT FOUND (skipped)")

        # ── FINAL VERIFICATION ── (catch any auto-toggles or interference)
        ALL_EXPECTED = {**TOGGLES_BEFORE_FBX, **TOGGLES_AFTER_FBX}
        print("[browser] FINAL toggle verification:")
        all_ok = True
        for cb_id, expected in ALL_EXPECTED.items():
            try:
                cb = driver.find_element(By.ID, cb_id)
                actual = cb.is_selected()
                ok = actual == expected
                if not ok:
                    all_ok = False
                    cb.click()
                    time.sleep(0.3)
                print(f"  {cb_id:20s} expected={str(expected):5s}  actual={str(actual):5s}  {'OK' if ok else 'FIXED'}")
            except Exception:
                print(f"  {cb_id:20s} NOT FOUND")
        if all_ok:
            print("[browser] ALL TOGGLES VERIFIED CORRECT")
        else:
            print("[browser] WARNING: Had to fix toggles after FBX load!")
            time.sleep(1)

        # Wait for multi-frame diagnostic summary (polls logs every 2s)
        print("[browser] Waiting for multi-frame diagnostic (180 frames)...")
        mf_found = False
        elapsed = 0
        all_logs = []
        while elapsed < MAX_WAIT:
            time.sleep(2)
            elapsed += 2
            new_logs = driver.get_log("browser")
            all_logs.extend(new_logs)
            # Check if multi-frame summary appeared
            for entry in new_logs:
                if "MULTI-FRAME DIAGNOSTIC SUMMARY" in entry.get("message", ""):
                    mf_found = True
                    break
            if mf_found:
                # Grab any remaining logs
                time.sleep(1)
                all_logs.extend(driver.get_log("browser"))
                break
            print(f"  ... {elapsed}s", end="\r")

        if not mf_found:
            print(f"\n[warn] Multi-frame diagnostic didn't complete in {MAX_WAIT}s, capturing what we have")

        # ── FIXED-FRAME SHARPENING EVALUATION ──
        # Test upper/lower sharpening configs using deterministic multi-frame averaging
        SHARPEN_CONFIGS = [
            {'label': 'baseline',    'upper': 0.0, 'lower': 0.0},
            {'label': 'upper=0.7',   'upper': 0.7, 'lower': 0.0},
            {'label': 'split',       'upper': 0.7, 'lower': 0.3},
            {'label': 'aggressive',  'upper': 0.9, 'lower': 0.5},
        ]
        N_SAMPLES = 8  # evenly spaced frames across the clip

        sharpen_parsed = {}  # label -> { seg_name: {mean_pct, avgmax_pct, mean_mm, avgmax_mm, tag} }
        for cfg in SHARPEN_CONFIGS:
            label = cfg['label']
            print(f"\n[sharpen] Config '{label}': upper={cfg['upper']:.1f} lower={cfg['lower']:.1f}")

            # Set both sliders
            driver.execute_script(f"""
                var su = document.getElementById('sharpenUpperSlider');
                var sl = document.getElementById('sharpenLowerSlider');
                su.value = {cfg['upper']};
                sl.value = {cfg['lower']};
                su.dispatchEvent(new Event('input'));
            """)
            time.sleep(1)  # wait for weight rebuild

            # Drain old logs
            driver.get_log("browser")

            # Call fixed-frame evaluation
            result = driver.execute_script(f"return window.evaluateRigidityFixedFrame({N_SAMPLES});")

            # Capture logs from the evaluation
            new_logs = driver.get_log("browser")
            all_logs.extend(new_logs)

            if result:
                sharpen_parsed[label] = result
                for seg, d in sorted(result.items()):
                    if seg.startswith('_'): continue  # skip _segments, _global
                    seg_ascii = seg.replace('\u2192', '->').encode('ascii', 'replace').decode('ascii')
                    print(f"  {seg_ascii:<22} mean={d['mean_pct']:.2f}% avgmax={d['avgmax_pct']:.2f}% "
                          f"dist={d['mean_mm']:.1f}/{d['avgmax_mm']:.1f}mm {d['tag']}")
            else:
                print("  [warn] evaluateRigidityFixedFrame returned null")
                sharpen_parsed[label] = {}

        # Reset sliders
        driver.execute_script("""
            var su = document.getElementById('sharpenUpperSlider');
            var sl = document.getElementById('sharpenLowerSlider');
            su.value = 0; sl.value = 0;
            su.dispatchEvent(new Event('input'));
        """)
        time.sleep(0.5)

        # Save comparison file
        sharpen_file = CAGE_WALK_DIR / "sharpen_comparison.txt"
        with open(sharpen_file, "w", encoding="utf-8") as f:
            f.write(f"=== FIXED-FRAME SHARPENING COMPARISON ({N_SAMPLES} samples/config) ===\n")
            f.write(f"FBX: {fbx_path.name}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for cfg in SHARPEN_CONFIGS:
                label = cfg['label']
                f.write(f"--- {label} (upper={cfg['upper']:.1f} lower={cfg['lower']:.1f}) ---\n")
                data = sharpen_parsed.get(label, {})
                for seg in sorted(data.keys()):
                    if seg.startswith('_'): continue
                    d = data[seg]
                    f.write(f"  {seg:<22} mean={d['mean_pct']:.2f}% avgmax={d['avgmax_pct']:.2f}% "
                            f"blend={d['blend']:.3f} >2%:{d['high_err_pct']:.1f}% "
                            f"dist={d['mean_mm']:.1f}/{d['avgmax_mm']:.1f}mm {d['tag']}\n")
                f.write("\n")

            # Comparison table
            all_segs = sorted(set(seg for p in sharpen_parsed.values() for seg in p if not seg.startswith('_')))
            labels = [c['label'] for c in SHARPEN_CONFIGS]
            if all_segs:
                f.write("\n=== COMPARISON TABLE (averaged over {} frames) ===\n".format(N_SAMPLES))
                header = f"{'Segment':<24}"
                for lb in labels:
                    header += f"  {lb:>10} mean% avgmx% mean_mm avgmx_mm tag       "
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")

                for seg in all_segs:
                    row = f"{seg:<24}"
                    for lb in labels:
                        d = sharpen_parsed.get(lb, {}).get(seg)
                        if d:
                            row += f"  {d['mean_pct']:10.2f}  {d['avgmax_pct']:6.2f}  {d['mean_mm']:7.1f}  {d['avgmax_mm']:8.1f}  {d['tag']:<10}"
                        else:
                            row += f"  {'---':>10}  {'---':>6}  {'---':>7}  {'---':>8}  {'---':<10}"
                    f.write(row + "\n")

                # Delta vs baseline
                baseline = sharpen_parsed.get('baseline', {})
                if baseline:
                    f.write(f"\n=== IMPROVEMENT vs BASELINE ===\n")
                    f.write(f"{'Segment':<24}  {'Metric':<10}")
                    for lb in labels[1:]:
                        f.write(f"  {lb:>12} ")
                    f.write("\n")
                    for seg in all_segs:
                        b = baseline.get(seg)
                        if not b:
                            continue
                        row_mean = f"{seg:<24}  {'mean%':<10}"
                        row_max = f"{'':<24}  {'avgmax%':<10}"
                        row_mmean = f"{'':<24}  {'mean_mm':<10}"
                        for lb in labels[1:]:
                            d = sharpen_parsed.get(lb, {}).get(seg)
                            if d:
                                dm = d['mean_pct'] - b['mean_pct']
                                dx = d['avgmax_pct'] - b['avgmax_pct']
                                dmm = d['mean_mm'] - b['mean_mm']
                                row_mean += f"  {dm:+12.2f} "
                                row_max += f"  {dx:+12.2f} "
                                row_mmean += f"  {dmm:+12.1f} "
                            else:
                                row_mean += f"  {'n/a':>12} "
                                row_max += f"  {'n/a':>12} "
                                row_mmean += f"  {'n/a':>12} "
                        f.write(row_mean + "\n")
                        f.write(row_max + "\n")
                        f.write(row_mmean + "\n")

        print(f"\n[sharpen] Comparison saved to {sharpen_file}")

        # ── LBS vs DQS COMPARISON ──
        # Use the best sharpen config (aggressive: upper=0.9, lower=0.5) for both modes
        print("\n[lbs-vs-dqs] Setting sharpen to aggressive (upper=0.9, lower=0.5)...")
        driver.execute_script("""
            var su = document.getElementById('sharpenUpperSlider');
            var sl = document.getElementById('sharpenLowerSlider');
            su.value = 0.9; sl.value = 0.5;
            su.dispatchEvent(new Event('input'));
        """)
        time.sleep(1)

        # Drain old logs
        driver.get_log("browser")

        print("[lbs-vs-dqs] Running evaluateLBSvsDQS(8)...")
        lbs_dqs = driver.execute_script("return window.evaluateLBSvsDQS(8);")

        new_logs = driver.get_log("browser")
        all_logs.extend(new_logs)

        if lbs_dqs and lbs_dqs.get('lbs') and lbs_dqs.get('dqs'):
            lbs_r = lbs_dqs['lbs']
            dqs_r = lbs_dqs['dqs']
            lbs_segs = lbs_r.get('_segments', lbs_r)
            dqs_segs = dqs_r.get('_segments', dqs_r)
            lbs_g = lbs_r.get('_global', {})
            dqs_g = dqs_r.get('_global', {})

            # Print summary
            print("\n[lbs-vs-dqs] === GLOBAL METRICS ===")
            print(f"  {'Metric':<24} {'LBS':>12} {'DQS':>12} {'Delta':>12}")
            print(f"  {'inverted_count':<24} {lbs_g.get('inverted_count',0):>12} {dqs_g.get('inverted_count',0):>12} {dqs_g.get('inverted_count',0) - lbs_g.get('inverted_count',0):>+12}")
            print(f"  {'inverted_pct':<24} {lbs_g.get('inverted_pct',0):>11.3f}% {dqs_g.get('inverted_pct',0):>11.3f}% {dqs_g.get('inverted_pct',0) - lbs_g.get('inverted_pct',0):>+11.3f}%")
            print(f"  {'jitter_mean_mm':<24} {lbs_g.get('jitter_mean_mm',0):>12.1f} {dqs_g.get('jitter_mean_mm',0):>12.1f} {dqs_g.get('jitter_mean_mm',0) - lbs_g.get('jitter_mean_mm',0):>+12.1f}")
            print(f"  {'jitter_max_mm':<24} {lbs_g.get('jitter_max_mm',0):>12.1f} {dqs_g.get('jitter_max_mm',0):>12.1f} {dqs_g.get('jitter_max_mm',0) - lbs_g.get('jitter_max_mm',0):>+12.1f}")

            # Per-segment comparison (focus on legs + pelvis)
            LEG_SEGS = ['l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'pelvis']
            all_segs_lbs_dqs = sorted(set(list(lbs_segs.keys()) + list(dqs_segs.keys())) - {'_segments', '_global'})

            print(f"\n  {'Segment':<28} {'LBS mean%':>10} {'DQS mean%':>10} {'Delta':>8}   {'LBS avgmx%':>11} {'DQS avgmx%':>11} {'Delta':>8}   {'LBS tag':<12} {'DQS tag':<12}")
            print("  " + "-" * 130)
            for seg in all_segs_lbs_dqs:
                seg_ascii = seg.replace('\u2192', '->').encode('ascii', 'replace').decode('ascii')
                ld = lbs_segs.get(seg, {})
                dd = dqs_segs.get(seg, {})
                lm = ld.get('mean_pct', 0)
                dm = dd.get('mean_pct', 0)
                lx = ld.get('avgmax_pct', 0)
                dx = dd.get('avgmax_pct', 0)
                lt = ld.get('tag', '---')
                dt = dd.get('tag', '---')
                is_leg = any(k in seg for k in LEG_SEGS)
                marker = ' <-- LEG' if is_leg else ''
                print(f"  {seg_ascii:<28} {lm:>10.2f} {dm:>10.2f} {dm-lm:>+8.2f}   {lx:>11.2f} {dx:>11.2f} {dx-lx:>+8.2f}   {lt:<12} {dt:<12}{marker}")

            # Save to file
            dqs_file = CAGE_WALK_DIR / "lbs_vs_dqs.txt"
            with open(dqs_file, "w", encoding="utf-8") as f:
                f.write(f"=== LBS vs DQS COMPARISON (8 samples, sharpen upper=0.9 lower=0.5) ===\n")
                f.write(f"FBX: {fbx_path.name}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("--- GLOBAL METRICS ---\n")
                f.write(f"  {'Metric':<24} {'LBS':>12} {'DQS':>12} {'Delta':>12}\n")
                f.write(f"  {'inverted_count':<24} {lbs_g.get('inverted_count',0):>12} {dqs_g.get('inverted_count',0):>12} {dqs_g.get('inverted_count',0) - lbs_g.get('inverted_count',0):>+12}\n")
                f.write(f"  {'inverted_pct':<24} {lbs_g.get('inverted_pct',0):>11.3f}% {dqs_g.get('inverted_pct',0):>11.3f}% {dqs_g.get('inverted_pct',0) - lbs_g.get('inverted_pct',0):>+11.3f}%\n")
                f.write(f"  {'jitter_mean_mm':<24} {lbs_g.get('jitter_mean_mm',0):>12.1f} {dqs_g.get('jitter_mean_mm',0):>12.1f} {dqs_g.get('jitter_mean_mm',0) - lbs_g.get('jitter_mean_mm',0):>+12.1f}\n")
                f.write(f"  {'jitter_max_mm':<24} {lbs_g.get('jitter_max_mm',0):>12.1f} {dqs_g.get('jitter_max_mm',0):>12.1f} {dqs_g.get('jitter_max_mm',0) - lbs_g.get('jitter_max_mm',0):>+12.1f}\n")

                f.write(f"\n--- PER-SEGMENT RIGIDITY ---\n")
                f.write(f"  {'Segment':<28} {'LBS mean%':>10} {'DQS mean%':>10} {'Delta':>8}   {'LBS avgmx%':>11} {'DQS avgmx%':>11} {'Delta':>8}   {'LBS mm':>8} {'DQS mm':>8}   {'LBS tag':<12} {'DQS tag':<12}\n")
                f.write("  " + "-" * 155 + "\n")
                for seg in all_segs_lbs_dqs:
                    ld = lbs_segs.get(seg, {})
                    dd = dqs_segs.get(seg, {})
                    lm = ld.get('mean_pct', 0)
                    dm = dd.get('mean_pct', 0)
                    lx = ld.get('avgmax_pct', 0)
                    dx = dd.get('avgmax_pct', 0)
                    lmm = ld.get('mean_mm', 0)
                    dmm = dd.get('mean_mm', 0)
                    lt = ld.get('tag', '---')
                    dt = dd.get('tag', '---')
                    is_leg = any(k in seg for k in LEG_SEGS)
                    marker = ' <-- LEG' if is_leg else ''
                    f.write(f"  {seg:<28} {lm:>10.2f} {dm:>10.2f} {dm-lm:>+8.2f}   {lx:>11.2f} {dx:>11.2f} {dx-lx:>+8.2f}   {lmm:>8.1f} {dmm:>8.1f}   {lt:<12} {dt:<12}{marker}\n")

            print(f"\n[lbs-vs-dqs] Comparison saved to {dqs_file}")
        else:
            print("[lbs-vs-dqs] evaluateLBSvsDQS returned null or incomplete data")

        # ── CIRCUMFERENCE LBS vs DQS ──
        print("\n[circumference] Running evaluateCircumferenceLBSvsDQS(8)...")
        circ_cmp = driver.execute_script("return window.evaluateCircumferenceLBSvsDQS(8);")

        if circ_cmp and circ_cmp.get('lbs') and circ_cmp.get('dqs'):
            circ_lbs = circ_cmp['lbs']
            circ_dqs = circ_cmp['dqs']
            all_labels = sorted(set(list(circ_lbs.keys()) + list(circ_dqs.keys())))

            print(f"\n[circumference] === LBS vs DQS CIRCUMFERENCE (r'/r, 8 samples) ===")
            print(f"  {'Segment':<16} {'LBS mean':>9} {'DQS mean':>9} {'Delta':>7}   {'LBS p5':>7} {'DQS p5':>7} {'Delta':>7}   {'LBS p95':>8} {'DQS p95':>8} {'Delta':>7}   {'LBS flat':>9} {'DQS flat':>9}")
            print("  " + "-" * 130)
            for label in all_labels:
                ld = circ_lbs.get(label, {})
                dd = circ_dqs.get(label, {})
                lm = ld.get('mean', 0); dm = dd.get('mean', 0)
                lp5 = ld.get('p5', 0); dp5 = dd.get('p5', 0)
                lp95 = ld.get('p95', 0); dp95 = dd.get('p95', 0)
                lf = ld.get('flattenScore', 0); df = dd.get('flattenScore', 0)
                print(f"  {label:<16} {lm:>9.3f} {dm:>9.3f} {dm-lm:>+7.3f}   {lp5:>7.3f} {dp5:>7.3f} {dp5-lp5:>+7.3f}   {lp95:>8.3f} {dp95:>8.3f} {dp95-lp95:>+7.3f}   {lf:>+9.4f} {df:>+9.4f}")

            # Save to circumference file
            circ_file = CAGE_WALK_DIR / "circumference_lbs_vs_dqs.txt"
            with open(circ_file, "w", encoding="utf-8") as f:
                f.write(f"=== CIRCUMFERENCE LBS vs DQS (r'/r, 8 samples) ===\n")
                f.write(f"FBX: {fbx_path.name}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"  {'Segment':<16} {'LBS mean':>9} {'DQS mean':>9} {'Delta':>7}   {'LBS p5':>7} {'DQS p5':>7} {'Delta':>7}   {'LBS p95':>8} {'DQS p95':>8} {'Delta':>7}   {'LBS flat':>9} {'DQS flat':>9}\n")
                f.write("  " + "-" * 130 + "\n")
                for label in all_labels:
                    ld = circ_lbs.get(label, {})
                    dd = circ_dqs.get(label, {})
                    lm = ld.get('mean', 0); dm = dd.get('mean', 0)
                    lp5 = ld.get('p5', 0); dp5 = dd.get('p5', 0)
                    lp95 = ld.get('p95', 0); dp95 = dd.get('p95', 0)
                    lf = ld.get('flattenScore', 0); df = dd.get('flattenScore', 0)
                    f.write(f"  {label:<16} {lm:>9.3f} {dm:>9.3f} {dm-lm:>+7.3f}   {lp5:>7.3f} {dp5:>7.3f} {dp5-lp5:>+7.3f}   {lp95:>8.3f} {dp95:>8.3f} {dp95-lp95:>+7.3f}   {lf:>+9.4f} {df:>+9.4f}\n")
            print(f"[circumference] Saved to {circ_file}")
        else:
            print("[circumference] evaluateCircumferenceLBSvsDQS returned null or incomplete data")

        # Reset sliders back to defaults
        driver.execute_script("""
            var su = document.getElementById('sharpenUpperSlider');
            var sl = document.getElementById('sharpenLowerSlider');
            su.value = 0.7; sl.value = 0.3;
            su.dispatchEvent(new Event('input'));
        """)
        time.sleep(0.5)

        # Capture screenshots from multiple angles
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        print("\n[browser] Capturing screenshots...")
        # Turn off strain/bone-error overlays for clean visual
        # Use JS click to avoid element interception by overlapping UI elements
        for cb_id in ["showStrain", "showBoneError", "showFBXGT"]:
            cb = driver.find_element(By.ID, cb_id)
            if cb.is_selected():
                driver.execute_script("arguments[0].click()", cb)
        time.sleep(0.5)

        # Helper: set camera via JS and screenshot
        def take_screenshot(name, js_code=None):
            if js_code:
                driver.execute_script(js_code)
                time.sleep(0.3)
            path = SCREENSHOT_DIR / f"{name}.png"
            driver.save_screenshot(str(path))
            print(f"  -> {path.name}")

        # Default view (whatever the current camera is)
        take_screenshot("01_default")

        # Front view
        take_screenshot("02_front", """
            if (window.camera) {
                camera.position.set(0, -0.2, 1.5);
                camera.lookAt(0, -0.2, 0);
            }
        """)

        # Back view (to see posterior thigh issues)
        take_screenshot("03_back", """
            if (window.camera) {
                camera.position.set(0, -0.2, -1.5);
                camera.lookAt(0, -0.2, 0);
            }
        """)

        # Side view (left)
        take_screenshot("04_side_left", """
            if (window.camera) {
                camera.position.set(-1.5, -0.2, 0);
                camera.lookAt(0, -0.2, 0);
            }
        """)

        # Crotch/thigh close-up from front-below
        take_screenshot("05_crotch_front", """
            if (window.camera) {
                camera.position.set(0, -0.5, 0.8);
                camera.lookAt(0, -0.3, 0);
            }
        """)

        # Crotch from back-below (to see between-legs mesh)
        take_screenshot("06_crotch_back", """
            if (window.camera) {
                camera.position.set(0, -0.5, -0.8);
                camera.lookAt(0, -0.3, 0);
            }
        """)

        # Now with debug regions on (to see region assignments)
        dr_cb = driver.find_element(By.ID, "debugRegions")
        if not dr_cb.is_selected():
            dr_cb.click()
        time.sleep(0.3)

        take_screenshot("07_regions_front", """
            if (window.camera) {
                camera.position.set(0, -0.2, 1.5);
                camera.lookAt(0, -0.2, 0);
            }
        """)

        take_screenshot("08_regions_back", """
            if (window.camera) {
                camera.position.set(0, -0.2, -1.5);
                camera.lookAt(0, -0.2, 0);
            }
        """)

        take_screenshot("09_regions_crotch", """
            if (window.camera) {
                camera.position.set(0, -0.5, 0.8);
                camera.lookAt(0, -0.3, 0);
            }
        """)

        # Turn off debug regions
        dr_cb = driver.find_element(By.ID, "debugRegions")
        if dr_cb.is_selected():
            dr_cb.click()

        print(f"[browser] Captured {len(all_logs)} log entries")

        # Save raw logs
        raw_lines = [f"=== AUTO-CAPTURED DIAGNOSTIC LOGS ===",
                     f"FBX: {fbx_path.name}",
                     f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                     f"Total log entries: {len(all_logs)}",
                     f"{'='*60}\n"]
        for entry in all_logs:
            level = entry.get("level", "INFO")
            msg = entry.get("message", "")
            raw_lines.append(f"[{level}] {msg}")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(raw_lines))
        print(f"[done] Raw logs -> {OUTPUT_FILE}")

        # Parse and save summary
        metrics = parse_metrics(all_logs)
        summary = format_summary(metrics, fbx_path.name)

        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"[done] Summary -> {SUMMARY_FILE}")
        print()
        print(summary.encode('ascii', 'replace').decode('ascii'))

    except Exception as e:
        print(f"[error] {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n[browser] Closing Chrome...")
        driver.quit()
        httpd.shutdown()
        print("[done] Finished.")

if __name__ == "__main__":
    main()
