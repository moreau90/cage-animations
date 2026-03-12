import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);
const runId = process.argv[2];

if (!runId) {
  // Show latest run
  const { data: runs } = await sb.from('runs').select('id,goal,status,decision,created_at').order('created_at', { ascending: false }).limit(5);
  console.log('=== Recent Runs ===');
  for (const r of runs) {
    console.log(`  ${r.id.slice(0,8)}  ${r.status.padEnd(10)} ${r.decision || '-'.padEnd(8)}  ${r.goal}`);
  }
  console.log('\nUsage: node show-results.mjs <run-id>');
  process.exit(0);
}

// Get run
const { data: run } = await sb.from('runs').select('*').eq('id', runId).single();
console.log(`\n=== Run: ${run.goal} ===`);
console.log(`Status: ${run.status}  Decision: ${run.decision || 'pending'}`);
if (run.decision_reasons) console.log(`Reasons: ${run.decision_reasons.join(', ')}`);

// Get scores
const { data: scores } = await sb.from('scores').select('*').eq('run_id', runId);
console.log(`\n=== Scores (${scores.length}) ===`);
for (const s of scores) {
  console.log(`  ${s.metric.padEnd(30)} ${s.value.toFixed(3)} ${s.unit}`);
}

// Group scores by worker
const scoresByWorker = {};
for (const s of scores) {
  const w = s.worker || 'unknown';
  if (!scoresByWorker[w]) scoresByWorker[w] = [];
  scoresByWorker[w].push(s);
}

// Display scores grouped by worker
for (const [worker, wScores] of Object.entries(scoresByWorker)) {
  console.log(`\n--- ${worker} ---`);
  for (const s of wScores) {
    const val = typeof s.value === 'number' ? s.value.toFixed(3) : String(s.value);
    console.log(`  ${s.metric.padEnd(35)} ${val.padStart(10)} ${s.unit}`);
  }
}

// Show finger alignment details
const fingerScores = scores.filter(s => s.worker === 'finger-alignment');
if (fingerScores.length > 0) {
  console.log('\n=== Finger Alignment ===');
  const deviation = fingerScores.filter(s => s.metric.includes('centerline_deviation'));
  const spacing = fingerScores.filter(s => s.metric.includes('spacing'));
  const boundary = fingerScores.filter(s => s.metric.includes('boundary'));

  if (deviation.length > 0) {
    console.log('  Centerline Deviation (mm):');
    for (const s of deviation) {
      const label = s.metric.replace('centerline_deviation_mm_', '');
      const flag = s.value > 5 ? ' REJECT' : s.value > 3 ? ' REVIEW' : '';
      console.log(`    ${label.padEnd(20)} ${s.value.toFixed(2)}mm${flag}`);
    }
  }
  if (spacing.length > 0) {
    for (const s of spacing) {
      const flag = s.value > 0.5 ? ' REVIEW' : '';
      console.log(`  ${s.metric.padEnd(35)} ${s.value.toFixed(3)}${flag}`);
    }
  }
  if (boundary.length > 0) {
    for (const s of boundary) {
      console.log(`  ${s.metric.padEnd(35)} ${s.value.toFixed(3)}`);
    }
  }
}

// Show joint centering details
const centeringScores = scores.filter(s => s.worker === 'joint-centering');
if (centeringScores.length > 0) {
  console.log('\n=== Joint Centering ===');
  // Group by joint name
  const joints = {};
  for (const s of centeringScores) {
    const parts = s.metric.split('_');
    // Metrics like centering_offset_mm_l_knee, centering_along_bone_mm_l_knee
    // Extract joint name (last 1-2 segments) and metric type
    let jointName, metricType;
    if (s.metric.startsWith('centering_offset_mm_')) {
      jointName = s.metric.replace('centering_offset_mm_', '');
      metricType = 'total';
    } else if (s.metric.startsWith('centering_along_bone_mm_')) {
      jointName = s.metric.replace('centering_along_bone_mm_', '');
      metricType = 'along';
    } else if (s.metric.startsWith('centering_lateral_mm_')) {
      jointName = s.metric.replace('centering_lateral_mm_', '');
      metricType = 'lateral';
    } else if (s.metric.startsWith('centering_depth_mm_')) {
      jointName = s.metric.replace('centering_depth_mm_', '');
      metricType = 'depth';
    } else if (s.metric.includes('worst_centering') || s.metric.includes('avg_centering')) {
      console.log(`  ${s.metric.padEnd(35)} ${s.value.toFixed(2)}mm`);
      continue;
    } else {
      continue;
    }
    if (!joints[jointName]) joints[jointName] = {};
    joints[jointName][metricType] = s.value;
  }

  if (Object.keys(joints).length > 0) {
    console.log('  Joint'.padEnd(20) + 'Total'.padStart(8) + 'Along'.padStart(8) + 'Lateral'.padStart(8) + 'Depth'.padStart(8));
    console.log('  ' + '-'.repeat(50));
    for (const [name, m] of Object.entries(joints).sort((a, b) => (b[1].total || 0) - (a[1].total || 0))) {
      const flag = (m.total || 0) > 15 ? ' REJECT' : (m.total || 0) > 8 ? ' REVIEW' : '';
      console.log(
        `  ${name.padEnd(18)}` +
        `${(m.total?.toFixed(1) || '-').padStart(8)}` +
        `${(m.along?.toFixed(1) || '-').padStart(8)}` +
        `${(m.lateral?.toFixed(1) || '-').padStart(8)}` +
        `${(m.depth?.toFixed(1) || '-').padStart(8)}` +
        flag
      );
    }
  }
}

// Get diagnostics
const { data: diags } = await sb.from('diagnostics').select('*').eq('run_id', runId);
for (const d of diags) {
  if (d.diagnostic_type === 'skeleton_comparison') {
    const p = d.payload;
    console.log('\n=== Symmetry ===');
    for (const s of p.symmetryResults) {
      const ratio = (s.ratio == null || isNaN(s.ratio)) ? 'N/A' : s.ratio.toFixed(3);
      const left = (s.leftMm == null || isNaN(s.leftMm)) ? 'N/A' : s.leftMm.toFixed(1) + 'mm';
      const right = (s.rightMm == null || isNaN(s.rightMm)) ? 'N/A' : s.rightMm.toFixed(1) + 'mm';
      console.log(`  ${s.label.padEnd(12)} L=${left.padEnd(10)} R=${right.padEnd(10)} ratio=${ratio.toString().padEnd(6)} ${s.isMismatch ? 'MISMATCH' : 'OK'}`);
    }
    console.log('\n=== Bone Lengths ===');
    for (const b of p.boneLengthComparisons) {
      if (b.ourLength != null && !isNaN(b.ourLength)) {
        console.log(`  ${b.label.padEnd(20)} ${b.ourLength.toFixed(1)}mm`);
      }
    }
  }
  if (d.diagnostic_type === 'decision') {
    console.log('\n=== Orchestrator Decision ===');
    console.log(`  ${d.payload.decision}: ${d.payload.reasons.join(', ')}`);
  }
}

// Show child runs (mutation families)
const { data: children } = await sb.from('runs').select('id,goal,status,decision,created_at').eq('parent_run_id', runId).order('created_at');
if (children && children.length > 0) {
  console.log(`\n=== Child Runs (${children.length}) ===`);
  for (const c of children) {
    console.log(`  ${c.id.slice(0,8)}  ${c.status.padEnd(10)} ${(c.decision || '-').padEnd(8)}  ${c.goal}`);
  }
}

// ─── Search/Loop Results ───
// If this run has loop config, show round-by-round progression
const { data: cfgRow } = await sb.from('run_configs').select('config').eq('run_id', runId).single();
const loopConfig = cfgRow?.config?.loop;
if (loopConfig) {
  console.log('\n╔═══════════════════════════════════════════════════╗');
  console.log('║            OPTIMIZATION SEARCH RESULTS            ║');
  console.log('╚═══════════════════════════════════════════════════╝');
  console.log(`  Strategy:         ${loopConfig.baseStrategy}`);
  console.log(`  Max rounds:       ${loopConfig.maxRounds}`);
  console.log(`  Stale rounds:     ${loopConfig.staleRounds}`);
  console.log(`  Candidates/round: ${loopConfig.candidateCount}`);

  // Round scores
  const { data: roundScores } = await sb
    .from('scores')
    .select('metric, value')
    .eq('run_id', runId)
    .eq('worker', 'loop-controller')
    .like('metric', 'round_%_best_composite')
    .order('metric', { ascending: true });

  if (roundScores && roundScores.length > 0) {
    console.log('\n  ┌─────────┬──────────────────┐');
    console.log('  │  Round  │  Best Composite   │');
    console.log('  ├─────────┼──────────────────┤');

    let bestScore = Infinity;
    let bestRound = 0;
    for (const rs of roundScores) {
      const roundNum = rs.metric.match(/round_(\d+)/)?.[1] || '?';
      const improved = rs.value < bestScore - 0.001;
      if (improved) {
        bestScore = rs.value;
        bestRound = parseInt(roundNum);
      }
      const marker = improved ? ' *' : '  ';
      console.log(`  │   ${roundNum.padStart(3)}   │  ${rs.value.toFixed(6).padStart(12)}${marker}  │`);
    }

    console.log('  └─────────┴──────────────────┘');
    console.log(`  (* = improvement over previous best)`);
    console.log(`\n  Overall best: ${bestScore.toFixed(6)} (round ${bestRound})`);
  }

  // Loop controller decisions
  const { data: lcDiags } = await sb
    .from('diagnostics')
    .select('payload, created_at')
    .eq('run_id', runId)
    .eq('worker', 'loop-controller')
    .eq('diagnostic_type', 'loop_decision')
    .order('created_at', { ascending: true });

  if (lcDiags && lcDiags.length > 0) {
    console.log(`\n  ─── Loop Decisions (${lcDiags.length}) ───`);
    for (const d of lcDiags) {
      const p = d.payload;
      const ts = new Date(d.created_at).toLocaleTimeString();
      if (p.action === 'stop') {
        console.log(`  [${ts}] STOP round ${p.round}: ${p.reason}`);
        if (p.bestChildId) console.log(`           Winner: ${p.bestChildId.slice(0,8)}... (score: ${p.bestScore?.toFixed(6) || '?'})`);
      } else {
        console.log(`  [${ts}] CONTINUE round ${p.round} → ${p.nextRound}`);
        console.log(`           Best: ${p.bestChildId?.slice(0,8)}... (score: ${p.bestScore?.toFixed(6) || '?'})`);
        if (p.reasoning) {
          for (const r of p.reasoning) {
            console.log(`           - ${r}`);
          }
        }
      }
    }
  }
}

// Show lessons learned for this run
const { data: lessons } = await sb.from('lessons_learned').select('*').contains('supporting_runs', [runId]);
if (lessons && lessons.length > 0) {
  console.log(`\n=== Lessons Learned ===`);
  for (const l of lessons) {
    const conf = l.confidence != null ? ` (confidence: ${l.confidence.toFixed(2)})` : '';
    console.log(`  ${l.pattern || l.category}${conf}`);
    if (l.insight) console.log(`    ${l.insight}`);
  }
}

process.exit(0);
