import type { ScoreEntry } from '../types/job-payloads.js';

/** Threshold-based decision engine (Phase 1 — no AI calls) */
export function decide(scores: ScoreEntry[]): { decision: 'accept' | 'reject' | 'review'; reasons: string[] } {
  const reasons: string[] = [];
  let reject = false;
  let review = false;

  const scoreMap = new Map<string, number>();
  for (const s of scores) {
    scoreMap.set(s.metric, s.value);
  }

  // Skeleton thresholds
  const maxJointDiff = scoreMap.get('max_joint_diff_mm');
  if (maxJointDiff !== undefined) {
    if (maxJointDiff > 20) {
      reject = true;
      reasons.push(`max_joint_diff_mm=${maxJointDiff.toFixed(1)}mm exceeds 20mm threshold`);
    } else if (maxJointDiff > 10) {
      review = true;
      reasons.push(`max_joint_diff_mm=${maxJointDiff.toFixed(1)}mm exceeds 10mm review threshold`);
    }
  }

  const symmetryMismatches = scoreMap.get('symmetry_mismatches');
  if (symmetryMismatches !== undefined && symmetryMismatches > 2) {
    review = true;
    reasons.push(`${symmetryMismatches} symmetry mismatches`);
  }

  // Deformation thresholds
  const avgStrain = scoreMap.get('avg_strain');
  if (avgStrain !== undefined) {
    if (avgStrain > 0.15) {
      reject = true;
      reasons.push(`avg_strain=${(avgStrain * 100).toFixed(1)}% exceeds 15% threshold`);
    } else if (avgStrain > 0.08) {
      review = true;
      reasons.push(`avg_strain=${(avgStrain * 100).toFixed(1)}% exceeds 8% review threshold`);
    }
  }

  const highStrainCount = scoreMap.get('high_strain_count');
  if (highStrainCount !== undefined && highStrainCount > 50) {
    review = true;
    reasons.push(`${highStrainCount} vertices with >30% strain`);
  }

  const rigidityBad = scoreMap.get('rigidity_bad_segments');
  if (rigidityBad !== undefined && rigidityBad > 3) {
    reject = true;
    reasons.push(`${rigidityBad} segments with rigidity issues`);
  }

  const circumIssues = scoreMap.get('circumference_issues');
  if (circumIssues !== undefined && circumIssues > 4) {
    review = true;
    reasons.push(`${circumIssues} circumference issues`);
  }

  // Finger alignment thresholds — prefer per-joint metric over chain average
  const worstFingerDev = scoreMap.get('worst_joint_finger_deviation_mm')
    ?? scoreMap.get('worst_centerline_deviation_mm');
  if (worstFingerDev !== undefined) {
    if (worstFingerDev > 5) {
      reject = true;
      reasons.push(`worst finger joint deviation=${worstFingerDev.toFixed(1)}mm exceeds 5mm threshold`);
    } else if (worstFingerDev > 3) {
      review = true;
      reasons.push(`worst finger joint deviation=${worstFingerDev.toFixed(1)}mm exceeds 3mm review threshold`);
    }
  }

  const leftSpacingCV = scoreMap.get('left_spacing_cv');
  const rightSpacingCV = scoreMap.get('right_spacing_cv');
  for (const [label, cv] of [['left', leftSpacingCV], ['right', rightSpacingCV]] as const) {
    if (cv !== undefined && cv > 0.5) {
      review = true;
      reasons.push(`${label}_spacing_cv=${cv.toFixed(2)} — uneven finger spacing`);
    }
  }

  // Joint centering thresholds — only limb joints where centroid IS the correct target.
  // Hips, shoulders, spine, neck are handled by the visual worker (not scorable).
  const SCORED_CENTERING_JOINTS = [
    'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
    'l_knee', 'r_knee', 'l_ankle', 'r_ankle',
  ];
  let worstCentering = 0;
  let sumCentering = 0;
  let countCentering = 0;
  for (const joint of SCORED_CENTERING_JOINTS) {
    const val = scoreMap.get(`centering_${joint}_total_mm`);
    if (val !== undefined) {
      if (val > worstCentering) worstCentering = val;
      sumCentering += val;
      countCentering++;
    }
  }
  const avgCentering = countCentering > 0 ? sumCentering / countCentering : 0;

  if (worstCentering > 15) {
    reject = true;
    reasons.push(`worst centering=${worstCentering.toFixed(1)}mm exceeds 15mm threshold`);
  } else if (worstCentering > 8) {
    review = true;
    reasons.push(`worst centering=${worstCentering.toFixed(1)}mm exceeds 8mm review threshold`);
  }

  if (avgCentering > 5) {
    review = true;
    reasons.push(`avg centering=${avgCentering.toFixed(1)}mm exceeds 5mm review threshold`);
  }

  // ─── Centering-converged accept path ───
  if (!reject && countCentering > 0 && worstCentering < 3) {
    reasons.push(`worst centering=${worstCentering.toFixed(1)}mm < 3mm — centering converged`);
    if (review) {
      reasons.push('Centering convergence overrides review-level warnings');
    }
    return { decision: 'accept', reasons };
  }

  if (reasons.length === 0) {
    reasons.push('All metrics within acceptable thresholds');
  }

  const decision = reject ? 'reject' : review ? 'review' : 'accept';
  return { decision, reasons };
}
