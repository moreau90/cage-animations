// src/qa/skeleton-qa.ts
var STANDARD_JOINTS = [
  "hips",
  "l_hip",
  "r_hip",
  "l_knee",
  "r_knee",
  "l_ankle",
  "r_ankle",
  "l_toe",
  "r_toe",
  "l_shoulder",
  "r_shoulder",
  "l_elbow",
  "r_elbow",
  "l_wrist",
  "r_wrist",
  "l_mid_knuckle",
  "r_mid_knuckle",
  "neck",
  "head",
  "chest",
  "spine_joint",
  "spine1_joint",
  "spine2_joint",
  "l_collar",
  "r_collar"
];
var BONE_DEFS = [
  ["hips", "l_hip", "L hip offset"],
  ["hips", "r_hip", "R hip offset"],
  ["l_hip", "l_knee", "L thigh"],
  ["r_hip", "r_knee", "R thigh"],
  ["l_knee", "l_ankle", "L shin"],
  ["r_knee", "r_ankle", "R shin"],
  ["l_ankle", "l_toe", "L foot"],
  ["r_ankle", "r_toe", "R foot"],
  ["l_shoulder", "l_elbow", "L upper arm"],
  ["r_shoulder", "r_elbow", "R upper arm"],
  ["l_elbow", "l_wrist", "L forearm"],
  ["r_elbow", "r_wrist", "R forearm"],
  ["l_wrist", "l_mid_knuckle", "L hand"],
  ["r_wrist", "r_mid_knuckle", "R hand"],
  ["hips", "spine_joint", "Pelvis\u2192spine"],
  ["spine_joint", "spine1_joint", "Spine 0\u21921"],
  ["spine1_joint", "spine2_joint", "Spine 1\u21922"],
  ["spine2_joint", "neck", "Spine 2\u2192neck"],
  ["neck", "head", "Neck\u2192head"],
  ["l_collar", "l_shoulder", "L collar"],
  ["r_collar", "r_shoulder", "R collar"]
];
function boneLen(src, a, b) {
  const pa = src[a], pb = src[b];
  if (!pa || !pb) return NaN;
  return Math.sqrt((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2 + (pb[2] - pa[2]) ** 2) * 1e3;
}
function compareSkeletons(ourJoints, refRestPose, anchor, origJoints) {
  const refWorld = {};
  if (refRestPose) {
    refWorld["hips"] = anchor;
    for (const jn of STANDARD_JOINTS) {
      if (jn === "hips") continue;
      const rp = refRestPose[jn];
      if (rp) {
        refWorld[jn] = [rp[0] + anchor[0], rp[1] + anchor[1], rp[2] + anchor[2]];
      }
    }
  }
  const jointComparisons = [];
  let maxDiff = 0, maxDiffName = "";
  for (const jn of STANDARD_JOINTS) {
    const pj = ourJoints[jn] || null;
    const rpj = refWorld[jn] || null;
    let diffMm = 0;
    let status = "OK";
    if (!pj && !rpj) continue;
    if (!pj) {
      status = "MISSING_OURS";
    } else if (!rpj) {
      status = "MISSING_REF";
    } else {
      diffMm = Math.sqrt(
        (pj[0] - rpj[0]) ** 2 + (pj[1] - rpj[1]) ** 2 + (pj[2] - rpj[2]) ** 2
      ) * 1e3;
      if (diffMm > 1) status = "MISMATCH";
      if (diffMm > maxDiff) {
        maxDiff = diffMm;
        maxDiffName = jn;
      }
    }
    jointComparisons.push({ jointName: jn, ourPos: pj, refPos: rpj, diffMm, status });
  }
  const boneLengthComparisons = [];
  for (const [a, b, label] of BONE_DEFS) {
    const pLen = boneLen(ourJoints, a, b);
    const fLen = boneLen(refWorld, a, b);
    const oLen = origJoints ? boneLen(origJoints, a, b) : NaN;
    const ratio = !isNaN(oLen) && !isNaN(fLen) && fLen > 0 ? oLen / fLen : NaN;
    const isMismatch = !isNaN(ratio) && (ratio < 0.9 || ratio > 1.1);
    boneLengthComparisons.push({
      label,
      parent: a,
      child: b,
      ourLength: pLen,
      refLength: fLen,
      origLength: oLen,
      ratio,
      isMismatch
    });
  }
  const totalLegOurs = boneLen(ourJoints, "l_hip", "l_knee") + boneLen(ourJoints, "l_knee", "l_ankle");
  const totalLegRef = boneLen(refWorld, "l_hip", "l_knee") + boneLen(refWorld, "l_knee", "l_ankle");
  const totalArmOurs = boneLen(ourJoints, "l_shoulder", "l_elbow") + boneLen(ourJoints, "l_elbow", "l_wrist");
  const totalArmRef = boneLen(refWorld, "l_shoulder", "l_elbow") + boneLen(refWorld, "l_elbow", "l_wrist");
  return {
    jointComparisons,
    boneLengthComparisons,
    maxJointDiffMm: maxDiff,
    maxJointDiffName: maxDiffName,
    totalLegOurs,
    totalLegRef,
    totalArmOurs,
    totalArmRef
  };
}
function checkSymmetry(joints) {
  const PAIRS = [
    ["l_hip", "l_knee", "r_hip", "r_knee", "Thigh"],
    ["l_knee", "l_ankle", "r_knee", "r_ankle", "Shin"],
    ["l_ankle", "l_toe", "r_ankle", "r_toe", "Foot"],
    ["l_shoulder", "l_elbow", "r_shoulder", "r_elbow", "Upper arm"],
    ["l_elbow", "l_wrist", "r_elbow", "r_wrist", "Forearm"]
  ];
  return PAIRS.map(([lp, lc, rp, rc, label]) => {
    const leftMm = boneLen(joints, lp, lc);
    const rightMm = boneLen(joints, rp, rc);
    const ratio = !isNaN(leftMm) && !isNaN(rightMm) && rightMm > 0 ? leftMm / rightMm : NaN;
    const isMismatch = !isNaN(ratio) && (ratio < 0.95 || ratio > 1.05);
    return { label, leftMm, rightMm, ratio, isMismatch };
  });
}
export {
  checkSymmetry,
  compareSkeletons
};
