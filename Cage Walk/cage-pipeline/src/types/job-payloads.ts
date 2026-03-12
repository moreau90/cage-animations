import type { Vec3 } from 'cage-core';

/** Job names used as BullMQ queue/job identifiers */
export const JOB_NAMES = {
  SKELETON_QA: 'skeleton-qa',
  DEFORMATION_QA: 'deformation-qa',
  FINGER_ALIGNMENT: 'finger-alignment',
  JOINT_CENTERING: 'joint-centering',
  RENDER_PREVIEW: 'render-preview',
  PROCESSING: 'processing',
  GENERATION: 'generation',
  PATTERN_ANALYSIS: 'pattern-analysis',
  ORCHESTRATOR: 'orchestrator',
  LOOP_CONTROLLER: 'loop-controller',
} as const;

/** Payload for skeleton-qa jobs */
export interface SkeletonQAPayload {
  runId: string;
  /** Path to placed_joints.json artifact */
  jointsPath: string;
  /** Path to reference skeleton artifact (optional) */
  refSkeletonPath?: string;
}

/** Payload for deformation-qa jobs */
export interface DeformationQAPayload {
  runId: string;
  /** Path to deformed positions Float32Array binary */
  deformedPosPath: string;
  /** Path to rest positions Float32Array binary */
  restPosPath: string;
  /** Path to mesh bone weights JSON */
  boneWeightsPath: string;
  /** Path to joints JSON */
  jointsPath: string;
  /** Path to bone transforms JSON */
  boneTransformsPath: string;
  /** Path to adjacency data JSON */
  adjacencyPath: string;
  /** Alpha blend factor */
  alpha: number;
}

/** Payload for orchestrator jobs */
export interface OrchestratorPayload {
  runId: string;
}

/** Result from skeleton-qa worker */
export interface SkeletonQAResult {
  runId: string;
  maxJointDiffMm: number;
  maxJointDiffName: string;
  symmetryMismatches: number;
  totalScores: number;
}

/** Result from deformation-qa worker */
export interface DeformationQAResult {
  runId: string;
  avgStrain: number;
  highStrainCount: number;
  rigidityBadSegments: number;
  circumferenceIssues: number;
  totalScores: number;
}

/** Result from orchestrator */
export interface OrchestratorResult {
  runId: string;
  decision: 'accept' | 'reject' | 'review';
  reasons: string[];
}

// ─── Finger Alignment ───

/** Payload for finger-alignment jobs */
export interface FingerAlignmentPayload {
  runId: string;
  /** Path to mesh.glb file */
  glbPath: string;
  /** Path to placed_joints.json artifact */
  jointsPath: string;
}

/** Per-joint deviation within a finger chain */
export interface FingerJointDeviation {
  jointName: string;
  /** Signed Y deviation at this joint's X position: mesh_center - skeleton (mm) */
  deviationYMm: number;
  /** Signed Z deviation at this joint's X position: mesh_center - skeleton (mm) */
  deviationZMm: number;
  /** Total deviation magnitude at this joint (mm) */
  deviationMm: number;
  /** Number of cross-sections averaged near this joint */
  sampleCount: number;
}

/** Per-finger centering result */
export interface FingerCenterlineScore {
  finger: string;
  hand: 'left' | 'right';
  /** Average deviation of skeleton centerline from mesh tube center (mm) */
  deviationMm: number;
  /** Signed average Y deviation: mesh_center - skeleton (mm) */
  deviationYMm: number;
  /** Signed average Z deviation: mesh_center - skeleton (mm) */
  deviationZMm: number;
  /** Number of cross-sections sampled */
  sampleCount: number;
  /** Per-joint deviations (one per joint in the chain) */
  perJoint: FingerJointDeviation[];
}

/** Result from finger-alignment worker */
export interface FingerAlignmentResult {
  runId: string;
  /** Per-finger centerline deviation scores */
  perFinger: FingerCenterlineScore[];
  /** Max deviation across all fingers (mm) */
  worstDeviationMm: number;
  worstDeviationFinger: string;
  /** Spacing uniformity per hand: std(inter-finger gaps) / mean(gaps) — lower=better */
  leftSpacingCV: number;
  rightSpacingCV: number;
  /** Boundary fit quality per hand (R² of boundary regression lines) */
  leftBoundaryFitR2: number;
  rightBoundaryFitR2: number;
  totalScores: number;
}

// ─── Joint Centering ───

/** Payload for joint-centering jobs */
export interface JointCenteringPayload {
  runId: string;
  /** Path to mesh.glb file */
  glbPath: string;
  /** Path to placed_joints.json artifact */
  jointsPath: string;
}

/** Per-joint centering measurement */
export interface JointCenteringScore {
  joint: string;
  /** Total offset from joint to mesh center (mm) */
  totalOffsetMm: number;
  /** Offset along bone axis (mm) */
  alongBoneMm: number;
  /** Lateral offset perpendicular to bone (mm) */
  lateralMm: number;
  /** Depth offset perpendicular to bone (mm) */
  depthMm: number;
  /** World-space offset [X, Y, Z] in mm */
  offsetWorldMm: [number, number, number];
  /** Mesh centroid at this slice [X, Y, Z] in meters */
  meshCentroid: [number, number, number];
  /** Number of intersection points in the cross-section */
  slicePointCount: number;
  /** If set, this joint used an anatomical target instead of centroid */
  anatomicalNote?: string;
}

/** Result from joint-centering worker */
export interface JointCenteringResult {
  runId: string;
  /** Per-joint centering measurements */
  perJoint: JointCenteringScore[];
  /** Worst centering offset (mm) */
  worstOffsetMm: number;
  worstOffsetJoint: string;
  /** Average centering offset across all joints (mm) */
  avgOffsetMm: number;
  totalScores: number;
}

// ─── Processing ───

/** Payload for processing jobs (FK/LBS pipeline) */
export interface ProcessingPayload {
  runId: string;
  glbPath: string;
  jointsPath: string;
  keyframesPath: string;
  boneWeightsPath?: string;
  alpha?: number;
}

/** Result from processing worker */
export interface ProcessingResult {
  runId: string;
  framesProcessed: number;
  worstFrameIdx: number;
  artifactPaths: {
    deformedPos: string;
    restPos: string;
    boneWeights: string;
    boneTransforms: string;
    adjacency: string;
  };
}

// ─── Generation ───

/** Payload for generation jobs */
export interface GenerationPayload {
  runId: string;
  strategy: string;
  mutationConfig: Record<string, unknown>;
  candidateCount: number;
}

/** Result from generation worker */
export interface GenerationResult {
  runId: string;
  childRunIds: string[];
  strategy: string;
  candidateCount: number;
}

// ─── Render Preview ───

/** Payload for render-preview jobs */
export interface RenderPreviewPayload {
  runId: string;
  angles?: Array<{ name: string; azimuth: number; elevation: number }>;
  extractRuntime?: boolean;
}

/** Result from render-preview worker */
export interface RenderPreviewResult {
  runId: string;
  screenshotPaths: string[];
  runtimeDataPath?: string;
}

// ─── Pattern Analysis ───

/** Payload for pattern-analysis jobs */
export interface PatternAnalysisPayload {
  runId: string;
  familyRootId?: string;
}

/** Result from pattern-analysis worker */
export interface PatternAnalysisResult {
  runId: string;
  insightsWritten: number;
  topCorrelations: Array<{ parameter: string; metric: string; correlation: number }>;
}

// ─── Loop Controller ───

/** Payload for loop-controller jobs */
export interface LoopControllerPayload {
  /** The search run ID (top-level parent) */
  searchId: string;
  /** The parent run whose children just completed */
  parentRunId: string;
  /** Current round number (1-based) */
  round: number;
  /** Maximum rounds allowed */
  maxRounds: number;
  /** Stop after N rounds with no improvement */
  staleRounds: number;
  /** Current mutation strategy name */
  baseStrategy: string;
  /** Current mutation config */
  baseMutationConfig: Record<string, unknown>;
  /** Number of candidates per round */
  candidateCount: number;
}

/** Result from loop-controller worker */
export interface LoopControllerResult {
  runId: string;
  /** Whether the loop continues or stops */
  action: 'continue' | 'stop';
  /** Why the loop stopped (if action=stop) */
  stopReason?: 'accepted' | 'budget_exhausted' | 'plateau' | 'ai_satisfied';
  /** Best child run ID from this round */
  bestChildId?: string;
  /** Best composite score from this round */
  bestScore?: number;
  /** Next round number (if continuing) */
  nextRound?: number;
}

/** A single score entry for the scores table */
export interface ScoreEntry {
  run_id: string;
  worker: string;
  metric: string;
  value: number;
  confidence: number;
  unit: string;
}
