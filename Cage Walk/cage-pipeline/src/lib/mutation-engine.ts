/**
 * Mutation Engine — generates parameter variations for optimization search.
 *
 * Strategies:
 * - joint_jitter: perturb joint positions by ±N mm
 * - sharpening_sweep: sweep sharpening intensity values
 * - alpha_sweep: sweep LBS alpha blend factor
 * - finger_z_offset: shift finger Z positions
 * - ground_correction: adjust ground-level offset
 * - directed_correction: apply specific per-joint offset vectors from AI strategist
 * - centering_correction: negate QA centering offsets with damping (algorithmic)
 */
import type { Vec3 } from 'cage-core';

/** Seeded PRNG (mulberry32) for reproducible mutations */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export interface MutationConfig {
  strategy: string;
  /** Number of candidates to generate */
  count: number;
  /** PRNG seed for reproducibility */
  seed?: number;
  /** Strategy-specific parameters */
  params: Record<string, unknown>;
}

export interface MutatedCandidate {
  /** Mutated joint positions */
  joints: { [name: string]: Vec3 };
  /** Mutated pipeline config */
  config: Record<string, unknown>;
  /** Description of what was mutated */
  label: string;
  /** Mutation details for tracking */
  mutations: Record<string, unknown>;
}

/**
 * Generate mutated candidates from a base configuration.
 */
export function generateMutations(
  baseJoints: { [name: string]: Vec3 },
  baseConfig: Record<string, unknown>,
  mutation: MutationConfig,
): MutatedCandidate[] {
  const rng = mulberry32(mutation.seed ?? Date.now());
  const candidates: MutatedCandidate[] = [];

  switch (mutation.strategy) {
    case 'joint_jitter':
      candidates.push(...jointJitter(baseJoints, baseConfig, mutation, rng));
      break;
    case 'sharpening_sweep':
      candidates.push(...sharpeningSweep(baseJoints, baseConfig, mutation));
      break;
    case 'alpha_sweep':
      candidates.push(...alphaSweep(baseJoints, baseConfig, mutation));
      break;
    case 'finger_z_offset':
      candidates.push(...fingerZOffset(baseJoints, baseConfig, mutation, rng));
      break;
    case 'ground_correction':
      candidates.push(...groundCorrectionSweep(baseJoints, baseConfig, mutation));
      break;
    case 'directed_correction':
      candidates.push(...directedCorrection(baseJoints, baseConfig, mutation));
      break;
    case 'centering_correction':
      candidates.push(...centeringCorrection(baseJoints, baseConfig, mutation));
      break;
    default:
      throw new Error(`Unknown mutation strategy: ${mutation.strategy}`);
  }

  return candidates;
}

/** Perturb joint positions by random offsets within ±magnitude mm */
function jointJitter(
  baseJoints: { [name: string]: Vec3 },
  baseConfig: Record<string, unknown>,
  mutation: MutationConfig,
  rng: () => number,
): MutatedCandidate[] {
  const magnitude = (mutation.params.magnitude_mm as number ?? 2) / 1000; // convert mm to meters
  const jointFilter = mutation.params.joints as string[] | undefined;
  const count = mutation.count;
  const candidates: MutatedCandidate[] = [];

  for (let i = 0; i < count; i++) {
    const newJoints: { [name: string]: Vec3 } = {};
    const deltas: { [name: string]: Vec3 } = {};

    for (const [name, pos] of Object.entries(baseJoints)) {
      if (jointFilter && !jointFilter.includes(name)) {
        newJoints[name] = [...pos] as Vec3;
        continue;
      }
      const dx = (rng() * 2 - 1) * magnitude;
      const dy = (rng() * 2 - 1) * magnitude;
      const dz = (rng() * 2 - 1) * magnitude;
      newJoints[name] = [pos[0] + dx, pos[1] + dy, pos[2] + dz];
      deltas[name] = [dx * 1000, dy * 1000, dz * 1000]; // store in mm
    }

    candidates.push({
      joints: newJoints,
      config: { ...baseConfig },
      label: `jitter_${magnitude * 1000}mm_#${i}`,
      mutations: { strategy: 'joint_jitter', magnitude_mm: magnitude * 1000, deltas },
    });
  }

  return candidates;
}

/** Sweep sharpening intensity values */
function sharpeningSweep(
  baseJoints: { [name: string]: Vec3 },
  baseConfig: Record<string, unknown>,
  mutation: MutationConfig,
): MutatedCandidate[] {
  const min = mutation.params.min as number ?? 0;
  const max = mutation.params.max as number ?? 5;
  const count = mutation.count;
  const candidates: MutatedCandidate[] = [];

  for (let i = 0; i < count; i++) {
    const value = count > 1 ? min + (max - min) * (i / (count - 1)) : (min + max) / 2;
    candidates.push({
      joints: { ...baseJoints },
      config: { ...baseConfig, sharpeningIntensity: value },
      label: `sharpen_${value.toFixed(2)}`,
      mutations: { strategy: 'sharpening_sweep', sharpeningIntensity: value },
    });
  }

  return candidates;
}

/** Sweep alpha blend factor */
function alphaSweep(
  baseJoints: { [name: string]: Vec3 },
  baseConfig: Record<string, unknown>,
  mutation: MutationConfig,
): MutatedCandidate[] {
  const min = mutation.params.min as number ?? 0.5;
  const max = mutation.params.max as number ?? 1.0;
  const count = mutation.count;
  const candidates: MutatedCandidate[] = [];

  for (let i = 0; i < count; i++) {
    const value = count > 1 ? min + (max - min) * (i / (count - 1)) : (min + max) / 2;
    candidates.push({
      joints: { ...baseJoints },
      config: { ...baseConfig, alpha: value },
      label: `alpha_${value.toFixed(3)}`,
      mutations: { strategy: 'alpha_sweep', alpha: value },
    });
  }

  return candidates;
}

/** Shift finger Z positions by random offsets */
function fingerZOffset(
  baseJoints: { [name: string]: Vec3 },
  baseConfig: Record<string, unknown>,
  mutation: MutationConfig,
  rng: () => number,
): MutatedCandidate[] {
  const magnitude = (mutation.params.magnitude_mm as number ?? 1) / 1000;
  const count = mutation.count;
  const candidates: MutatedCandidate[] = [];

  const fingerPrefixes = [
    'l_index', 'l_mid', 'l_middle', 'l_ring', 'l_pinky', 'l_thumb',
    'r_index', 'r_mid', 'r_middle', 'r_ring', 'r_pinky', 'r_thumb',
  ];

  for (let i = 0; i < count; i++) {
    const newJoints: { [name: string]: Vec3 } = {};
    const deltas: { [name: string]: number } = {};

    for (const [name, pos] of Object.entries(baseJoints)) {
      const isFinger = fingerPrefixes.some(p => name.startsWith(p));
      if (isFinger) {
        const dz = (rng() * 2 - 1) * magnitude;
        newJoints[name] = [pos[0], pos[1], pos[2] + dz];
        deltas[name] = dz * 1000;
      } else {
        newJoints[name] = [...pos] as Vec3;
      }
    }

    candidates.push({
      joints: newJoints,
      config: { ...baseConfig },
      label: `finger_z_${magnitude * 1000}mm_#${i}`,
      mutations: { strategy: 'finger_z_offset', magnitude_mm: magnitude * 1000, deltas },
    });
  }

  return candidates;
}

/**
 * Apply specific per-joint correction vectors from AI strategist.
 * Generates candidates with different blend factors [0.5, 0.7, 0.85, 1.0].
 * corrections = { joint_name: [dx_mm, dy_mm, dz_mm], ... }
 */
function directedCorrection(
  baseJoints: { [name: string]: Vec3 },
  baseConfig: Record<string, unknown>,
  mutation: MutationConfig,
): MutatedCandidate[] {
  const corrections = mutation.params.corrections as { [joint: string]: [number, number, number] } | undefined;
  if (!corrections || Object.keys(corrections).length === 0) {
    // No corrections provided — return a single unchanged candidate
    return [{
      joints: { ...baseJoints },
      config: { ...baseConfig },
      label: 'directed_no_corrections',
      mutations: { strategy: 'directed_correction', note: 'no corrections provided' },
    }];
  }

  const blendFactors = [0.5, 0.7, 0.85, 1.0];
  const count = Math.min(mutation.count, blendFactors.length);
  const candidates: MutatedCandidate[] = [];

  for (let i = 0; i < count; i++) {
    const blend = blendFactors[i];
    const newJoints: { [name: string]: Vec3 } = {};
    const appliedDeltas: { [name: string]: Vec3 } = {};

    for (const [name, pos] of Object.entries(baseJoints)) {
      const corr = corrections[name];
      if (corr) {
        // corrections are in mm, convert to meters
        const dx = corr[0] * blend / 1000;
        const dy = corr[1] * blend / 1000;
        const dz = corr[2] * blend / 1000;
        newJoints[name] = [pos[0] + dx, pos[1] + dy, pos[2] + dz];
        appliedDeltas[name] = [corr[0] * blend, corr[1] * blend, corr[2] * blend];
      } else {
        newJoints[name] = [...pos] as Vec3;
      }
    }

    const pct = Math.round(blend * 100);
    candidates.push({
      joints: newJoints,
      config: { ...baseConfig },
      label: `directed_blend${pct}pct`,
      mutations: {
        strategy: 'directed_correction',
        blend,
        corrections,
        appliedDeltas,
      },
    });
  }

  return candidates;
}

/**
 * Algorithmic centering correction: apply QA centering offsets with damping.
 * offset = centroid - joint, so ADD to move joint toward center.
 * centering_offsets = { joint_name: { x_mm, y_mm, z_mm } }
 * finger_deviations = { finger_name: { y_mm, z_mm } }
 */
function centeringCorrection(
  baseJoints: { [name: string]: Vec3 },
  baseConfig: Record<string, unknown>,
  mutation: MutationConfig,
): MutatedCandidate[] {
  const offsets = mutation.params.centering_offsets as { [joint: string]: { x_mm: number; y_mm: number; z_mm: number } } | undefined;
  const fingerDevs = mutation.params.finger_deviations as { [finger: string]: { y_mm: number; z_mm: number } } | undefined;
  const damping = mutation.params.damping as number ?? 0.6;
  const count = mutation.count;
  const candidates: MutatedCandidate[] = [];

  // Vary damping across candidates for robustness
  const dampingFactors = count > 1
    ? Array.from({ length: count }, (_, i) => damping * (0.5 + 0.5 * i / (count - 1)))
    : [damping];

  for (let i = 0; i < dampingFactors.length; i++) {
    const d = dampingFactors[i];
    const newJoints: { [name: string]: Vec3 } = {};
    const appliedDeltas: { [name: string]: Vec3 } = {};

    for (const [name, pos] of Object.entries(baseJoints)) {
      let dx = 0, dy = 0, dz = 0;

      // Apply centering correction: offset = centroid - joint, so ADD to move toward center
      if (offsets && offsets[name]) {
        dx = offsets[name].x_mm * d / 1000;
        dy = offsets[name].y_mm * d / 1000;
        dz = offsets[name].z_mm * d / 1000;
      }

      // Apply finger deviation correction: dev = meshCenter - skeleton, ADD to center
      if (fingerDevs) {
        // Match finger name to joint: e.g., l_index → l_index1, l_index2, l_index3
        for (const [finger, dev] of Object.entries(fingerDevs)) {
          if (name.startsWith(finger)) {
            dy += dev.y_mm * d / 1000;
            dz += dev.z_mm * d / 1000;
            break;
          }
        }
      }

      if (dx !== 0 || dy !== 0 || dz !== 0) {
        newJoints[name] = [pos[0] + dx, pos[1] + dy, pos[2] + dz];
        appliedDeltas[name] = [dx * 1000, dy * 1000, dz * 1000]; // store in mm
      } else {
        newJoints[name] = [...pos] as Vec3;
      }
    }

    const pct = Math.round(d * 100);
    candidates.push({
      joints: newJoints,
      config: { ...baseConfig },
      label: `centering_damp${pct}pct_#${i}`,
      mutations: {
        strategy: 'centering_correction',
        damping: d,
        appliedDeltas,
      },
    });
  }

  return candidates;
}

/** Sweep ground correction offset */
function groundCorrectionSweep(
  baseJoints: { [name: string]: Vec3 },
  baseConfig: Record<string, unknown>,
  mutation: MutationConfig,
): MutatedCandidate[] {
  const min = mutation.params.min as number ?? -0.01;
  const max = mutation.params.max as number ?? 0.02;
  const count = mutation.count;
  const candidates: MutatedCandidate[] = [];

  for (let i = 0; i < count; i++) {
    const value = count > 1 ? min + (max - min) * (i / (count - 1)) : (min + max) / 2;
    candidates.push({
      joints: { ...baseJoints },
      config: { ...baseConfig, groundCorrectionOffset: value },
      label: `ground_${(value * 1000).toFixed(1)}mm`,
      mutations: { strategy: 'ground_correction', groundCorrectionOffset: value },
    });
  }

  return candidates;
}
