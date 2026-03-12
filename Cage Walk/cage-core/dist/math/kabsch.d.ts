import { Vec3, RigidTransform } from '../types/index.js';

/**
 * Kabsch algorithm: best-fit rigid transform (R, t) mapping rest → current.
 * Parameterized — no globals. Pass rest positions as Vec3[] and current as flat Float32Array.
 */
declare function computeRigidTransformKabsch(indices: number[], restPositions: Vec3[], currentPositions: Float32Array): RigidTransform;

export { computeRigidTransformKabsch };
