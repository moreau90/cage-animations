import { Vec3 } from '../types/index.cjs';

/** Rotate point around an axis through origin, returning new position */
declare function rotateAroundAxis(px: number, py: number, pz: number, ox: number, oy: number, oz: number, ax: number, ay: number, az: number, angle: number): Vec3;
/** Compute delta rotation from fromDir→toDir, apply to applyToDir */
declare function applyDeltaRotation(fromDir: Vec3, toDir: Vec3, applyToDir: Vec3): Vec3;
/** Compute axis-angle rotation MATRIX from restDir to targetDir (3x3 flat array) */
declare function computeIKRotMatrix(restDir: Vec3, targetDir: Vec3): number[] | Float64Array;

export { applyDeltaRotation, computeIKRotMatrix, rotateAroundAxis };
