import type { Vec3 } from '../types/index.js';
import { vsub, vadd, vdot, vlen, vscale, vcross, vnorm } from './vec3.js';
import { mat3_rotAxis } from './mat3.js';

/** Rotate point around an axis through origin, returning new position */
export function rotateAroundAxis(
  px: number, py: number, pz: number,
  ox: number, oy: number, oz: number,
  ax: number, ay: number, az: number,
  angle: number,
): Vec3 {
  const x = px - ox, y = py - oy, z = pz - oz;
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  const len = Math.hypot(ax, ay, az) || 1;
  ax /= len; ay /= len; az /= len;

  const nx = (t * ax * ax + c) * x + (t * ax * ay - s * az) * y + (t * ax * az + s * ay) * z;
  const ny = (t * ax * ay + s * az) * x + (t * ay * ay + c) * y + (t * ay * az - s * ax) * z;
  const nz = (t * ax * az - s * ay) * x + (t * ay * az + s * ax) * y + (t * az * az + c) * z;
  return [nx + ox, ny + oy, nz + oz];
}

/** Compute delta rotation from fromDir→toDir, apply to applyToDir */
export function applyDeltaRotation(fromDir: Vec3, toDir: Vec3, applyToDir: Vec3): Vec3 {
  let ax = vcross(fromDir, toDir);
  const axLen = vlen(ax);
  const d = Math.max(-1, Math.min(1, vdot(fromDir, toDir)));
  const ang = Math.acos(d);
  if (Math.abs(ang) < 1e-4) return applyToDir;
  if (axLen < 1e-6) {
    ax = Math.abs(fromDir[1]) < 0.9
      ? vcross(fromDir, [0, 1, 0])
      : vcross(fromDir, [1, 0, 0]);
    if (vlen(ax) < 1e-6) return applyToDir;
  }
  ax = vnorm(ax);
  const r = rotateAroundAxis(
    applyToDir[0], applyToDir[1], applyToDir[2],
    0, 0, 0, ax[0], ax[1], ax[2], ang,
  );
  return vnorm(r);
}

/** Compute axis-angle rotation MATRIX from restDir to targetDir (3x3 flat array) */
export function computeIKRotMatrix(restDir: Vec3, targetDir: Vec3): number[] | Float64Array {
  let ax = vcross(restDir, targetDir);
  const axLen = vlen(ax);
  const d = Math.max(-1, Math.min(1, vdot(restDir, targetDir)));
  const ang = Math.acos(d);
  if (Math.abs(ang) < 1e-4) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  if (axLen < 1e-6) {
    ax = Math.abs(restDir[1]) < 0.9
      ? vcross(restDir, [0, 1, 0])
      : vcross(restDir, [1, 0, 0]);
    if (vlen(ax) < 1e-6) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }
  ax = vnorm(ax);
  return mat3_rotAxis(ax[0], ax[1], ax[2], ang);
}
