import type { BoundaryLine } from '../types/index.js';

/** Data point for boundary line fitting: distance from knuckle + Z value */
export interface DZPoint {
  d: number;
  z: number;
}

/**
 * Fit a line Z = slope * d + intercept to an array of {d, z} points.
 * Uses ordinary least-squares linear regression.
 * Returns null if fewer than 2 points.
 */
export function fitLineZD(pts: DZPoint[]): BoundaryLine | null {
  const n = pts.length;
  if (n < 2) return null;
  let sD = 0, sZ = 0, sDD = 0, sDZ = 0;
  for (let i = 0; i < n; i++) {
    sD += pts[i].d; sZ += pts[i].z;
    sDD += pts[i].d * pts[i].d; sDZ += pts[i].d * pts[i].z;
  }
  const den = n * sDD - sD * sD;
  if (Math.abs(den) < 1e-12) return { slope: 0, intercept: sZ / n };
  const slope = (n * sDZ - sD * sZ) / den;
  const intercept = (sZ - slope * sD) / n;
  return { slope, intercept };
}

/** Evaluate a boundary line at distance d */
export function evalBoundaryLine(line: BoundaryLine, d: number): number {
  return line.slope * d + line.intercept;
}
