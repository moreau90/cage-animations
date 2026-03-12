import type { TriRef, SlicePoint, BoundaryLine } from '../types/index.js';
import { sliceMeshAtX } from './cross-section.js';
import { fitLineZD, evalBoundaryLine } from './regression.js';
import type { DZPoint } from './regression.js';

/** Region found in a cross-section's Z binning */
export interface ZRegion {
  zLo: number;
  zHi: number;
}

/** Result of finger boundary extrapolation */
export interface BoundaryExtrapolationResult {
  /** 5 boundary lines (outerLo, gap01, gap12, gap23, outerHi) */
  boundaryLines: (BoundaryLine | null)[];
  /** Finger index order sorted by Z (low→high) */
  fingerOrderByZ: number[] | null;
  /** Per-finger boundary line index: lo side */
  fingerBndLo: number[];
  /** Per-finger boundary line index: hi side */
  fingerBndHi: number[];
  /** Whether all boundaries were successfully fit */
  ok: boolean;
}

/**
 * Find contiguous filled regions from dorsal Z-binned points.
 * Returns array of { zLo, zHi } sorted low→high by construction.
 */
export function findZRegions(
  dorsalPts: SlicePoint[],
  binW: number = 0.001,
): ZRegion[] {
  let zMin = Infinity, zMax = -Infinity;
  for (let pi = 0; pi < dorsalPts.length; pi++) {
    if (dorsalPts[pi].z < zMin) zMin = dorsalPts[pi].z;
    if (dorsalPts[pi].z > zMax) zMax = dorsalPts[pi].z;
  }
  const nBins = Math.ceil((zMax - zMin) / binW) + 1;
  if (nBins < 2 || nBins > 500) return [];
  const occupied = new Uint8Array(nBins);
  for (let pi = 0; pi < dorsalPts.length; pi++) {
    const bi = Math.min(nBins - 1, Math.floor((dorsalPts[pi].z - zMin) / binW));
    occupied[bi] = 1;
  }

  const regions: ZRegion[] = [];
  let regStart = -1;
  for (let bi = 0; bi <= nBins; bi++) {
    if (bi < nBins && occupied[bi]) {
      if (regStart < 0) regStart = bi;
    } else {
      if (regStart >= 0) {
        regions.push({
          zLo: zMin + regStart * binW,
          zHi: zMin + bi * binW,
        });
        regStart = -1;
      }
    }
  }
  return regions;
}

/**
 * Match Z regions to finger indices by nearest seed Z.
 * Returns array of region indices per finger, or null entries if unmatched.
 */
export function matchRegionsToFingers(
  regions: ZRegion[],
  seedZ: number[],
): number[] {
  const fingerRI = [-1, -1, -1, -1];
  const fingerRD = [Infinity, Infinity, Infinity, Infinity];
  for (let ri = 0; ri < regions.length; ri++) {
    const rc = (regions[ri].zLo + regions[ri].zHi) / 2;
    let bestFi = 0, bestD = Math.abs(rc - seedZ[0]);
    for (let fi = 1; fi < 4; fi++) {
      const dd = Math.abs(rc - seedZ[fi]);
      if (dd < bestD) { bestD = dd; bestFi = fi; }
    }
    if (bestD < fingerRD[bestFi]) {
      fingerRD[bestFi] = bestD;
      fingerRI[bestFi] = ri;
    }
  }
  return fingerRI;
}

/**
 * Run Phase 1+2 of boundary extrapolation:
 * Scan cross-sections outward from knuckle, find 4-finger separations,
 * record boundary Z positions, fit lines to each boundary.
 */
export function extrapolateBoundaries(
  handTris: TriRef[],
  meshRestPos: Float32Array,
  knuckleX: number,
  knuckleY: number,
  handSign: number,
  seedZ: number[],
  fingerLen: number,
): BoundaryExtrapolationResult {
  const scanStep = 0.002;
  const maxScanDist = fingerLen * 1.15;
  const scanStartDist = 0.005;
  const binW = 0.001;

  const bndPts: DZPoint[][] = [[], [], [], [], []];
  let fiByZ: number[] | null = null;

  for (let d = scanStartDist; d <= maxScanDist; d += scanStep) {
    const xPos = knuckleX + d * handSign;
    const pts = sliceMeshAtX(xPos, handTris, meshRestPos);

    // Collect hand-level points
    const rawPts: SlicePoint[] = [];
    for (let pi = 0; pi < pts.length; pi++) {
      if (Math.abs(pts[pi].y - knuckleY) <= 0.035) rawPts.push(pts[pi]);
    }
    if (rawPts.length < 4) continue;

    // Filter to dorsal points (Y > median)
    rawPts.sort((a, b) => a.y - b.y);
    const medianY = rawPts[Math.floor(rawPts.length / 2)].y;
    const dorsalPts: SlicePoint[] = [];
    for (let pi = 0; pi < rawPts.length; pi++) {
      if (rawPts[pi].y >= medianY) dorsalPts.push(rawPts[pi]);
    }
    if (dorsalPts.length < 2) continue;

    const regions = findZRegions(dorsalPts, binW);
    if (regions.length < 4) continue;

    const fingerRI = matchRegionsToFingers(regions, seedZ);
    if (fingerRI[0] < 0 || fingerRI[1] < 0 || fingerRI[2] < 0 || fingerRI[3] < 0) continue;

    // Establish Z ordering
    const curOrder = [0, 1, 2, 3].sort((a, b) =>
      (regions[fingerRI[a]].zLo + regions[fingerRI[a]].zHi) -
      (regions[fingerRI[b]].zLo + regions[fingerRI[b]].zHi));
    if (!fiByZ) fiByZ = curOrder.slice();

    // Record boundary data points
    bndPts[0].push({ d, z: regions[fingerRI[fiByZ[0]]].zLo });
    bndPts[4].push({ d, z: regions[fingerRI[fiByZ[3]]].zHi });
    for (let gi = 0; gi < 3; gi++) {
      const gapZ = (regions[fingerRI[fiByZ[gi]]].zHi +
                     regions[fingerRI[fiByZ[gi + 1]]].zLo) / 2;
      bndPts[gi + 1].push({ d, z: gapZ });
    }
  }

  // Phase 2: Fit lines
  const bndLines = bndPts.map(fitLineZD);
  const ok = fiByZ !== null && bndLines.every(l => l !== null);

  // Map each finger to its lo/hi boundary indices
  const fingerBndLo = [0, 0, 0, 0];
  const fingerBndHi = [0, 0, 0, 0];
  if (fiByZ) {
    for (let k = 0; k < 4; k++) {
      fingerBndLo[fiByZ[k]] = k;
      fingerBndHi[fiByZ[k]] = k + 1;
    }
  }

  return {
    boundaryLines: bndLines,
    fingerOrderByZ: fiByZ,
    fingerBndLo,
    fingerBndHi,
    ok,
  };
}
