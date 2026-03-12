import { BoundaryLine, TriRef, SlicePoint } from '../types/index.cjs';

/** Region found in a cross-section's Z binning */
interface ZRegion {
    zLo: number;
    zHi: number;
}
/** Result of finger boundary extrapolation */
interface BoundaryExtrapolationResult {
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
declare function findZRegions(dorsalPts: SlicePoint[], binW?: number): ZRegion[];
/**
 * Match Z regions to finger indices by nearest seed Z.
 * Returns array of region indices per finger, or null entries if unmatched.
 */
declare function matchRegionsToFingers(regions: ZRegion[], seedZ: number[]): number[];
/**
 * Run Phase 1+2 of boundary extrapolation:
 * Scan cross-sections outward from knuckle, find 4-finger separations,
 * record boundary Z positions, fit lines to each boundary.
 */
declare function extrapolateBoundaries(handTris: TriRef[], meshRestPos: Float32Array, knuckleX: number, knuckleY: number, handSign: number, seedZ: number[], fingerLen: number): BoundaryExtrapolationResult;

export { type BoundaryExtrapolationResult, type ZRegion, extrapolateBoundaries, findZRegions, matchRegionsToFingers };
