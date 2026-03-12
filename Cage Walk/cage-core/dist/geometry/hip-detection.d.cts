/** Result of hip geometry auto-detection */
interface HipGeometry {
    hipsCenterY: number;
    hipJointY: number;
    hipSpreadHalf: number;
    crotchY: number;
    hipShelfY: number;
}
/**
 * Auto-detect hip geometry from mesh cross-sections.
 * Scans mesh to find crotch (center density peak) and hip shelf (widest above crotch),
 * then derives hip positions using anatomical proportions.
 *
 * @param roughY - Rough Y estimate (groin click or stored position)
 * @param meshH - Total mesh height
 * @param meshRestPos - Interleaved rest positions [x,y,z, x,y,z, ...], or null for fallback
 */
declare function detectHipGeometry(roughY: number, meshH: number, meshRestPos: Float32Array | null): HipGeometry;

export { type HipGeometry, detectHipGeometry };
