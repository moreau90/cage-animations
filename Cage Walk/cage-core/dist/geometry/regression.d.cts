import { BoundaryLine } from '../types/index.cjs';

/** Data point for boundary line fitting: distance from knuckle + Z value */
interface DZPoint {
    d: number;
    z: number;
}
/**
 * Fit a line Z = slope * d + intercept to an array of {d, z} points.
 * Uses ordinary least-squares linear regression.
 * Returns null if fewer than 2 points.
 */
declare function fitLineZD(pts: DZPoint[]): BoundaryLine | null;
/** Evaluate a boundary line at distance d */
declare function evalBoundaryLine(line: BoundaryLine, d: number): number;

export { type DZPoint, evalBoundaryLine, fitLineZD };
