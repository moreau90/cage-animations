import { TriRef, SlicePoint } from '../types/index.js';

/**
 * Slice mesh triangles at plane X=x0, returns array of {y, z} intersection points.
 * Uses exact triangle-plane intersection — no approximation.
 */
declare function sliceMeshAtX(x0: number, triList: TriRef[], mPos: Float32Array): SlicePoint[];
/**
 * Pre-filter mesh triangles within a hand region bounding box.
 * Returns TriRef[] for use with sliceMeshAtX.
 */
declare function filterHandTriangles(meshRestPos: Float32Array, meshIndex: Uint32Array, knuckleX: number, knuckleY: number, knZMin: number, knZMax: number, handSign: number): TriRef[];

export { filterHandTriangles, sliceMeshAtX };
