import { Vec3 } from '../types/index.cjs';

/**
 * General-purpose mesh plane slicer.
 * Slices a triangle mesh with an arbitrary plane (defined by point + normal),
 * returning intersection points in both 3D world coordinates and
 * bone-local 2D coordinates (for centroid computation).
 */

/** A 3D intersection point on the slice plane, with 2D local coords */
interface PlaneSlicePoint {
    /** World-space 3D position */
    wx: number;
    wy: number;
    wz: number;
    /** 2D coordinates in the plane's local frame (u = tangent, v = bitangent) */
    u: number;
    v: number;
}
/** Result of slicing a mesh with a plane */
interface PlaneSliceResult {
    points: PlaneSlicePoint[];
    /** Centroid in 2D plane-local coords */
    centroid2D: {
        u: number;
        v: number;
    };
    /** Centroid in 3D world coords */
    centroid3D: Vec3;
    /** Per-axis offsets from the query point to the mesh centroid (world space) */
    offsetWorld: Vec3;
    /** Per-axis offsets in bone-local frame: [along-bone, lateral, depth] */
    offsetLocal: Vec3;
}
/**
 * Options for sliceMeshAtPlane.
 */
interface PlaneSliceOptions {
    /** Optional bone axis for local-frame decomposition (defaults to planeNormal) */
    boneAxis?: Vec3;
    /**
     * Maximum distance (in mesh units) from planePoint for a triangle vertex
     * to be considered. Triangles where ALL 3 vertices are farther than this
     * distance from planePoint are skipped. Use this to prevent the slice from
     * catching unrelated geometry (e.g., slicing at a shoulder and catching
     * torso triangles on the other side of the body).
     */
    maxRadius?: number;
}
/**
 * Slice a triangle mesh with an arbitrary plane.
 *
 * @param planePoint - A point on the slice plane (e.g., the joint position)
 * @param planeNormal - Normal of the slice plane (e.g., bone axis direction, normalized)
 * @param meshPos - Interleaved mesh positions [x,y,z, ...]
 * @param indices - Triangle index buffer
 * @param boneAxisOrOpts - Optional bone axis Vec3 (legacy) or PlaneSliceOptions object
 * @returns PlaneSliceResult with centroid and offset measurements
 */
declare function sliceMeshAtPlane(planePoint: Vec3, planeNormal: Vec3, meshPos: Float32Array, indices: Uint32Array, boneAxisOrOpts?: Vec3 | PlaneSliceOptions): PlaneSliceResult;

export { type PlaneSliceOptions, type PlaneSlicePoint, type PlaneSliceResult, sliceMeshAtPlane };
