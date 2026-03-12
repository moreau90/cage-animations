/**
 * General-purpose mesh plane slicer.
 * Slices a triangle mesh with an arbitrary plane (defined by point + normal),
 * returning intersection points in both 3D world coordinates and
 * bone-local 2D coordinates (for centroid computation).
 */
import type { Vec3 } from '../types/index.js';

/** A 3D intersection point on the slice plane, with 2D local coords */
export interface PlaneSlicePoint {
  /** World-space 3D position */
  wx: number;
  wy: number;
  wz: number;
  /** 2D coordinates in the plane's local frame (u = tangent, v = bitangent) */
  u: number;
  v: number;
}

/** Result of slicing a mesh with a plane */
export interface PlaneSliceResult {
  points: PlaneSlicePoint[];
  /** Centroid in 2D plane-local coords */
  centroid2D: { u: number; v: number };
  /** Centroid in 3D world coords */
  centroid3D: Vec3;
  /** Per-axis offsets from the query point to the mesh centroid (world space) */
  offsetWorld: Vec3;
  /** Per-axis offsets in bone-local frame: [along-bone, lateral, depth] */
  offsetLocal: Vec3;
}

/**
 * Build an orthonormal frame from a normal direction.
 * Returns [tangent, bitangent] vectors.
 */
function buildPlaneFrame(normal: Vec3): [Vec3, Vec3] {
  const [nx, ny, nz] = normal;
  // Choose initial vector not parallel to normal
  let tx: number, ty: number, tz: number;
  if (Math.abs(nx) < 0.9) {
    // cross(normal, [1,0,0])
    tx = 0;
    ty = nz;
    tz = -ny;
  } else {
    // cross(normal, [0,1,0])
    tx = -nz;
    ty = 0;
    tz = nx;
  }
  const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz);
  tx /= tLen; ty /= tLen; tz /= tLen;

  // bitangent = cross(normal, tangent)
  const bx = ny * tz - nz * ty;
  const by = nz * tx - nx * tz;
  const bz = nx * ty - ny * tx;

  return [[tx, ty, tz], [bx, by, bz]];
}

/**
 * Options for sliceMeshAtPlane.
 */
export interface PlaneSliceOptions {
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
export function sliceMeshAtPlane(
  planePoint: Vec3,
  planeNormal: Vec3,
  meshPos: Float32Array,
  indices: Uint32Array,
  boneAxisOrOpts?: Vec3 | PlaneSliceOptions,
): PlaneSliceResult {
  // Handle both legacy Vec3 arg and new options object
  let boneAxis: Vec3 | undefined;
  let maxRadius = Infinity;
  if (boneAxisOrOpts) {
    if (Array.isArray(boneAxisOrOpts)) {
      boneAxis = boneAxisOrOpts;
    } else {
      boneAxis = boneAxisOrOpts.boneAxis;
      if (boneAxisOrOpts.maxRadius !== undefined) {
        maxRadius = boneAxisOrOpts.maxRadius;
      }
    }
  }

  const [px, py, pz] = planePoint;
  const [nx, ny, nz] = planeNormal;
  const maxRadiusSq = maxRadius * maxRadius;

  // Build 2D frame on the plane
  const [tangent, bitangent] = buildPlaneFrame(planeNormal);
  const [tx, ty, tz] = tangent;
  const [bx, by, bz] = bitangent;

  const points: PlaneSlicePoint[] = [];
  const nTri = indices.length / 3;

  for (let ti = 0; ti < nTri; ti++) {
    const ai = indices[ti * 3], bi2 = indices[ti * 3 + 1], ci = indices[ti * 3 + 2];
    const a3 = ai * 3, b3 = bi2 * 3, c3 = ci * 3;

    const ax = meshPos[a3], ay = meshPos[a3 + 1], az = meshPos[a3 + 2];
    const bvx = meshPos[b3], bvy = meshPos[b3 + 1], bvz = meshPos[b3 + 2];
    const cx = meshPos[c3], cy = meshPos[c3 + 1], cz = meshPos[c3 + 2];

    // Radius filtering: skip triangle if ALL vertices are too far from planePoint
    if (maxRadius < Infinity) {
      const distSqA = (ax - px) ** 2 + (ay - py) ** 2 + (az - pz) ** 2;
      const distSqB = (bvx - px) ** 2 + (bvy - py) ** 2 + (bvz - pz) ** 2;
      const distSqC = (cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2;
      if (distSqA > maxRadiusSq && distSqB > maxRadiusSq && distSqC > maxRadiusSq) continue;
    }

    // Signed distance from each vertex to plane
    const da = (ax - px) * nx + (ay - py) * ny + (az - pz) * nz;
    const db = (bvx - px) * nx + (bvy - py) * ny + (bvz - pz) * nz;
    const dc = (cx - px) * nx + (cy - py) * ny + (cz - pz) * nz;

    // Skip if all on same side
    if (da > 0 && db > 0 && dc > 0) continue;
    if (da < 0 && db < 0 && dc < 0) continue;

    // Check each edge for crossing
    const edges: [number, number, number, number, number, number, number, number][] = [
      [da, db, ax, ay, az, bvx, bvy, bvz],
      [db, dc, bvx, bvy, bvz, cx, cy, cz],
      [da, dc, ax, ay, az, cx, cy, cz],
    ];

    for (const e of edges) {
      const d0 = e[0], d1 = e[1];
      if ((d0 > 0 && d1 > 0) || (d0 < 0 && d1 < 0)) continue;
      if (d0 === 0 && d1 === 0) continue;
      const t = d0 / (d0 - d1);
      const wx = e[2] + t * (e[5] - e[2]);
      const wy = e[3] + t * (e[6] - e[3]);
      const wz = e[4] + t * (e[7] - e[4]);

      // Project onto 2D plane frame
      const dx = wx - px, dy = wy - py, dz = wz - pz;
      const u = dx * tx + dy * ty + dz * tz;
      const v = dx * bx + dy * by + dz * bz;

      points.push({ wx, wy, wz, u, v });
    }
  }

  // Compute centroid
  let sumU = 0, sumV = 0, sumWx = 0, sumWy = 0, sumWz = 0;
  for (const pt of points) {
    sumU += pt.u;
    sumV += pt.v;
    sumWx += pt.wx;
    sumWy += pt.wy;
    sumWz += pt.wz;
  }
  const n = points.length || 1;
  const centroid2D = { u: sumU / n, v: sumV / n };
  const centroid3D: Vec3 = [sumWx / n, sumWy / n, sumWz / n];

  // World-space offset from query point to centroid
  const offsetWorld: Vec3 = [
    centroid3D[0] - px,
    centroid3D[1] - py,
    centroid3D[2] - pz,
  ];

  // Bone-local offset decomposition
  const axis = boneAxis ?? planeNormal;
  const [anx, any_, anz] = axis;
  const [at, ab] = buildPlaneFrame(axis);

  const alongBone = offsetWorld[0] * anx + offsetWorld[1] * any_ + offsetWorld[2] * anz;
  const lateral = offsetWorld[0] * at[0] + offsetWorld[1] * at[1] + offsetWorld[2] * at[2];
  const depth = offsetWorld[0] * ab[0] + offsetWorld[1] * ab[1] + offsetWorld[2] * ab[2];
  const offsetLocal: Vec3 = [alongBone, lateral, depth];

  return {
    points,
    centroid2D,
    centroid3D,
    offsetWorld,
    offsetLocal,
  };
}
