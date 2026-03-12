// src/geometry/plane-slice.ts
function buildPlaneFrame(normal) {
  const [nx, ny, nz] = normal;
  let tx, ty, tz;
  if (Math.abs(nx) < 0.9) {
    tx = 0;
    ty = nz;
    tz = -ny;
  } else {
    tx = -nz;
    ty = 0;
    tz = nx;
  }
  const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz);
  tx /= tLen;
  ty /= tLen;
  tz /= tLen;
  const bx = ny * tz - nz * ty;
  const by = nz * tx - nx * tz;
  const bz = nx * ty - ny * tx;
  return [[tx, ty, tz], [bx, by, bz]];
}
function sliceMeshAtPlane(planePoint, planeNormal, meshPos, indices, boneAxisOrOpts) {
  let boneAxis;
  let maxRadius = Infinity;
  if (boneAxisOrOpts) {
    if (Array.isArray(boneAxisOrOpts)) {
      boneAxis = boneAxisOrOpts;
    } else {
      boneAxis = boneAxisOrOpts.boneAxis;
      if (boneAxisOrOpts.maxRadius !== void 0) {
        maxRadius = boneAxisOrOpts.maxRadius;
      }
    }
  }
  const [px, py, pz] = planePoint;
  const [nx, ny, nz] = planeNormal;
  const maxRadiusSq = maxRadius * maxRadius;
  const [tangent, bitangent] = buildPlaneFrame(planeNormal);
  const [tx, ty, tz] = tangent;
  const [bx, by, bz] = bitangent;
  const points = [];
  const nTri = indices.length / 3;
  for (let ti = 0; ti < nTri; ti++) {
    const ai = indices[ti * 3], bi2 = indices[ti * 3 + 1], ci = indices[ti * 3 + 2];
    const a3 = ai * 3, b3 = bi2 * 3, c3 = ci * 3;
    const ax = meshPos[a3], ay = meshPos[a3 + 1], az = meshPos[a3 + 2];
    const bvx = meshPos[b3], bvy = meshPos[b3 + 1], bvz = meshPos[b3 + 2];
    const cx = meshPos[c3], cy = meshPos[c3 + 1], cz = meshPos[c3 + 2];
    if (maxRadius < Infinity) {
      const distSqA = (ax - px) ** 2 + (ay - py) ** 2 + (az - pz) ** 2;
      const distSqB = (bvx - px) ** 2 + (bvy - py) ** 2 + (bvz - pz) ** 2;
      const distSqC = (cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2;
      if (distSqA > maxRadiusSq && distSqB > maxRadiusSq && distSqC > maxRadiusSq) continue;
    }
    const da = (ax - px) * nx + (ay - py) * ny + (az - pz) * nz;
    const db = (bvx - px) * nx + (bvy - py) * ny + (bvz - pz) * nz;
    const dc = (cx - px) * nx + (cy - py) * ny + (cz - pz) * nz;
    if (da > 0 && db > 0 && dc > 0) continue;
    if (da < 0 && db < 0 && dc < 0) continue;
    const edges = [
      [da, db, ax, ay, az, bvx, bvy, bvz],
      [db, dc, bvx, bvy, bvz, cx, cy, cz],
      [da, dc, ax, ay, az, cx, cy, cz]
    ];
    for (const e of edges) {
      const d0 = e[0], d1 = e[1];
      if (d0 > 0 && d1 > 0 || d0 < 0 && d1 < 0) continue;
      if (d0 === 0 && d1 === 0) continue;
      const t = d0 / (d0 - d1);
      const wx = e[2] + t * (e[5] - e[2]);
      const wy = e[3] + t * (e[6] - e[3]);
      const wz = e[4] + t * (e[7] - e[4]);
      const dx = wx - px, dy = wy - py, dz = wz - pz;
      const u = dx * tx + dy * ty + dz * tz;
      const v = dx * bx + dy * by + dz * bz;
      points.push({ wx, wy, wz, u, v });
    }
  }
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
  const centroid3D = [sumWx / n, sumWy / n, sumWz / n];
  const offsetWorld = [
    centroid3D[0] - px,
    centroid3D[1] - py,
    centroid3D[2] - pz
  ];
  const axis = boneAxis ?? planeNormal;
  const [anx, any_, anz] = axis;
  const [at, ab] = buildPlaneFrame(axis);
  const alongBone = offsetWorld[0] * anx + offsetWorld[1] * any_ + offsetWorld[2] * anz;
  const lateral = offsetWorld[0] * at[0] + offsetWorld[1] * at[1] + offsetWorld[2] * at[2];
  const depth = offsetWorld[0] * ab[0] + offsetWorld[1] * ab[1] + offsetWorld[2] * ab[2];
  const offsetLocal = [alongBone, lateral, depth];
  return {
    points,
    centroid2D,
    centroid3D,
    offsetWorld,
    offsetLocal
  };
}
export {
  sliceMeshAtPlane
};
