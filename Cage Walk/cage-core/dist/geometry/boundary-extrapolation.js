// src/geometry/cross-section.ts
function sliceMeshAtX(x0, triList, mPos) {
  const pts = [];
  for (let ti = 0; ti < triList.length; ti++) {
    const t = triList[ti];
    const ax = mPos[t.a3], ay = mPos[t.a3 + 1], az = mPos[t.a3 + 2];
    const bx = mPos[t.b3], by = mPos[t.b3 + 1], bz = mPos[t.b3 + 2];
    const cx = mPos[t.c3], cy = mPos[t.c3 + 1], cz = mPos[t.c3 + 2];
    const sa = ax - x0, sb = bx - x0, sc = cx - x0;
    if (sa > 0 && sb > 0 && sc > 0) continue;
    if (sa < 0 && sb < 0 && sc < 0) continue;
    const edges = [
      [sa, sb, ax, ay, az, bx, by, bz],
      [sb, sc, bx, by, bz, cx, cy, cz],
      [sa, sc, ax, ay, az, cx, cy, cz]
    ];
    for (let ei = 0; ei < 3; ei++) {
      const e = edges[ei];
      const d0 = e[0], d1 = e[1];
      if (d0 > 0 && d1 > 0 || d0 < 0 && d1 < 0) continue;
      if (d0 === 0 && d1 === 0) continue;
      const t_ = d0 / (d0 - d1);
      pts.push({
        y: e[3] + t_ * (e[6] - e[3]),
        z: e[4] + t_ * (e[7] - e[4])
      });
    }
  }
  return pts;
}

// src/geometry/regression.ts
function fitLineZD(pts) {
  const n = pts.length;
  if (n < 2) return null;
  let sD = 0, sZ = 0, sDD = 0, sDZ = 0;
  for (let i = 0; i < n; i++) {
    sD += pts[i].d;
    sZ += pts[i].z;
    sDD += pts[i].d * pts[i].d;
    sDZ += pts[i].d * pts[i].z;
  }
  const den = n * sDD - sD * sD;
  if (Math.abs(den) < 1e-12) return { slope: 0, intercept: sZ / n };
  const slope = (n * sDZ - sD * sZ) / den;
  const intercept = (sZ - slope * sD) / n;
  return { slope, intercept };
}

// src/geometry/boundary-extrapolation.ts
function findZRegions(dorsalPts, binW = 1e-3) {
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
  const regions = [];
  let regStart = -1;
  for (let bi = 0; bi <= nBins; bi++) {
    if (bi < nBins && occupied[bi]) {
      if (regStart < 0) regStart = bi;
    } else {
      if (regStart >= 0) {
        regions.push({
          zLo: zMin + regStart * binW,
          zHi: zMin + bi * binW
        });
        regStart = -1;
      }
    }
  }
  return regions;
}
function matchRegionsToFingers(regions, seedZ) {
  const fingerRI = [-1, -1, -1, -1];
  const fingerRD = [Infinity, Infinity, Infinity, Infinity];
  for (let ri = 0; ri < regions.length; ri++) {
    const rc = (regions[ri].zLo + regions[ri].zHi) / 2;
    let bestFi = 0, bestD = Math.abs(rc - seedZ[0]);
    for (let fi = 1; fi < 4; fi++) {
      const dd = Math.abs(rc - seedZ[fi]);
      if (dd < bestD) {
        bestD = dd;
        bestFi = fi;
      }
    }
    if (bestD < fingerRD[bestFi]) {
      fingerRD[bestFi] = bestD;
      fingerRI[bestFi] = ri;
    }
  }
  return fingerRI;
}
function extrapolateBoundaries(handTris, meshRestPos, knuckleX, knuckleY, handSign, seedZ, fingerLen) {
  const scanStep = 2e-3;
  const maxScanDist = fingerLen * 1.15;
  const scanStartDist = 5e-3;
  const binW = 1e-3;
  const bndPts = [[], [], [], [], []];
  let fiByZ = null;
  for (let d = scanStartDist; d <= maxScanDist; d += scanStep) {
    const xPos = knuckleX + d * handSign;
    const pts = sliceMeshAtX(xPos, handTris, meshRestPos);
    const rawPts = [];
    for (let pi = 0; pi < pts.length; pi++) {
      if (Math.abs(pts[pi].y - knuckleY) <= 0.035) rawPts.push(pts[pi]);
    }
    if (rawPts.length < 4) continue;
    rawPts.sort((a, b) => a.y - b.y);
    const medianY = rawPts[Math.floor(rawPts.length / 2)].y;
    const dorsalPts = [];
    for (let pi = 0; pi < rawPts.length; pi++) {
      if (rawPts[pi].y >= medianY) dorsalPts.push(rawPts[pi]);
    }
    if (dorsalPts.length < 2) continue;
    const regions = findZRegions(dorsalPts, binW);
    if (regions.length < 4) continue;
    const fingerRI = matchRegionsToFingers(regions, seedZ);
    if (fingerRI[0] < 0 || fingerRI[1] < 0 || fingerRI[2] < 0 || fingerRI[3] < 0) continue;
    const curOrder = [0, 1, 2, 3].sort((a, b) => regions[fingerRI[a]].zLo + regions[fingerRI[a]].zHi - (regions[fingerRI[b]].zLo + regions[fingerRI[b]].zHi));
    if (!fiByZ) fiByZ = curOrder.slice();
    bndPts[0].push({ d, z: regions[fingerRI[fiByZ[0]]].zLo });
    bndPts[4].push({ d, z: regions[fingerRI[fiByZ[3]]].zHi });
    for (let gi = 0; gi < 3; gi++) {
      const gapZ = (regions[fingerRI[fiByZ[gi]]].zHi + regions[fingerRI[fiByZ[gi + 1]]].zLo) / 2;
      bndPts[gi + 1].push({ d, z: gapZ });
    }
  }
  const bndLines = bndPts.map(fitLineZD);
  const ok = fiByZ !== null && bndLines.every((l) => l !== null);
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
    ok
  };
}
export {
  extrapolateBoundaries,
  findZRegions,
  matchRegionsToFingers
};
