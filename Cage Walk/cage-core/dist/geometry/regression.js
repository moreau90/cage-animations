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
function evalBoundaryLine(line, d) {
  return line.slope * d + line.intercept;
}
export {
  evalBoundaryLine,
  fitLineZD
};
