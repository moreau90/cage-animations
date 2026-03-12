"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/geometry/mesh-adjacency.ts
var mesh_adjacency_exports = {};
__export(mesh_adjacency_exports, {
  buildMeshAdjacency: () => buildMeshAdjacency
});
module.exports = __toCommonJS(mesh_adjacency_exports);
function buildMeshAdjacency(meshRestPos, indices) {
  const nTri = indices.length / 3;
  const nMesh = meshRestPos.length / 3;
  const edgeSet = /* @__PURE__ */ new Set();
  const edgeAArr = [];
  const edgeBArr = [];
  const edgeRestLenArr = [];
  for (let t = 0; t < nTri; t++) {
    const a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
    const pairs = [[a, b], [b, c], [a, c]];
    for (const [v0, v1] of pairs) {
      const lo = Math.min(v0, v1), hi = Math.max(v0, v1);
      const key = lo * nMesh + hi;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edgeAArr.push(lo);
        edgeBArr.push(hi);
        const dx = meshRestPos[hi * 3] - meshRestPos[lo * 3];
        const dy = meshRestPos[hi * 3 + 1] - meshRestPos[lo * 3 + 1];
        const dz = meshRestPos[hi * 3 + 2] - meshRestPos[lo * 3 + 2];
        edgeRestLenArr.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
  }
  const nEdges = edgeAArr.length;
  const edgeA = new Uint32Array(edgeAArr);
  const edgeB = new Uint32Array(edgeBArr);
  const edgeRestLen = new Float32Array(edgeRestLenArr);
  const adjCount = new Uint16Array(nMesh);
  for (let i = 0; i < nEdges; i++) {
    adjCount[edgeA[i]]++;
    adjCount[edgeB[i]]++;
  }
  const adjStart = new Uint32Array(nMesh);
  for (let i = 1; i < nMesh; i++) adjStart[i] = adjStart[i - 1] + adjCount[i - 1];
  const totalAdj = nMesh > 0 ? adjStart[nMesh - 1] + adjCount[nMesh - 1] : 0;
  const adjList = new Uint32Array(totalAdj);
  const adjFill = new Uint16Array(nMesh);
  for (let i = 0; i < nEdges; i++) {
    const a = edgeA[i], b = edgeB[i];
    adjList[adjStart[a] + adjFill[a]++] = b;
    adjList[adjStart[b] + adjFill[b]++] = a;
  }
  return {
    edgeA,
    edgeB,
    edgeRestLen,
    nEdges,
    nVerts: nMesh,
    adjList,
    adjStart,
    adjCount
  };
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  buildMeshAdjacency
});
