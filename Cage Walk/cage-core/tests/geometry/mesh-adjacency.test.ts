import { describe, it, expect } from 'vitest';
import { buildMeshAdjacency } from '../../src/geometry/mesh-adjacency.js';

describe('buildMeshAdjacency', () => {
  // Two triangles sharing an edge: (0,0,0)-(1,0,0)-(0,1,0) and (1,0,0)-(1,1,0)-(0,1,0)
  const positions = new Float32Array([
    0, 0, 0,   // v0
    1, 0, 0,   // v1
    0, 1, 0,   // v2
    1, 1, 0,   // v3
  ]);
  const indices = new Uint32Array([0, 1, 2, 1, 3, 2]);

  it('counts edges correctly (shared edge counted once)', () => {
    const adj = buildMeshAdjacency(positions, indices);
    // 2 triangles with 3 edges each = 6, minus 1 shared = 5 unique edges
    expect(adj.nEdges).toBe(5);
    expect(adj.nVerts).toBe(4);
  });

  it('computes rest edge lengths', () => {
    const adj = buildMeshAdjacency(positions, indices);
    for (let i = 0; i < adj.nEdges; i++) {
      expect(adj.edgeRestLen[i]).toBeGreaterThan(0);
    }
    // Edge from v0(0,0,0) to v1(1,0,0) should be length 1
    let found = false;
    for (let i = 0; i < adj.nEdges; i++) {
      if ((adj.edgeA[i] === 0 && adj.edgeB[i] === 1) ||
          (adj.edgeA[i] === 1 && adj.edgeB[i] === 0)) {
        expect(adj.edgeRestLen[i]).toBeCloseTo(1.0, 6);
        found = true;
      }
    }
    expect(found).toBe(true);
  });

  it('builds neighbor lists with correct counts', () => {
    const adj = buildMeshAdjacency(positions, indices);
    // v0 connects to v1, v2 → 2 neighbors
    expect(adj.adjCount[0]).toBe(2);
    // v1 connects to v0, v2, v3 → 3 neighbors
    expect(adj.adjCount[1]).toBe(3);
    // v2 connects to v0, v1, v3 → 3 neighbors
    expect(adj.adjCount[2]).toBe(3);
    // v3 connects to v1, v2 → 2 neighbors
    expect(adj.adjCount[3]).toBe(2);
  });

  it('neighbor lists contain correct vertex indices', () => {
    const adj = buildMeshAdjacency(positions, indices);
    // Check v0's neighbors
    const v0Neighbors = new Set<number>();
    for (let j = adj.adjStart[0]; j < adj.adjStart[0] + adj.adjCount[0]; j++) {
      v0Neighbors.add(adj.adjList[j]);
    }
    expect(v0Neighbors.has(1)).toBe(true);
    expect(v0Neighbors.has(2)).toBe(true);
    expect(v0Neighbors.has(3)).toBe(false);
  });

  it('handles single triangle', () => {
    const pos = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const idx = new Uint32Array([0, 1, 2]);
    const adj = buildMeshAdjacency(pos, idx);
    expect(adj.nEdges).toBe(3);
    expect(adj.nVerts).toBe(3);
  });

  it('handles empty mesh', () => {
    const pos = new Float32Array(0);
    const idx = new Uint32Array(0);
    const adj = buildMeshAdjacency(pos, idx);
    expect(adj.nEdges).toBe(0);
    expect(adj.nVerts).toBe(0);
  });
});
