#!/usr/bin/env node
/**
 * finger_placement.js
 *
 * Mesh-based finger joint placement for cage animation system.
 * Parses mesh.glb directly (no npm dependencies), identifies fingers via
 * BFS flood-fill after webbing removal, computes centerlines, places joints.
 * Updates placed_joints.json and generates a Three.js debug visualization.
 */

'use strict';
const fs = require('fs');
const path = require('path');

const BASE_DIR = 'C:/Users/rmore/Godot/Cage Animations/Cage Walk';
const GLB_PATH = path.join(BASE_DIR, 'mesh.glb');
const JOINTS_PATH = path.join(BASE_DIR, 'placed_joints.json');
const DEBUG_HTML_PATH = path.join(BASE_DIR, 'finger_bfs_debug.html');

// ═══════════════════════════════════════════════════════════════════
// 1. PARSE GLB
// ═══════════════════════════════════════════════════════════════════

function parseGLB(filePath) {
    const buf = fs.readFileSync(filePath);
    const magic = buf.toString('ascii', 0, 4);
    if (magic !== 'glTF') throw new Error('Not a GLB file');
    const version = buf.readUInt32LE(4);
    console.log(`[GLB] Version ${version}, file size ${buf.length} bytes`);

    const c0Len = buf.readUInt32LE(12);
    const json = JSON.parse(buf.toString('utf8', 20, 20 + c0Len));
    const c1DataOffset = 20 + c0Len + 8;
    const prim = json.meshes[0].primitives[0];

    // Positions
    const posAcc = json.accessors[prim.attributes.POSITION];
    const posBV = json.bufferViews[posAcc.bufferView];
    const posStart = c1DataOffset + posBV.byteOffset + (posAcc.byteOffset || 0);
    const vertexCount = posAcc.count;
    const positions = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount * 3; i++) {
        positions[i] = buf.readFloatLE(posStart + i * 4);
    }

    // Indices
    const idxAcc = json.accessors[prim.indices];
    const idxBV = json.bufferViews[idxAcc.bufferView];
    const idxStart = c1DataOffset + idxBV.byteOffset + (idxAcc.byteOffset || 0);
    const indexCount = idxAcc.count;
    let indices;
    if (idxAcc.componentType === 5125) {
        indices = new Uint32Array(indexCount);
        for (let i = 0; i < indexCount; i++) indices[i] = buf.readUInt32LE(idxStart + i * 4);
    } else {
        indices = new Uint16Array(indexCount);
        for (let i = 0; i < indexCount; i++) indices[i] = buf.readUInt16LE(idxStart + i * 2);
    }

    console.log(`[GLB] ${vertexCount} vertices, ${indexCount / 3} triangles`);
    return { positions, indices, vertexCount, triCount: indexCount / 3 };
}

// ═══════════════════════════════════════════════════════════════════
// 2. IDENTIFY HAND REGIONS
// ═══════════════════════════════════════════════════════════════════

function identifyHandVertices(positions, vertexCount) {
    const leftHand = [], rightHand = [];
    for (let i = 0; i < vertexCount; i++) {
        const x = positions[i * 3];
        if (x < -0.33) leftHand.push(i);
        else if (x > 0.33) rightHand.push(i);
    }
    console.log(`[HANDS] Left: ${leftHand.length} verts, Right: ${rightHand.length} verts`);
    return { leftHand, rightHand };
}

// ═══════════════════════════════════════════════════════════════════
// 3. ASSIGN FINGERS VIA FBX PROXIMITY OR CORRECTED FILE
// ═══════════════════════════════════════════════════════════════════

/**
 * Load corrected finger assignment from a user-painted JSON file.
 * Returns Map<vertexIndex, fingerName> or null if not available.
 */
// Global blend weights map: vi → weight (0-1) for the assigned finger.
// Core verts get 1.0, overlap verts get their blend weight (e.g. 0.7).
// Used by computeCenterline to downweight contaminated overlap verts.
const vertBlendWeight = new Map();

function loadCorrectedAssignment(isLeft) {
    const ALGO_PATH = path.join(BASE_DIR, 'finger_assignment.json');
    const CORRECTED_PATH = path.join(BASE_DIR, 'finger_assignment_corrected.json');
    const ALT_PATH = path.join(BASE_DIR, 'finger_assignment (1).json');

    for (const p of [CORRECTED_PATH, ALGO_PATH, ALT_PATH]) {
        try {
            const data = JSON.parse(fs.readFileSync(p, 'utf8'));
            const hand = isLeft ? data.left : data.right;
            if (!hand || hand.length === 0) continue;

            const assignment = new Map();
            const nameToIdx = { pinky: 0, ring: 1, middle: 2, index: 3, thumb: 4 };
            for (const v of hand) {
                const idx = nameToIdx[v.finger];
                if (idx !== undefined) {
                    assignment.set(v.vi, idx);
                    // Load blend weight: if this vert has a blend, use the weight for its assigned finger
                    if (v.blend && v.blend[v.finger] !== undefined) {
                        vertBlendWeight.set(v.vi, v.blend[v.finger]);
                    } else {
                        vertBlendWeight.set(v.vi, 1.0);
                    }
                }
            }

            if (assignment.size > 100) {
                const blendedCount = [...vertBlendWeight.values()].filter(w => w < 1.0).length;
                console.log(`[ASSIGN] Loaded assignment from ${path.basename(p)}: ${assignment.size} verts (${blendedCount} with blend weights)`);
                return assignment;
            }
        } catch { /* not found */ }
    }
    return null;
}

/**
 * Load finger skeleton reference from placed_joints.json (self-bootstrapping)
 * or fbx_mapped_fingers.json as fallback.
 * Returns { finger_name: [[x,y,z], [x,y,z], ...] } in meters, or null.
 */
function loadFingerReference(side) {
    // Priority 1: FBX mapped positions (anatomically correct skeleton, stable reference)
    // Using FBX as fixed reference prevents drift from iterative self-bootstrapping
    const FBX_REF_PATH = path.join(BASE_DIR, 'fbx_mapped_fingers.json');
    try {
        const data = JSON.parse(fs.readFileSync(FBX_REF_PATH, 'utf8'));
        const fingers = {};
        for (const name of ['pinky', 'ring', 'middle', 'index', 'thumb']) {
            const joints = [];
            for (let i = 1; i <= 4; i++) {
                const key = `${side}_${name}${i}`;
                if (data[key]) joints.push(data[key]);
            }
            if (joints.length >= 2) fingers[name] = joints;
        }
        if (Object.keys(fingers).length >= 4) {
            console.log(`[REF] Using fbx_mapped_fingers.json for ${side} hand`);
            return fingers;
        }
    } catch { /* no FBX ref */ }

    // Priority 2: Fall back to placed_joints.json (if no FBX available)
    try {
        const joints = JSON.parse(fs.readFileSync(JOINTS_PATH, 'utf8'));
        const P = joints.P;
        const fingers = {};
        for (const name of ['pinky', 'ring', 'middle', 'index', 'thumb']) {
            const chain = [];
            for (let i = 1; i <= 3; i++) {
                let key;
                if (name === 'middle' && i === 1) key = `${side}_mid_knuckle`;
                else key = `${side}_${name}${i}`;
                if (P[key]) chain.push(P[key]);
            }
            if (chain.length >= 2) fingers[name] = chain;
        }
        if (Object.keys(fingers).length >= 4) {
            console.log(`[REF] Using placed_joints.json finger positions for ${side} hand`);
            return fingers;
        }
    } catch { /* no placed joints yet */ }

    return null;
}

/**
 * Distance from a 3D point to a polyline (sequence of line segments).
 */
function distToPolyline(px, py, pz, polyline) {
    let minDist = Infinity;
    for (let i = 0; i < polyline.length - 1; i++) {
        const [ax, ay, az] = polyline[i];
        const [bx, by, bz] = polyline[i + 1];
        const abx = bx - ax, aby = by - ay, abz = bz - az;
        const apx = px - ax, apy = py - ay, apz = pz - az;
        const abLen2 = abx * abx + aby * aby + abz * abz;
        const t = abLen2 > 0 ? Math.max(0, Math.min(1, (apx * abx + apy * aby + apz * abz) / abLen2)) : 0;
        const cx = ax + t * abx, cy = ay + t * aby, cz = az + t * abz;
        const dx = px - cx, dy = py - cy, dz = pz - cz;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d < minDist) minDist = d;
    }
    // Also check endpoints
    for (const [ex, ey, ez] of polyline) {
        const dx = px - ex, dy = py - ey, dz = pz - ez;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d < minDist) minDist = d;
    }
    return minDist;
}

/**
 * Assign vertices to fingers by proximity to FBX skeleton lines.
 * Each vertex is assigned to the finger whose skeleton polyline is closest.
 * Vertices farther than MAX_DIST from any skeleton are classified as "palm" (idx 5).
 */
function assignByFBXProximity(positions, handVerts, fbxFingers, isLeft) {
    const handName = isLeft ? 'left' : 'right';
    const nameToIdx = { pinky: 0, ring: 1, middle: 2, index: 3, thumb: 4, palm: 5 };

    // Max distance from any finger skeleton to be considered a "finger" vertex.
    // Beyond this → palm/wrist/forearm vertex.
    const MAX_FINGER_DIST = 0.030; // 30mm — accounts for FBX mapping error

    // Separate thumb from finger assignment:
    // - Vertices with Y >= 0.185 are in the finger zone → only compare to 4 fingers (no thumb)
    // - Vertices with Y < 0.185 are in the thumb zone → compare to all including thumb
    const fingerOnlyNames = ['pinky', 'ring', 'middle', 'index'];
    const allNames = ['pinky', 'ring', 'middle', 'index', 'thumb'];

    // Compute the knuckle line X threshold: only assign "finger" to verts past the knuckle base
    // Use the most palm-ward bone1 position as the cutoff
    let knuckleX = isLeft ? -Infinity : Infinity;
    for (const fname of fingerOnlyNames) {
        if (!fbxFingers[fname] || fbxFingers[fname].length === 0) continue;
        const baseX = fbxFingers[fname][0][0]; // bone1 X position
        if (isLeft) {
            if (baseX > knuckleX) knuckleX = baseX; // least negative = most palm-ward
        } else {
            if (baseX < knuckleX) knuckleX = baseX; // least positive = most palm-ward
        }
    }
    // Add 30mm palm-ward padding to compensate for FBX mapping error
    knuckleX += isLeft ? 0.030 : -0.030;
    console.log(`[ASSIGN] ${handName} knuckle X threshold: ${(knuckleX * 1000).toFixed(1)}mm`);

    const assignment = new Map();
    let palmCount = 0;
    for (const vi of handVerts) {
        const px = positions[vi * 3], py = positions[vi * 3 + 1], pz = positions[vi * 3 + 2];

        // Vertices palm-ward of knuckle line → palm (unless in thumb zone)
        const isPastKnuckle = isLeft ? (px < knuckleX) : (px > knuckleX);
        const isFingerZone = py >= 0.185;
        const candidates = isFingerZone ? fingerOnlyNames : allNames;

        // For non-thumb finger zone: must be past knuckle line
        if (isFingerZone && !isPastKnuckle) {
            assignment.set(vi, nameToIdx.palm);
            palmCount++;
            continue;
        }

        // Compute distance to all candidate fingers
        const dists = [];
        for (const fname of candidates) {
            if (!fbxFingers[fname]) continue;
            dists.push({ name: fname, dist: distToPolyline(px, py, pz, fbxFingers[fname]) });
        }
        dists.sort((a, b) => a.dist - b.dist);

        if (dists.length === 0 || dists[0].dist > MAX_FINGER_DIST) {
            assignment.set(vi, nameToIdx.palm);
            palmCount++;
        } else {
            let bestFinger = dists[0].name;

            // Z-midpoint tie-breaking: if two adjacent fingers are within 5mm of each other,
            // use the Z midpoint between their skeleton bases to decide ownership.
            // This fixes webbing area misassignment (e.g. pinky→ring).
            if (dists.length >= 2) {
                const margin = dists[1].dist - dists[0].dist;
                if (margin < 0.005) { // within 5mm
                    const f1 = dists[0].name, f2 = dists[1].name;
                    const b1 = fbxFingers[f1], b2 = fbxFingers[f2];
                    if (b1 && b2 && b1[0] && b2[0]) {
                        const midZ = (b1[0][2] + b2[0][2]) / 2;
                        // The finger with the closer Z to the vertex wins
                        const f1zDist = Math.abs(pz - b1[0][2]);
                        const f2zDist = Math.abs(pz - b2[0][2]);
                        bestFinger = f1zDist <= f2zDist ? f1 : f2;
                    }
                }
            }

            assignment.set(vi, nameToIdx[bestFinger]);
        }
    }

    const counts = [0, 0, 0, 0, 0, 0];
    for (const [, fi] of assignment) counts[fi]++;
    const names = ['pinky', 'ring', 'middle', 'index', 'thumb', 'palm'];
    console.log(`[ASSIGN] ${handName} FBX proximity: ${assignment.size} verts — ${names.map((n, i) => n + ':' + counts[i]).join(', ')}`);
    return assignment;
}

// Keep the old Z-band boundary approach as fallback
function findFingerBoundaries(positions, handVerts, isLeft) {
    const handName = isLeft ? 'left' : 'right';
    const sign = isLeft ? -1 : 1; // left=-X tips, right=+X tips

    // Finger verts: above thumb Y threshold
    const fingerVerts = handVerts.filter(i => positions[i * 3 + 1] > 0.185);
    const xVals = fingerVerts.map(i => positions[i * 3]);
    const xMin = Math.min(...xVals);
    const xMax = Math.max(...xVals);
    console.log(`[BOUNDS] ${handName}: ${fingerVerts.length} finger verts, X=[${(xMin*1000).toFixed(0)}, ${(xMax*1000).toFixed(0)}]mm`);

    // Step 1: Build X-extent profile (most extreme X per 2mm Z band)
    const zBands = new Map();
    for (const vi of fingerVerts) {
        const x = positions[vi * 3] * 1000;
        const z = Math.round(positions[vi * 3 + 2] * 500) * 2; // 2mm bins
        const current = zBands.get(z);
        // "Most extreme" = most negative X for left hand, most positive for right
        if (!current || (isLeft ? x < current : x > current)) {
            zBands.set(z, x);
        }
    }

    const profile = [...zBands.entries()].sort((a, b) => a[0] - b[0])
        .filter(([z]) => z >= -95 && z <= -5); // finger Z range only

    console.log(`[BOUNDS] ${handName}: X-extent profile (${profile.length} Z bands)`);

    // Step 2: Find finger peaks in X-extent profile
    // Smooth with ±4mm window, then find peaks (most extreme X = local minima for left, maxima for right)
    const smoothedX = profile.map(([z]) => {
        let sum = 0, count = 0;
        for (const [z2, x2] of profile) {
            if (Math.abs(z2 - z) <= 4) { sum += x2; count++; }
        }
        return { z, x: sum / count };
    });

    // Find peaks: where X is most extreme (most negative for left)
    const fingerPeaks = [];
    for (let i = 1; i < smoothedX.length - 1; i++) {
        const isPeak = isLeft
            ? (smoothedX[i].x <= smoothedX[i - 1].x && smoothedX[i].x <= smoothedX[i + 1].x)
            : (smoothedX[i].x >= smoothedX[i - 1].x && smoothedX[i].x >= smoothedX[i + 1].x);
        if (isPeak) {
            // Must reach at least 85% of the overall extreme to be a real finger tip
            const overall = isLeft ? Math.min(...smoothedX.map(s => s.x)) : Math.max(...smoothedX.map(s => s.x));
            const reach = isLeft ? smoothedX[i].x / overall : smoothedX[i].x / overall;
            if (reach > 0.95) {
                fingerPeaks.push(smoothedX[i]);
            }
        }
    }

    // Deduplicate nearby peaks (keep the most extreme within 10mm Z)
    fingerPeaks.sort((a, b) => isLeft ? a.x - b.x : b.x - a.x); // most extreme first
    const distinctPeaks = [];
    for (const p of fingerPeaks) {
        if (!distinctPeaks.some(d => Math.abs(d.z - p.z) < 10)) {
            distinctPeaks.push(p);
        }
    }
    distinctPeaks.sort((a, b) => a.z - b.z);

    console.log(`[BOUNDS] ${handName}: finger peaks (${distinctPeaks.length}):`);
    for (const p of distinctPeaks) {
        console.log(`  Z=${p.z}mm X=${p.x.toFixed(0)}mm`);
    }

    // Step 3: Find boundaries using the X-extent profile valleys.
    //
    // Strategy:
    // - The X-extent profile shows how far each Z-band reaches toward the tips
    // - Between fingers, the mesh retreats toward the palm (valleys in profile)
    // - We find the 2 deepest valleys = ring-middle and middle-index boundaries
    // - For pinky-ring: the two fingers have similar length and form a merged peak.
    //   We find the pinky-ring boundary by looking for a cross-section Z gap
    //   within the merged peak's Z range.

    const tipX = isLeft ? xMin * 1000 : xMax * 1000;
    const palmX = isLeft ? xMax * 1000 : xMin * 1000;

    // Find valleys in the smoothed X-extent profile
    // Valley = local maximum in X (less extreme = more toward palm) compared to neighbors
    const valleys = [];
    for (let i = 2; i < smoothedX.length - 2; i++) {
        const { z, x } = smoothedX[i];
        // For left hand: valley = less negative X (higher value) than both sides
        // For right hand: valley = less positive X (lower value) than both sides
        const isValley = isLeft
            ? (x > smoothedX[i - 2].x && x > smoothedX[i + 2].x)
            : (x < smoothedX[i - 2].x && x < smoothedX[i + 2].x);
        if (!isValley) continue;

        // How much does the profile retreat from the overall extreme?
        const overallExtreme = isLeft
            ? Math.min(...smoothedX.map(s => s.x))
            : Math.max(...smoothedX.map(s => s.x));
        const retreat = isLeft ? (x - overallExtreme) : (overallExtreme - x);

        if (retreat > 5) { // at least 5mm retreat from peak to count as valley
            valleys.push({ z, x, retreat });
        }
    }

    // Deduplicate valleys within 6mm Z
    valleys.sort((a, b) => b.retreat - a.retreat);
    const distinctValleys = [];
    for (const v of valleys) {
        if (!distinctValleys.some(d => Math.abs(d.z - v.z) < 6)) {
            distinctValleys.push(v);
        }
    }
    distinctValleys.sort((a, b) => a.z - b.z);

    console.log(`[BOUNDS] ${handName}: X-extent valleys (${distinctValleys.length}):`);
    for (const v of distinctValleys) {
        console.log(`  Z=${v.z}mm retreat=${v.retreat.toFixed(0)}mm`);
    }

    // Take the 2 deepest valleys as ring-middle and middle-index
    const sortedByRetreat = [...distinctValleys].sort((a, b) => b.retreat - a.retreat);
    const mainValleys = sortedByRetreat.slice(0, 2).sort((a, b) => a.z - b.z);

    if (mainValleys.length < 2) {
        console.log(`[BOUNDS] ${handName}: WARNING — fewer than 2 X-extent valleys found`);
        return { boundaries: [], fingerCenters: [] };
    }

    console.log(`[BOUNDS] ${handName}: main valleys: Z=${mainValleys[0].z}mm (${mainValleys[0].retreat.toFixed(0)}mm), Z=${mainValleys[1].z}mm (${mainValleys[1].retreat.toFixed(0)}mm)`);

    // Step 4: Find missing boundaries via cross-section gap analysis.
    //
    // The 2 X-extent valleys divide fingers into 3 groups. Each group
    // could contain 1 or 2 merged fingers. For groups wider than 20mm
    // in Z span, search for a cross-section gap to split them.
    //
    // Expected structure: [pinky] valley [ring+middle] valley [index]
    // The ring-middle gap doesn't show in X-extent because both fingers
    // have similar reach. We find it via cross-section within the merged group.

    const allZ = fingerVerts.map(i => positions[i * 3 + 2] * 1000);
    const fingerZMin = Math.min(...allZ);
    const fingerZMax = Math.max(...allZ);

    // Define the 3 groups from the 2 valleys
    const groups = [
        { name: 'group0', zMin: fingerZMin, zMax: mainValleys[0].z },
        { name: 'group1', zMin: mainValleys[0].z, zMax: mainValleys[1].z },
        { name: 'group2', zMin: mainValleys[1].z, zMax: fingerZMax },
    ];

    for (const g of groups) {
        g.span = g.zMax - g.zMin;
        console.log(`[BOUNDS] ${handName}: ${g.name} Z=[${g.zMin.toFixed(0)}, ${g.zMax.toFixed(0)}] span=${g.span.toFixed(0)}mm`);
    }

    // Find the ring-middle boundary within the middle group (group1).
    // This is the group BETWEEN the two X-extent valleys — it contains
    // ring+middle merged because both fingers have similar X-reach.
    // Groups 0 and 2 (edge groups) are single fingers (pinky, index).

    function findSubBoundary(groupZMin, groupZMax, label) {
        const gapVotes = new Map();

        for (let pct = 0.05; pct <= 0.35; pct += 0.01) {
            const xMm = tipX + (palmX - tipX) * pct;
            const sliceZ = [];
            for (const vi of fingerVerts) {
                const x = positions[vi * 3] * 1000;
                const z = positions[vi * 3 + 2] * 1000;
                if (Math.abs(x - xMm) < 2 && z > groupZMin && z < groupZMax) {
                    sliceZ.push(z);
                }
            }
            if (sliceZ.length < 4) continue;
            sliceZ.sort((a, b) => a - b);

            for (let i = 1; i < sliceZ.length; i++) {
                const gap = sliceZ[i] - sliceZ[i - 1];
                if (gap > 3) {
                    const zMid = Math.round((sliceZ[i] + sliceZ[i - 1]) / 2);
                    gapVotes.set(zMid, (gapVotes.get(zMid) || 0) + 1);
                }
            }
        }

        // Filter to interior only (>5mm from group edges)
        const candidates = [...gapVotes.entries()]
            .filter(([z]) => z > groupZMin + 5 && z < groupZMax - 5)
            .sort((a, b) => b[1] - a[1]);

        if (candidates.length === 0) return null;

        // Cluster nearby candidates (within 3mm)
        const clusters = [];
        const usedSet = new Set();
        for (const [z, votes] of candidates) {
            if (usedSet.has(z)) continue;
            let totalVotes = votes;
            let totalZ = z * votes;
            usedSet.add(z);
            for (const [z2, v2] of candidates) {
                if (!usedSet.has(z2) && Math.abs(z2 - z) <= 3) {
                    totalVotes += v2;
                    totalZ += z2 * v2;
                    usedSet.add(z2);
                }
            }
            clusters.push({ z: totalZ / totalVotes, votes: totalVotes });
        }
        clusters.sort((a, b) => b.votes - a.votes);

        console.log(`[BOUNDS] ${handName}: ${label} gap candidates:`);
        for (const c of clusters.slice(0, 5)) {
            console.log(`  Z=${c.z.toFixed(1)}mm (${c.votes} votes)${c === clusters[0] ? ' ← selected' : ''}`);
        }

        return clusters[0];
    }

    // Only split group1 (between the two valleys) — it contains ring+middle
    const middleGroup = groups[1];
    const ringMiddleSub = findSubBoundary(middleGroup.zMin, middleGroup.zMax, 'ring-middle');

    let subZ;
    if (!ringMiddleSub || ringMiddleSub.votes < 2) {
        console.log(`[BOUNDS] ${handName}: WARNING — could not find ring-middle boundary, using midpoint`);
        subZ = (middleGroup.zMin + middleGroup.zMax) / 2;
    } else {
        subZ = ringMiddleSub.z;
    }

    // Assemble: valley0 (pinky|ring), sub (ring|middle), valley1 (middle|index)
    const bestGaps = [
        { z: mainValleys[0].z, width: mainValleys[0].retreat },
        { z: subZ, width: 0 },
        { z: mainValleys[1].z, width: mainValleys[1].retreat },
    ].sort((a, b) => a.z - b.z);

    console.log(`[BOUNDS] ${handName}: inter-finger boundaries:`);
    const fingerNames = ['pinky', 'ring', 'middle', 'index'];
    for (let i = 0; i < bestGaps.length; i++) {
        console.log(`  ${fingerNames[i]}|${fingerNames[i + 1]}: Z=${bestGaps[i].z.toFixed(1)}mm (gap ${bestGaps[i].width.toFixed(1)}mm)`);
    }

    // Compute finger centers as midpoints between boundaries
    const fingerCenters = [
        (fingerZMin + bestGaps[0].z) / 2,           // pinky center
        (bestGaps[0].z + bestGaps[1].z) / 2,        // ring center
        (bestGaps[1].z + bestGaps[2].z) / 2,        // middle center
        (bestGaps[2].z + fingerZMax) / 2,            // index center
    ];

    console.log(`[BOUNDS] ${handName}: finger centers: ${fingerCenters.map(z => z.toFixed(1) + 'mm').join(', ')}`);

    return {
        boundaries: bestGaps.map(g => g.z / 1000), // convert to meters
        fingerCenters: fingerCenters.map(z => z / 1000),
    };
}

// ═══════════════════════════════════════════════════════════════════
// 4. ASSIGN VERTICES TO FINGERS BY Z-POSITION
// ═══════════════════════════════════════════════════════════════════

/**
 * Assigns each hand vertex to one of 4 fingers based on Z position.
 * Uses the 3 boundary Z positions to divide the Z range into 4 regions.
 * Returns assignment map: vertex index → finger index (0-3 = pinky-index).
 */
function assignFingersByZ(positions, handVerts, boundaries, isLeft) {
    const handName = isLeft ? 'left' : 'right';
    const assignment = new Map(); // vertex → finger index (0=pinky, 1=ring, 2=middle, 3=index)

    for (const vi of handVerts) {
        const y = positions[vi * 3 + 1];
        if (y < 0.185) continue; // skip thumb region

        const z = positions[vi * 3 + 2];
        let fingerIdx;
        if (z < boundaries[0]) fingerIdx = 0;       // pinky
        else if (z < boundaries[1]) fingerIdx = 1;   // ring
        else if (z < boundaries[2]) fingerIdx = 2;   // middle
        else fingerIdx = 3;                           // index

        assignment.set(vi, fingerIdx);
    }

    const counts = [0, 0, 0, 0];
    for (const [, fi] of assignment) counts[fi]++;
    const names = ['pinky', 'ring', 'middle', 'index'];
    console.log(`[ASSIGN] ${handName}: ${assignment.size} verts assigned — ${names.map((n, i) => n + ':' + counts[i]).join(', ')}`);

    return assignment;
}

// ═══════════════════════════════════════════════════════════════════
// 5. FINGER NAMES (index mapping)
// ═══════════════════════════════════════════════════════════════════

// Finger indices are already ordered: 0=pinky, 1=ring, 2=middle, 3=index
// (from most negative Z to most positive Z)
const FINGER_NAMES = ['pinky', 'ring', 'middle', 'index'];

// ═══════════════════════════════════════════════════════════════════
// 6. COMPUTE CENTERLINES
// ═══════════════════════════════════════════════════════════════════

function computeCenterline(positions, assignment, clusterIdx, isLeft, xClipMin, xClipMax, useYAxis) {
    // Gather all vertices assigned to this cluster
    const allVerts = [];
    for (const [vi, ci] of assignment) {
        if (ci !== clusterIdx) continue;
        allVerts.push(vi);
    }
    if (allVerts.length === 0) return null;

    // Clip along primary axis for centerline computation
    // For fingers: clip along X. For thumb: clip along Y.
    const clipAxis = useYAxis ? 1 : 0;
    let verts = allVerts;
    if (xClipMin !== undefined || xClipMax !== undefined) {
        verts = allVerts.filter(vi => {
            const v = positions[vi * 3 + clipAxis];
            if (xClipMin !== undefined && v < xClipMin) return false;
            if (xClipMax !== undefined && v > xClipMax) return false;
            return true;
        });
    }
    if (verts.length === 0) return null;

    // For thumb, slice along Y axis (thumb extends down -Y)
    // For fingers, slice along X axis (fingers extend along arm axis)
    const axisIdx = useYAxis ? 1 : 0;
    const axisVals = verts.map(v => positions[v * 3 + axisIdx]);
    const axisMin = Math.min(...axisVals);
    const axisMax = Math.max(...axisVals);

    // Cross-sections every 2mm along the primary axis
    // Use bounding-box center (not centroid) to avoid bias from uneven vertex density
    //
    // For thumb with thenar (>500 verts): tube-tracking from tip.
    // The thenar muscle verts outnumber tube verts 2:1 and skew the bbox center
    // toward the palm. Tube-tracking from the tip (where there's no thenar)
    // keeps only verts within 15mm of the previous slice's center, following the
    // actual digit tube. All verts still display as thumb (red).
    const centerline = [];
    const needsTracking = useYAxis && verts.length > 500;
    let prevCX = null, prevCZ = null;
    const trackRadius = 0.015; // 15mm
    let tubeKnuckle = null;
    for (let a = axisMin; a <= axisMax; a += 0.002) {
        const rawSlice = verts.filter(v => Math.abs(positions[v * 3 + axisIdx] - a) < 0.003);
        let sliceVerts = rawSlice;
        if (sliceVerts.length < 2) continue;

        if (needsTracking && prevCX !== null) {
            const filtered = sliceVerts.filter(v => {
                const vx = positions[v * 3], vz = positions[v * 3 + 2];
                return Math.abs(vx - prevCX) < trackRadius && Math.abs(vz - prevCZ) < trackRadius;
            });
            if (filtered.length >= 2) sliceVerts = filtered;

            // Detect knuckle: where raw/tube ratio first exceeds 2.0
            if (!tubeKnuckle && rawSlice.length > sliceVerts.length * 2) {
                if (centerline.length > 0) {
                    tubeKnuckle = { bboxCenter: [...centerline[centerline.length - 1]] };
                    console.log(`    [THUMB KNUCKLE] detected at Y=${(a*1000).toFixed(1)}mm (raw=${rawSlice.length} vs tube=${sliceVerts.length}), knuckle bbox=[${tubeKnuckle.bboxCenter.map(v=>(v*1000).toFixed(1))}]`);
                }
            }
        }

        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;
        let zMin = Infinity, zMax = -Infinity;
        for (const v of sliceVerts) {
            const vx = positions[v * 3], vy = positions[v * 3 + 1], vz = positions[v * 3 + 2];
            if (vx < xMin) xMin = vx; if (vx > xMax) xMax = vx;
            if (vy < yMin) yMin = vy; if (vy > yMax) yMax = vy;
            if (vz < zMin) zMin = vz; if (vz > zMax) zMax = vz;
        }
        const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2, cz = (zMin + zMax) / 2;
        centerline.push([cx, cy, cz]);
        if (needsTracking) { prevCX = cx; prevCZ = cz; }
    }

    if (centerline.length < 3) return { centerline, length: 0 };

    // Multiple passes of 3-point moving average smoothing
    let smoothed = [...centerline.map(p => [...p])];
    for (let pass = 0; pass < 5; pass++) {
        const next = [smoothed[0]];
        for (let i = 1; i < smoothed.length - 1; i++) {
            next.push([
                (smoothed[i - 1][0] + smoothed[i][0] + smoothed[i + 1][0]) / 3,
                (smoothed[i - 1][1] + smoothed[i][1] + smoothed[i + 1][1]) / 3,
                (smoothed[i - 1][2] + smoothed[i][2] + smoothed[i + 1][2]) / 3,
            ]);
        }
        next.push(smoothed[smoothed.length - 1]);
        smoothed = next;
    }

    let totalLen = 0;
    for (let i = 1; i < smoothed.length; i++) {
        const dx = smoothed[i][0] - smoothed[i - 1][0];
        const dy = smoothed[i][1] - smoothed[i - 1][1];
        const dz = smoothed[i][2] - smoothed[i - 1][2];
        totalLen += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    return { centerline: smoothed, length: totalLen, clippedVerts: allVerts, tubeKnuckle };
}

// ═══════════════════════════════════════════════════════════════════
// 7. PLACE JOINTS ALONG CENTERLINES
// ═══════════════════════════════════════════════════════════════════

function placeJointsOnCenterline(centerline, isThumb) {
    // Palm end = closer to wrist.
    // For fingers: palm = less extreme |X|
    // For thumb: palm = higher Y (palm plane), tip = lower Y
    let palmToTip;
    if (isThumb) {
        const y0 = centerline[0][1];
        const yN = centerline[centerline.length - 1][1];
        palmToTip = y0 > yN ? [...centerline] : [...centerline].reverse();
    } else {
        const x0 = Math.abs(centerline[0][0]);
        const xN = Math.abs(centerline[centerline.length - 1][0]);
        palmToTip = x0 < xN ? [...centerline] : [...centerline].reverse();
    }

    // Cumulative distances
    const cumDist = [0];
    for (let i = 1; i < palmToTip.length; i++) {
        const dx = palmToTip[i][0] - palmToTip[i - 1][0];
        const dy = palmToTip[i][1] - palmToTip[i - 1][1];
        const dz = palmToTip[i][2] - palmToTip[i - 1][2];
        cumDist.push(cumDist[i - 1] + Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
    const totalLen = cumDist[cumDist.length - 1];

    const fractions = [0.10, 0.38, 0.62, 0.90];
    const joints = [];
    for (const frac of fractions) {
        const targetDist = frac * totalLen;
        let segIdx = 0;
        for (let i = 1; i < cumDist.length; i++) {
            if (cumDist[i] >= targetDist) { segIdx = i - 1; break; }
            if (i === cumDist.length - 1) segIdx = i - 1;
        }
        const segLen = cumDist[segIdx + 1] - cumDist[segIdx];
        const t = segLen > 0 ? (targetDist - cumDist[segIdx]) / segLen : 0;
        const p0 = palmToTip[segIdx], p1 = palmToTip[Math.min(segIdx + 1, palmToTip.length - 1)];
        joints.push([
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2]),
        ]);
    }
    return joints;
}

// ═══════════════════════════════════════════════════════════════════
// PROCESS ONE HAND
// ═══════════════════════════════════════════════════════════════════

function processHand(positions, indices, handVerts, isLeft) {
    const handName = isLeft ? 'left' : 'right';
    const side = isLeft ? 'l' : 'r';
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Processing ${handName.toUpperCase()} HAND (${handVerts.length} vertices)`);
    console.log('='.repeat(60));

    let assignment = null;
    let boundaries = [];
    let assignMethod = 'none';

    // Priority 1: Proximity assignment using best available reference (placed_joints > FBX)
    const refFingers = loadFingerReference(side);
    if (refFingers) {
        assignment = assignByFBXProximity(positions, handVerts, refFingers, isLeft);
        assignMethod = 'proximity_ref';
        console.log(`[ASSIGN] Using proximity reference assignment`);
    }

    // Priority 2: Load algorithm/corrected assignment (with blend weights for overlap zones)
    // Checks: finger_assignment_corrected.json → finger_assignment.json → finger_assignment (1).json
    const corrected = loadCorrectedAssignment(isLeft);
    if (corrected) {
        assignment = corrected;
        assignMethod = 'corrected_or_algo';
        console.log(`[ASSIGN] Overriding with loaded assignment (${assignment.size} verts)`);
    }

    // Priority 3: Z-band boundaries (fallback)
    if (!assignment) {
        console.log(`[ASSIGN] Falling back to Z-band boundary detection`);
        const result = findFingerBoundaries(positions, handVerts, isLeft);
        boundaries = result.boundaries;
        if (boundaries.length >= 3) {
            assignment = assignFingersByZ(positions, handVerts, boundaries, isLeft);
            assignMethod = 'z_band';

            // Add thumb detection for Z-band mode
            for (const vi of handVerts) {
                const x = positions[vi * 3], y = positions[vi * 3 + 1];
                const absX = Math.abs(x);
                if (absX > 0.39 && absX < 0.43 && y < 0.185) {
                    assignment.set(vi, 4);
                }
            }
        }
    }

    if (!assignment || assignment.size === 0) {
        console.log(`[ERROR] ${handName}: no assignment method succeeded`);
        return {
            handName, side, isLeft, boundaries: [],
            assignment: new Map(), clusterToName: new Map(),
            centerlines: {}, jointPositions: {},
        };
    }

    // Set up cluster-to-name mapping
    const clusterToName = new Map();
    const names = ['pinky', 'ring', 'middle', 'index', 'thumb', 'palm'];
    for (let i = 0; i < 6; i++) clusterToName.set(i, names[i]);

    // Add unassigned hand verts as palm (index 5) so they appear in the editor
    for (const vi of handVerts) {
        if (!assignment.has(vi)) assignment.set(vi, 5); // 5 = palm
    }

    // Count per finger
    const counts = [0, 0, 0, 0, 0, 0];
    for (const [, fi] of assignment) counts[fi]++;
    console.log(`[ASSIGN] ${handName} (${assignMethod}): ${names.map((n, i) => n + ':' + counts[i]).join(', ')}`);

    // Compute centerlines and place joints
    const fingerXClipThresh = 0.41;
    const centerlines = {};
    const jointPositions = {};

    for (const [fingerIdx, name] of clusterToName) {
        // Skip palm — no centerline or joints needed
        if (name === 'palm') continue;
        console.log(`\n--- ${handName} ${name} ---`);

        let xClipMin, xClipMax;
        if (name === 'thumb') {
            // Clip thumb at wrist Y — thumb can't extend past wrist
            const wristKey = side + '_wrist';
            const pjData = JSON.parse(fs.readFileSync(JOINTS_PATH, 'utf8'));
            const wristY = pjData.P[wristKey] ? pjData.P[wristKey][1] : undefined;
            xClipMin = undefined;
            xClipMax = wristY; // applied to Y axis via clipAxis in computeCenterline
        } else {
            xClipMin = isLeft ? undefined : fingerXClipThresh;
            xClipMax = isLeft ? -fingerXClipThresh : undefined;
        }

        const useYAxis = (name === 'thumb');
        const cl = computeCenterline(positions, assignment, fingerIdx, isLeft, xClipMin, xClipMax, useYAxis);
        if (!cl || cl.centerline.length < 5) {
            console.log(`[WARN] Centerline too short for ${name} (${cl ? cl.centerline.length : 0} pts)`);
            continue;
        }
        centerlines[name] = cl;
        console.log(`[CL] ${name}: ${cl.centerline.length} pts, length ${(cl.length * 1000).toFixed(1)}mm`);

        const joints = placeJointsOnCenterline(cl.centerline, name === 'thumb');

        let jointNames;
        if (name === 'thumb') {
            jointNames = [`${side}_thumb1`, `${side}_thumb2`, `${side}_thumb3`];
        } else if (name === 'middle') {
            jointNames = [`${side}_mid_knuckle`, `${side}_middle2`, `${side}_middle3`];
        } else {
            jointNames = [`${side}_${name}1`, `${side}_${name}2`, `${side}_${name}3`];
        }

        for (let j = 0; j < 3; j++) {
            jointPositions[jointNames[j]] = joints[j];
            console.log(`  ${jointNames[j]}: [${joints[j].map(v => (v * 1000).toFixed(2)).join(', ')}] mm`);
        }
    }

    return {
        handName, side, isLeft,
        boundaries,
        assignment, clusterToName,
        centerlines, jointPositions,
    };
}

// ═══════════════════════════════════════════════════════════════════
// 11-12. UPDATE placed_joints.json
// ═══════════════════════════════════════════════════════════════════

function updatePlacedJoints(leftResult, rightResult) {
    const data = JSON.parse(fs.readFileSync(JOINTS_PATH, 'utf8'));
    const oldP = { ...data.P };

    console.log(`\n${'='.repeat(60)}`);
    console.log('UPDATING placed_joints.json');
    console.log('='.repeat(60));

    const allJoints = { ...leftResult.jointPositions, ...rightResult.jointPositions };

    for (const [name, pos] of Object.entries(allJoints)) {
        const oldPos = oldP[name];
        data.P[name] = pos;

        if (oldPos) {
            const delta = Math.sqrt(
                (pos[0] - oldPos[0]) ** 2 + (pos[1] - oldPos[1]) ** 2 + (pos[2] - oldPos[2]) ** 2
            ) * 1000;
            console.log(`  ${name.padEnd(16)} [${pos.map(v => (v * 1000).toFixed(1)).join(', ')}] mm  delta: ${delta.toFixed(2)}mm`);
        } else {
            console.log(`  ${name.padEnd(16)} [${pos.map(v => (v * 1000).toFixed(1)).join(', ')}] mm  (NEW)`);
        }
    }

    fs.writeFileSync(JOINTS_PATH, JSON.stringify(data, null, 2));
    console.log(`\nWrote ${Object.keys(allJoints).length} finger joints to ${JOINTS_PATH}`);
    return data;
}

// ═══════════════════════════════════════════════════════════════════
// 12b. SAVE ASSIGNMENT DATA FOR INTERACTIVE EDITOR
// ═══════════════════════════════════════════════════════════════════

// Load existing blend weights from finger_assignment.json before overwriting
let _existingBlends = null;
function loadExistingBlends() {
    if (_existingBlends) return _existingBlends;
    try {
        const data = JSON.parse(fs.readFileSync(path.join(BASE_DIR, 'finger_assignment.json'), 'utf8'));
        _existingBlends = new Map();
        for (const v of (data.left || [])) {
            if (v.blend) _existingBlends.set(v.vi, v.blend);
        }
        for (const v of (data.right || [])) {
            if (v.blend) _existingBlends.set(v.vi, v.blend);
        }
    } catch { _existingBlends = new Map(); }
    return _existingBlends;
}

function saveAssignmentData(positions, leftResult, rightResult) {
    const ASSIGN_PATH = path.join(BASE_DIR, 'finger_assignment.json');
    const fingerNames = ['pinky', 'ring', 'middle', 'index', 'thumb'];
    const blends = loadExistingBlends();

    function buildHandData(result) {
        const verts = [];
        for (const [vi, fi] of result.assignment) {
            const entry = {
                vi,
                x: positions[vi * 3],
                y: positions[vi * 3 + 1],
                z: positions[vi * 3 + 2],
                finger: result.clusterToName.get(fi) || fingerNames[fi] || 'unassigned',
            };
            // Preserve blend weights from valley_detect.js output
            const bw = blends.get(vi);
            if (bw) entry.blend = bw;
            verts.push(entry);
        }
        return verts;
    }

    const data = {
        left: buildHandData(leftResult),
        right: buildHandData(rightResult),
        boundaries: {
            left: leftResult.boundaries,
            right: rightResult.boundaries,
        },
    };

    fs.writeFileSync(ASSIGN_PATH, JSON.stringify(data));
    const blendCount = data.left.concat(data.right).filter(v => v.blend).length;
    console.log(`\nWrote assignment data to ${ASSIGN_PATH} (${data.left.length + data.right.length} verts, ${blendCount} with blend weights)`);
}

// ═══════════════════════════════════════════════════════════════════
// 13. DEBUG VISUALIZATION HTML
// ═══════════════════════════════════════════════════════════════════

function generateDebugHTML(positions, indices, leftResult, rightResult) {
    function gatherVertexData(result, positions) {
        const data = [];
        for (const [vi, ci] of result.assignment) {
            const name = result.clusterToName.get(ci) || 'unassigned';
            data.push({
                x: positions[vi * 3], y: positions[vi * 3 + 1], z: positions[vi * 3 + 2],
                finger: name
            });
        }
        return data;
    }

    function gatherCenterlines(result) {
        const cls = {};
        for (const [name, cl] of Object.entries(result.centerlines)) {
            cls[name] = cl.centerline;
        }
        return cls;
    }

    function computeStats(result) {
        const counts = {};
        for (const [, ci] of result.assignment) {
            const name = result.clusterToName.get(ci) || 'unassigned';
            counts[name] = (counts[name] || 0) + 1;
        }
        const clLengths = {};
        for (const [name, cl] of Object.entries(result.centerlines)) {
            clLengths[name] = (cl.length * 1000).toFixed(1);
        }
        return { counts, clLengths, joints: result.jointPositions };
    }

    const leftVerts = gatherVertexData(leftResult, positions);
    const rightVerts = gatherVertexData(rightResult, positions);
    const leftCLs = gatherCenterlines(leftResult);
    const rightCLs = gatherCenterlines(rightResult);
    const leftStats = computeStats(leftResult);
    const rightStats = computeStats(rightResult);

    const html = `<!DOCTYPE html>
<html>
<head>
<title>Finger BFS Debug Visualization</title>
<style>
body { margin: 0; overflow: hidden; background: #1a1a2e; font-family: monospace; color: #eee; }
#stats {
    position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.85);
    padding: 15px; border-radius: 8px; font-size: 11px; max-height: 90vh;
    overflow-y: auto; z-index: 10; min-width: 380px;
}
#stats h2 { margin: 5px 0; color: #ffcc00; font-size: 14px; }
#stats h3 { margin: 8px 0 3px; color: #88ccff; font-size: 12px; }
.fc { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
table { border-collapse: collapse; margin: 4px 0; }
td { padding: 1px 6px; }
.jp { color: #aaffaa; font-size: 10px; }
#controls {
    position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.85);
    padding: 8px; border-radius: 8px; z-index: 10;
}
button { margin: 2px; padding: 4px 8px; cursor: pointer; background: #333; color: #eee; border: 1px solid #555; border-radius: 3px; font-size: 11px; }
button:hover { background: #555; }
select { background: #333; color: #eee; border: 1px solid #555; padding: 4px; font-size: 11px; }
</style>
</head>
<body>
<div id="stats"><h2>Finger BFS Debug</h2><div id="sc"></div></div>
<div id="controls">
    <button onclick="setView('top')">Top (Y)</button>
    <button onclick="setView('front')">Front (Z)</button>
    <button onclick="setView('side')">Side (X)</button>
    <select onchange="focusHand(this.value)">
        <option value="left">Left Hand</option>
        <option value="right">Right Hand</option>
        <option value="both">Both</option>
    </select>
    <button onclick="toggle('webbing')">Webbing</button>
    <button onclick="toggle('cl')">Centerlines</button>
    <button onclick="toggle('joints')">Joints</button>
    <button onclick="toggle('labels')">Labels</button>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"><\/script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"><\/script>
<script>
const D = {
    lv: ${JSON.stringify(leftVerts)},
    rv: ${JSON.stringify(rightVerts)},
    lb: ${JSON.stringify(leftResult.boundaries.map(b => b * 1000))},
    rb: ${JSON.stringify(rightResult.boundaries.map(b => b * 1000))},
    lc: ${JSON.stringify(leftCLs)},
    rc: ${JSON.stringify(rightCLs)},
    lj: ${JSON.stringify(leftResult.jointPositions)},
    rj: ${JSON.stringify(rightResult.jointPositions)},
    ls: ${JSON.stringify(leftStats)},
    rs: ${JSON.stringify(rightStats)}
};
const FC = { index: 0xff6600, middle: 0x00cc00, ring: 0x0066ff, pinky: 0xcc00cc, thumb: 0xff0000, unassigned: 0x666666 };
const FCC = { index:'#ff6600', middle:'#00cc00', ring:'#0066ff', pinky:'#cc00cc', thumb:'#ff0000', unassigned:'#666' };

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.001, 10);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const ctrl = new THREE.OrbitControls(camera, renderer.domElement);
ctrl.enableDamping = true;

const groups = {};

function addPts(verts) {
    const p = new Float32Array(verts.length*3), c = new Float32Array(verts.length*3);
    for (let i = 0; i < verts.length; i++) {
        p[i*3]=verts[i].x; p[i*3+1]=verts[i].y; p[i*3+2]=verts[i].z;
        const col = new THREE.Color(FC[verts[i].finger]||0x666666);
        c[i*3]=col.r; c[i*3+1]=col.g; c[i*3+2]=col.b;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setAttribute('color', new THREE.BufferAttribute(c, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({size:0.0015, vertexColors:true})));
}
addPts(D.lv); addPts(D.rv);

// Draw Z-boundary planes as semi-transparent rectangles
function addBoundaryPlanes(boundaries, xCenter, parent) {
    for (const zMm of boundaries) {
        const z = zMm / 1000;
        const g = new THREE.PlaneGeometry(0.08, 0.04);
        const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({color:0xff4444, transparent:true, opacity:0.3, side:THREE.DoubleSide}));
        m.position.set(xCenter, 0.205, z);
        m.rotation.y = Math.PI / 2; // face along X axis
        parent.add(m);
    }
}
groups.webbing = new THREE.Group();
addBoundaryPlanes(D.lb, -0.46, groups.webbing);
addBoundaryPlanes(D.rb, 0.46, groups.webbing);
scene.add(groups.webbing);

function addCLs(cls, parent) {
    for (const [name, pts] of Object.entries(cls)) {
        if (pts.length < 2) continue;
        const vecs = pts.map(p => new THREE.Vector3(p[0],p[1],p[2]));
        const g = new THREE.BufferGeometry().setFromPoints(vecs);
        parent.add(new THREE.Line(g, new THREE.LineBasicMaterial({color: FC[name]||0xffffff, linewidth:2})));
    }
}
groups.cl = new THREE.Group();
addCLs(D.lc, groups.cl); addCLs(D.rc, groups.cl);
scene.add(groups.cl);

groups.joints = new THREE.Group();
groups.labels = new THREE.Group();

function addJoints(joints, jp, lp) {
    for (const [name, pos] of Object.entries(joints)) {
        let f='unassigned';
        if(name.includes('index'))f='index'; else if(name.includes('mid')||name.includes('middle'))f='middle';
        else if(name.includes('ring'))f='ring'; else if(name.includes('pinky'))f='pinky';
        else if(name.includes('thumb'))f='thumb';
        const m = new THREE.Mesh(new THREE.SphereGeometry(0.002,8,8), new THREE.MeshBasicMaterial({color:FC[f]}));
        m.position.set(pos[0],pos[1],pos[2]); jp.add(m);
        const cv = document.createElement('canvas'); cv.width=256; cv.height=48;
        const cx = cv.getContext('2d'); cx.fillStyle='white'; cx.font='20px monospace'; cx.fillText(name,2,32);
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv),transparent:true}));
        sp.position.set(pos[0],pos[1]+0.006,pos[2]); sp.scale.set(0.025,0.006,1); lp.add(sp);
    }
}
addJoints(D.lj, groups.joints, groups.labels);
addJoints(D.rj, groups.joints, groups.labels);
scene.add(groups.joints); scene.add(groups.labels);

scene.add(new THREE.AmbientLight(0xffffff,1));
camera.position.set(-0.42, 0.35, -0.05);
ctrl.target.set(-0.42, 0.206, -0.05);
ctrl.update();

function setView(t) {
    const c = ctrl.target.clone();
    if(t==='top') camera.position.set(c.x, c.y+0.15, c.z);
    else if(t==='front') camera.position.set(c.x, c.y, c.z+0.15);
    else camera.position.set(c.x+0.15, c.y, c.z);
    ctrl.update();
}
function focusHand(h) {
    const cx = h==='left'?-0.44:h==='right'?0.44:0;
    ctrl.target.set(cx, 0.206, -0.05);
    camera.position.set(cx, 0.36, -0.05);
    ctrl.update();
}
function toggle(name) { if(groups[name]) groups[name].visible = !groups[name].visible; }

function buildStats() {
    let h = '';
    for (const [side, stats] of [['Left Hand', D.ls], ['Right Hand', D.rs]]) {
        h += '<h3>'+side+'</h3><table>';
        for (const [f, cnt] of Object.entries(stats.counts)) {
            h += '<tr><td><span class="fc" style="background:'+(FCC[f]||'#666')+'"></span>'+f+'</td><td>'+cnt+'v</td><td>'+(stats.clLengths[f]||'?')+'mm</td></tr>';
        }
        h += '</table><h3>'+side+' Joints</h3><table>';
        const j = side.startsWith('L') ? D.lj : D.rj;
        for (const [n, p] of Object.entries(j)) {
            h += '<tr><td>'+n+'</td><td class="jp">['+p.map(v=>(v*1000).toFixed(1)).join(', ')+']</td></tr>';
        }
        h += '</table>';
    }
    document.getElementById('sc').innerHTML = h;
}
buildStats();

(function animate(){requestAnimationFrame(animate);ctrl.update();renderer.render(scene,camera)})();
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
<\/script>
</body>
</html>`;

    fs.writeFileSync(DEBUG_HTML_PATH, html);
    console.log(`\nWrote debug HTML to ${DEBUG_HTML_PATH}`);
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

function main() {
    console.log('='.repeat(60));
    console.log('FINGER PLACEMENT - Mesh-based joint placement');
    console.log('='.repeat(60));

    const { positions, indices, vertexCount, triCount } = parseGLB(GLB_PATH);
    const { leftHand, rightHand } = identifyHandVertices(positions, vertexCount);

    const leftResult = processHand(positions, indices, leftHand, true);
    const rightResult = processHand(positions, indices, rightHand, false);

    updatePlacedJoints(leftResult, rightResult);
    generateDebugHTML(positions, indices, leftResult, rightResult);

    // Save assignment data for the interactive editor
    saveAssignmentData(positions, leftResult, rightResult);

    console.log('\nDone!');
}

main();
