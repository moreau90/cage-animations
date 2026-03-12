/**
 * Finger Assignment v8 — 95.0% finger-only accuracy
 *
 * Assigns each left-hand mesh vertex to a finger (pinky/ring/middle/index/thumb/palm).
 * Uses FBX bone positions as reference and mesh geometry for boundaries.
 *
 * Pipeline:
 *   Phase 1 — Coarse assignment
 *     1. Thumb separation: each vert assigned to thumb if closer to thumb bone
 *        polyline than to any finger bone (with 5% bias toward thumb)
 *     2. Y-valley detection: aggregate inter-finger Y-profile valleys across
 *        multiple X-slices (15%-80% of finger length) to find 3 boundaries
 *     3. Constrained K-means: geodesic Voronoi from finger centroids, with
 *        Z-zone constraints (9.5mm margin from valley boundaries)
 *     4. Z-gap correction: detect physical gaps between middle/index in Z
 *
 *   Phase 2 — Iterative per-slice tube-edge refinement (5 rounds)
 *     For each X-slice (2mm steps, knuckle→tip):
 *     1. Collect per-finger Z distributions from current labels
 *     2. Compute boundary = midpoint of adjacent fingers' tube edges
 *        - Pinky/ring: 92nd/8th percentile (resists contamination)
 *        - Other boundaries: 90th/10th percentile
 *     3. Monotonic enforcement at deep tips (X > 65% toward tip):
 *        boundaries can only decrease (move toward -Z) as X goes tipward,
 *        preventing contaminated labels from pulling boundary back up
 *     4. Reassign all finger verts using the per-slice boundaries
 *     Converges after ~5 rounds (oscillates with period 2; 5 ends on better state)
 *
 *   Post-processing:
 *     - Bone-extent correction: reassign verts beyond a finger's 5th-percentile
 *       X extent to the nearest reachable finger bone
 *     - Neighbor smoothing: 75% supermajority vote, up to 5 passes
 *     - Palm labeling: verts with X > knuckleX → palm
 *
 * Inputs:
 *   - mesh.glb: Mixamo mesh (positions + indices)
 *   - placed_joints.json: joint positions (for knuckle X reference)
 *   - fbx_mapped_fingers.json: FBX finger bone positions (4 joints per finger)
 *
 * Output:
 *   - finger_assignment.json: per-vertex finger labels for left hand
 *
 * Accuracy (vs hand-painted ground truth):
 *   - 95.0% finger-only (excluding thumb, which uses bone distance)
 *   - Global centroid offset: pinky 1.0mm, ring 1.9mm, middle 0.9mm, index 1.4mm
 *   - Remaining errors are in genuine Z-overlap zones between adjacent fingers
 *   - A perfect linear boundary gives only 94.1% on pinky/ring (theoretical limit)
 */

const fs = require('fs');
const path = require('path');
const BASE_DIR = __dirname;

function loadMesh() {
    const glb = fs.readFileSync(path.join(BASE_DIR, 'mesh.glb'));
    let offset = 12;
    const jsonLen = glb.readUInt32LE(offset); offset += 8;
    const gltf = JSON.parse(glb.toString('utf8', offset, offset + jsonLen));
    const binOffset = offset + jsonLen + 8;
    const posA = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION];
    const posV = gltf.bufferViews[posA.bufferView];
    const positions = new Float32Array(glb.buffer.slice(glb.byteOffset+binOffset+posV.byteOffset, glb.byteOffset+binOffset+posV.byteOffset+posV.byteLength));
    const idxA = gltf.accessors[gltf.meshes[0].primitives[0].indices];
    const idxV = gltf.bufferViews[idxA.bufferView];
    const indices = new Uint16Array(glb.buffer.slice(glb.byteOffset+binOffset+idxV.byteOffset, glb.byteOffset+binOffset+idxV.byteOffset+idxV.byteLength));
    return { positions, indices, vertCount: posA.count };
}

function buildAdjacency(positions, indices, vertCount) {
    const posMap = new Map(), canonMap = new Int32Array(vertCount);
    for (let i=0;i<vertCount;i++) { const key=positions[i*3].toFixed(6)+','+positions[i*3+1].toFixed(6)+','+positions[i*3+2].toFixed(6); if(!posMap.has(key))posMap.set(key,i); canonMap[i]=posMap.get(key); }
    const adj = new Map();
    for (let i=0;i<indices.length;i+=3) {
        const tri=[canonMap[indices[i]],canonMap[indices[i+1]],canonMap[indices[i+2]]];
        for (let a=0;a<3;a++) for (let b=a+1;b<3;b++) {
            if(tri[a]===tri[b]) continue;
            const va=tri[a],vb=tri[b];
            const d=Math.sqrt((positions[va*3]-positions[vb*3])**2+(positions[va*3+1]-positions[vb*3+1])**2+(positions[va*3+2]-positions[vb*3+2])**2);
            if(!adj.has(va))adj.set(va,new Map()); if(!adj.has(vb))adj.set(vb,new Map());
            const ex=adj.get(va).get(vb); if(ex===undefined||d<ex){adj.get(va).set(vb,d);adj.get(vb).set(va,d);}
        }
    }
    return { adj, canonMap };
}

function distToPolyline(px,py,pz,points) {
    let min=Infinity;
    for(let i=0;i<points.length-1;i++){const a=points[i],b=points[i+1];const abx=b[0]-a[0],aby=b[1]-a[1],abz=b[2]-a[2];const apx=px-a[0],apy=py-a[1],apz=pz-a[2];const l2=abx*abx+aby*aby+abz*abz;const t=l2>0?Math.max(0,Math.min(1,(apx*abx+apy*aby+apz*abz)/l2)):0;const cx=a[0]+t*abx,cy=a[1]+t*aby,cz=a[2]+t*abz;min=Math.min(min,Math.sqrt((px-cx)**2+(py-cy)**2+(pz-cz)**2));}
    return min;
}

function detectYValleys(positions, verts, sliceX, sliceHalfW) {
    const sv=[]; for(const ci of verts){if(Math.abs(positions[ci*3]-sliceX)<=sliceHalfW)sv.push(ci);} if(sv.length<20)return[];
    let zMin=Infinity,zMax=-Infinity; for(const ci of sv){const z=positions[ci*3+2];if(z<zMin)zMin=z;if(z>zMax)zMax=z;}
    const BW=0.001,bins=[],nB=Math.ceil((zMax-zMin)/BW);
    for(let bi=0;bi<nB;bi++){const bMin=zMin+bi*BW,bMax=bMin+BW;let mY=-Infinity,c=0;for(const ci of sv){const z=positions[ci*3+2];if(z>=bMin&&z<bMax){mY=Math.max(mY,positions[ci*3+1]);c++;}}if(c>0)bins.push({z:bMin+BW/2,maxY:mY});}
    const sm=[];for(let i=0;i<bins.length;i++){let s=bins[i].maxY,n=1;if(i>0){s+=bins[i-1].maxY;n++;}if(i<bins.length-1){s+=bins[i+1].maxY;n++;}sm.push({z:bins[i].z,y:s/n});}
    const v=[];for(let i=2;i<sm.length-2;i++){if(sm[i].y<sm[i-1].y&&sm[i].y<sm[i+1].y){const d=((Math.max(sm[i-2].y,sm[i-1].y)+Math.max(sm[i+1].y,sm[i+2].y))/2-sm[i].y)*1000;if(d>0.5)v.push({z:sm[i].z,depth:d});}}
    v.sort((a,b)=>b.depth-a.depth);return v;
}

function aggregateValleys(positions,verts,knX,tipX,expected) {
    const fracs=[0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8];
    const nB=expected.length,coll=Array(nB).fill(null).map(()=>[]);
    for(const f of fracs){const sx=knX+f*(tipX-knX);const vals=detectYValleys(positions,verts,sx,0.004);if(!vals.length)continue;const top=vals.slice(0,6);const m=[];for(let bi=0;bi<nB;bi++)for(let vi=0;vi<top.length;vi++)m.push({bi,vi,dist:Math.abs(top[vi].z-expected[bi])});m.sort((a,b)=>a.dist-b.dist);const uB=new Set(),uV=new Set();for(const{bi,vi,dist}of m){if(uB.has(bi)||uV.has(vi)||dist>0.015)continue;coll[bi].push({z:top[vi].z,depth:top[vi].depth});uB.add(bi);uV.add(vi);}}
    const bounds=[];for(let i=0;i<nB;i++){const s=coll[i];if(s.length>=3){let wZ=0,wT=0;for(const x of s){wZ+=x.z*x.depth;wT+=x.depth;}bounds.push(wZ/wT);console.log(`[VALLEY] ${i}: Z=${(wZ/wT*1000).toFixed(1)}mm (${s.length})`);}else{bounds.push(expected[i]);console.log(`[VALLEY] ${i}: fallback`);}}
    return bounds;
}

function geodesicVoronoi(seeds,adj,allVerts) {
    const gD=new Map(),assign=new Map(),pq=[];
    function push(i){pq.push(i);let k=pq.length-1;while(k>0){const p=(k-1)>>1;if(pq[p][0]<=pq[k][0])break;[pq[p],pq[k]]=[pq[k],pq[p]];k=p;}}
    function pop(){const t=pq[0];const l=pq.pop();if(pq.length>0){pq[0]=l;let i=0;while(true){let s=i,l2=2*i+1,r=2*i+2;if(l2<pq.length&&pq[l2][0]<pq[s][0])s=l2;if(r<pq.length&&pq[r][0]<pq[s][0])s=r;if(s===i)break;[pq[s],pq[i]]=[pq[i],pq[s]];i=s;}}return t;}
    for(const s of seeds)for(const ci of s.verts)push([0,ci,s.name]);
    while(pq.length>0){const[d,ci,n]=pop();if(gD.has(ci)&&gD.get(ci)<=d)continue;gD.set(ci,d);assign.set(ci,n);const nb=adj.get(ci);if(!nb)continue;for(const[ni,el]of nb){if(!allVerts.has(ni))continue;const nd=d+el;if(!gD.has(ni)||nd<gD.get(ni))push([nd,ni,n]);}}
    return assign;
}

function detectZGapCenter(positions, verts, sliceX, sliceHalfW, zMin, zMax) {
    const sv=[];
    for(const ci of verts){
        if(Math.abs(positions[ci*3]-sliceX)<=sliceHalfW){
            const z=positions[ci*3+2];
            if(z>=zMin && z<=zMax) sv.push(z);
        }
    }
    if(sv.length<4) return null;
    sv.sort((a,b)=>a-b);
    let maxGap=0, gapCenter=null;
    for(let i=1;i<sv.length;i++){
        const gap=sv[i]-sv[i-1];
        if(gap>maxGap){maxGap=gap;gapCenter=(sv[i]+sv[i-1])/2;}
    }
    return maxGap>0.002 ? {center:gapCenter, gap:maxGap} : null;
}

function findClosestVert(P,S,t){let m=Infinity,b=-1;for(const ci of S){const d=(P[ci*3]-t[0])**2+(P[ci*3+1]-t[1])**2+(P[ci*3+2]-t[2])**2;if(d<m){m=d;b=ci;}}return b;}
function reportCounts(l,a){const c={};for(const[,n]of a)c[n]=(c[n]||0)+1;console.log(`[${l}] ${Object.entries(c).map(([n,v])=>`${n}:${v}`).join(', ')}`);}

// ═══════════════════════════════════════════════════════════════════
console.log('='.repeat(60));
const { positions, indices, vertCount } = loadMesh();
const { adj, canonMap } = buildAdjacency(positions, indices, vertCount);
const joints = JSON.parse(fs.readFileSync(path.join(BASE_DIR,'placed_joints.json'),'utf8'));
let fbxRef=null; try{fbxRef=JSON.parse(fs.readFileSync(path.join(BASE_DIR,'fbx_mapped_fingers.json'),'utf8'));}catch(e){}

const side='l';
const handCanon=new Set();for(let i=0;i<vertCount;i++){if(positions[i*3]<-0.33)handCanon.add(canonMap[i]);}
const knuckleX=joints.P[`${side}_mid_knuckle`][0];
const bones=[];if(fbxRef){for(const n of['pinky','ring','middle','index','thumb']){const pts=[];for(let j=1;j<=4;j++){const p=fbxRef[`${side}_${n}${j}`];if(p)pts.push([...p]);}if(pts.length>=2)bones.push({name:n,points:pts});}}

const thumbBone=bones.find(b=>b.name==='thumb'), fingerBones=bones.filter(b=>b.name!=='thumb');
const fingerVerts=[],thumbVerts=[];
for(const ci of handCanon){const px=positions[ci*3],py=positions[ci*3+1],pz=positions[ci*3+2];if(thumbBone){const dT=distToPolyline(px,py,pz,thumbBone.points);let dF=Infinity;for(const fb of fingerBones)dF=Math.min(dF,distToPolyline(px,py,pz,fb.points));if(dT<dF*1.05){thumbVerts.push(ci);continue;}}fingerVerts.push(ci);}
console.log(`Thumb:${thumbVerts.length} Fingers:${fingerVerts.length}`);

const sortedFB=[...fingerBones].sort((a,b)=>a.points[0][2]-b.points[0][2]);
const fingerNames=sortedFB.map(b=>b.name);
const fbxMids=[];for(let i=0;i<sortedFB.length-1;i++)fbxMids.push((sortedFB[i].points[0][2]+sortedFB[i+1].points[0][2])/2);
const tipX=Math.min(...fingerVerts.map(ci=>positions[ci*3]));

// ===== PHASE 1: V6 baseline assignment =====
console.log('\n--- Phase 1: V6 baseline (valley + K-means + Z-gap) ---');
const valleyBounds=aggregateValleys(positions,fingerVerts,knuckleX,tipX,fbxMids);

// Z-gap correction for middle/index
const fracs=[0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8];
const gapCenters=[];
for(const f of fracs){const sx=knuckleX+f*(tipX-knuckleX);const r=detectZGapCenter(positions,fingerVerts,sx,0.004,-0.045,-0.015);if(r)gapCenters.push(r);}
const filteredGaps=gapCenters.filter(g=>Math.abs(g.center-fbxMids[2])<0.006);
let zGapMidIdx = valleyBounds[2];
if(filteredGaps.length>=3){
    const sg=filteredGaps.map(g=>g.center).sort((a,b)=>a-b);
    zGapMidIdx=sg[Math.floor(sg.length/2)];
    console.log(`Z-gap middle/index: ${(zGapMidIdx*1000).toFixed(1)}mm`);
}

// Z-zone init + constrained K-means (v5 logic)
const assignment=new Map(), fingerVertSet=new Set(fingerVerts);
for(const ci of fingerVerts){const z=positions[ci*3+2];let fi=0;for(let bi=0;bi<valleyBounds.length;bi++){if(z>valleyBounds[bi])fi=bi+1;}assignment.set(ci,fingerNames[Math.min(fi,3)]);}
for(const ci of thumbVerts) assignment.set(ci,'thumb');

const fingerZMin={},fingerZMax={};
for(let fi=0;fi<fingerNames.length;fi++){
    fingerZMin[fingerNames[fi]]=fi>0?valleyBounds[fi-1]:-Infinity;
    fingerZMax[fingerNames[fi]]=fi<valleyBounds.length?valleyBounds[fi]:Infinity;
}

const MARGIN_MM=9.5;
for(let iter=0;iter<8;iter++){
    const cen={},cnt={};for(const n of fingerNames){cen[n]=[0,0,0];cnt[n]=0;}
    for(const[ci,n]of assignment){if(n==='thumb')continue;cen[n][0]+=positions[ci*3];cen[n][1]+=positions[ci*3+1];cen[n][2]+=positions[ci*3+2];cnt[n]++;}
    const seeds=[];for(const n of fingerNames){const c=cnt[n];if(!c)continue;seeds.push({name:n,verts:[findClosestVert(positions,fingerVertSet,[cen[n][0]/c,cen[n][1]/c,cen[n][2]/c])]});}
    const ga=geodesicVoronoi(seeds,adj,fingerVertSet);
    let ch=0,rejected=0;
    for(const[ci,nn]of ga){
        const o=assignment.get(ci);if(o==='thumb'||o===nn)continue;
        const z=positions[ci*3+2];
        let distOutside=0;
        if(z<fingerZMin[nn])distOutside=(fingerZMin[nn]-z)*1000;
        if(z>fingerZMax[nn])distOutside=(z-fingerZMax[nn])*1000;
        if(distOutside>MARGIN_MM){rejected++;continue;}
        assignment.set(ci,nn);ch++;
    }
    console.log(`  K-Means iter ${iter}: changed ${ch}, rejected ${rejected}`);if(ch===0)break;
}

// Z-gap post-correction for middle/index
let midIdxFixed=0;
for(const ci of fingerVerts){
    const cur=assignment.get(ci);if(cur!=='middle'&&cur!=='index')continue;
    const z=positions[ci*3+2];
    const shouldBe=z<=zGapMidIdx?'middle':'index';
    if(cur!==shouldBe){assignment.set(ci,shouldBe);midIdxFixed++;}
}
console.log(`  Middle/index Z-gap correction: ${midIdxFixed}`);
reportCounts('V6-BASELINE',assignment);

// ===== PHASE 2: Per-slice tube-edge boundaries (iterated) =====
let sliceBounds; // persist across rounds for Phase 3
for (let round = 0; round < 5; round++) {
console.log(`\n--- Phase 2 round ${round}: Per-slice tube-edge refinement ---`);

// Build per-slice boundaries from v6 labels
// For each X-slice, find tube edges of each finger and compute midpoint boundaries
const SLICE_W = 0.004; // 4mm half-width per slice
sliceBounds = new Map(); // key=xBin, value=[bound0, bound1, bound2]

const prevMono = [null, null, null];
for (let sx = knuckleX; sx >= tipX; sx -= 0.002) {
    const xBin = Math.round(sx * 10000);
    // Collect finger verts in this slice
    const perFinger = {};
    for (const n of fingerNames) perFinger[n] = [];

    for (const ci of fingerVerts) {
        if (Math.abs(positions[ci*3] - sx) > SLICE_W) continue;
        const label = assignment.get(ci);
        if (label && perFinger[label]) perFinger[label].push(positions[ci*3+2]);
    }

    // Compute boundaries using 90th/10th percentile (robust to outliers)
    const bounds = [];
    for (let bi = 0; bi < fingerNames.length - 1; bi++) {
        const lower = perFinger[fingerNames[bi]].sort((a,b)=>a-b); // lower Z finger
        const upper = perFinger[fingerNames[bi+1]].sort((a,b)=>a-b); // higher Z finger
        if (lower.length >= 3 && upper.length >= 3) {
            // 92/8 for pinky/ring (most contamination-prone), 90/10 for others
            const pctL = bi === 0 ? 0.92 : 0.90;
            const pctU = bi === 0 ? 0.08 : 0.10;
            const lowerEdge = lower[Math.floor(lower.length * pctL)];
            const upperEdge = upper[Math.floor(upper.length * pctU)];
            bounds.push((lowerEdge + upperEdge) / 2);
        } else {
            bounds.push(null);
        }
    }

    // Enforce monotonicity for pinky/ring (bi=0) and ring/middle (bi=1) at deep tips.
    // As fingers spread toward tips, inter-finger boundaries move toward -Z.
    // Prevent contaminated tip boundaries from jumping back up.
    const TIP_MONO_X = knuckleX + 0.65 * (tipX - knuckleX);
    for (const bi of [0, 1, 2]) {
        if (bounds[bi] !== null) {
            if (sx < TIP_MONO_X && prevMono[bi] !== null && bounds[bi] > prevMono[bi]) {
                bounds[bi] = prevMono[bi];
            }
            prevMono[bi] = bounds[bi];
        }
    }

    sliceBounds.set(xBin, bounds);
}

// Show per-slice boundaries at a few X positions
for (const x of [tipX, tipX*0.7+knuckleX*0.3, tipX*0.3+knuckleX*0.7, knuckleX]) {
    const xBin = Math.round(x * 10000);
    let b = sliceBounds.get(xBin);
    if (!b) { // find nearest
        let best = Infinity;
        for (const [k, v] of sliceBounds) { const d = Math.abs(k-xBin); if (d < best) { best = d; b = v; } }
    }
    if (b) console.log(`  X=${(x*1000).toFixed(0)}mm: [${b.map(v=>v?(v*1000).toFixed(1):'?').join(', ')}]mm`);
}

// Reassign all finger verts using per-slice boundaries
let perSliceChanged = 0;
for (const ci of fingerVerts) {
    const x = positions[ci*3];
    const z = positions[ci*3+2];
    const cur = assignment.get(ci);
    if (cur === 'thumb') continue;

    // Find nearest slice bounds
    const xBin = Math.round(x * 10000);
    let b = sliceBounds.get(xBin);
    if (!b) {
        let best = Infinity;
        for (const [k, v] of sliceBounds) { const d = Math.abs(k-xBin); if (d < best) { best = d; b = v; } }
    }
    if (!b) continue;

    // Determine finger from per-slice boundaries
    let fi = 0;
    for (let bi = 0; bi < b.length; bi++) {
        if (b[bi] !== null && z > b[bi]) fi = bi + 1;
    }
    const shouldBe = fingerNames[Math.min(fi, 3)];
    if (shouldBe !== cur) {
        assignment.set(ci, shouldBe);
        perSliceChanged++;
    }
}
console.log(`  Per-slice reassigned: ${perSliceChanged}`);
reportCounts(`ROUND-${round}`, assignment);

if (perSliceChanged === 0) break;
} // end iteration loop

reportCounts('V7-FINAL', assignment);

// ===== Bone-extent correction =====
// Each finger has a maximum X-extent (bone tip + margin). Verts beyond a
// finger's reach get reassigned to the nearest reachable finger bone.
console.log('\n--- Bone-extent correction ---');
{
    // Compute per-finger mesh extent from current labels (5th percentile X + 5mm margin)
    const fingerExtents = {};
    for (const n of fingerNames) {
        const xVals = [];
        for (const ci of fingerVerts) {
            if (assignment.get(ci) === n) xVals.push(positions[ci*3]);
        }
        if (xVals.length < 10) continue;
        xVals.sort((a,b) => a - b); // most negative first
        fingerExtents[n] = xVals[Math.floor(xVals.length * 0.05)] - 0.005; // 5th pct - 5mm margin
    }
    console.log(`  Extents: ${Object.entries(fingerExtents).map(([n,x])=>`${n}:${(x*1000).toFixed(0)}mm`).join(', ')}`);

    let extentFixed = 0;
    for (const ci of fingerVerts) {
        const x = positions[ci*3];
        const cur = assignment.get(ci);
        if (cur === 'thumb' || !fingerExtents[cur]) continue;

        // Check if vert is beyond this finger's reach
        if (x >= fingerExtents[cur]) continue; // within reach (x more positive = closer to palm)

        // Beyond reach — reassign to nearest reachable finger bone
        const px = positions[ci*3], py = positions[ci*3+1], pz = positions[ci*3+2];
        let bestName = cur, bestDist = Infinity;
        for (let fi = 0; fi < sortedFB.length; fi++) {
            if (px < fingerExtents[sortedFB[fi].name]) continue; // this finger can't reach either
            const d = distToPolyline(px, py, pz, sortedFB[fi].points);
            if (d < bestDist) { bestDist = d; bestName = sortedFB[fi].name; }
        }
        if (bestName !== cur) {
            assignment.set(ci, bestName);
            extentFixed++;
        }
    }
    console.log(`  Reassigned ${extentFixed} verts beyond finger reach`);
}

// Neighbor smoothing
console.log('\n--- Neighbor smoothing ---');
for(let pass=0;pass<5;pass++){
    let flipped=0;
    for(const ci of fingerVerts){
        const cur=assignment.get(ci);if(cur==='thumb')continue;
        const nb=adj.get(ci);if(!nb)continue;
        const votes={};
        for(const[ni] of nb){const nl=assignment.get(ni);if(nl&&nl!=='thumb'&&nl!=='palm')votes[nl]=(votes[nl]||0)+1;}
        let total=0,maxN='',maxC=0;
        for(const[n,c] of Object.entries(votes)){total+=c;if(c>maxC){maxC=c;maxN=n;}}
        if(maxN!==cur&&maxC>=total*0.75&&total>=3){assignment.set(ci,maxN);flipped++;}
    }
    console.log(`  Pass ${pass}: flipped ${flipped}`);
if(flipped===0)break;
}

// Outlier removal: pinky verts far from cluster get reassigned to ring
// Prevents stray high-Z pinky verts from pulling skeleton centroid toward ring/middle
console.log('\n--- Outlier removal ---');
{
    // Compute per-finger Z statistics
    const fingerZStats = {};
    for (const n of fingerNames) {
        const zs = [];
        for (const ci of fingerVerts) {
            if (assignment.get(ci) === n) zs.push(positions[ci*3+2]);
        }
        if (zs.length < 5) continue;
        zs.sort((a,b) => a-b);
        const mean = zs.reduce((a,b) => a+b, 0) / zs.length;
        const std = Math.sqrt(zs.reduce((a,b) => a + (b - mean)**2, 0) / zs.length);
        fingerZStats[n] = { mean, std, p95: zs[Math.floor(zs.length * 0.95)], p05: zs[Math.floor(zs.length * 0.05)] };
    }

    let removed = 0;
    for (const ci of fingerVerts) {
        const cur = assignment.get(ci);
        if (cur === 'thumb' || cur === 'palm') continue;
        const z = positions[ci*3+2];
        const stats = fingerZStats[cur];
        if (!stats) continue;

        // Check if this vert is a Z-outlier (beyond 2.5 std from mean)
        if (Math.abs(z - stats.mean) < 2.5 * stats.std) continue;

        // It's an outlier — reassign to nearest finger bone
        const px = positions[ci*3], py = positions[ci*3+1], pz = positions[ci*3+2];
        let bestName = cur, bestDist = Infinity;
        for (const fb of sortedFB) {
            if (fb.name === cur) continue;
            const d = distToPolyline(px, py, pz, fb.points);
            if (d < bestDist) { bestDist = d; bestName = fb.name; }
        }
        if (bestName !== cur) {
            assignment.set(ci, bestName);
            removed++;
        }
    }
    console.log(`  Reassigned ${removed} outlier verts`);
}

// POST-LABEL: palm verts
let palmCount=0;
for(const ci of fingerVerts){if(positions[ci*3]>knuckleX){assignment.set(ci,'palm');palmCount++;}}
console.log(`Palm labeled: ${palmCount} verts`);
reportCounts('FINAL',assignment);

// ===== Overlap detection + blend weights =====
// For each vert, compute how close it is to the boundary with the adjacent finger.
// Verts deep in their finger's core get weight=1.0 (hard assignment).
// Verts in the overlap zone get blended weights (e.g. 0.7 pinky / 0.3 ring).
// This prevents stretching when fingers spread — overlap verts stay anchored to both bones.
console.log('\n--- Overlap blend weights ---');
const blendWeights = new Map(); // ci → { finger: weight, neighbor: weight } or null (core vert)
{
    const fingerIdx = {}; for (let i = 0; i < fingerNames.length; i++) fingerIdx[fingerNames[i]] = i;

    // Per adjacent pair: compute overlap zone boundaries
    const overlapZones = []; // [{fLow, fHigh, zMin, zMax}]
    for (let bi = 0; bi < fingerNames.length - 1; bi++) {
        const fLow = fingerNames[bi], fHigh = fingerNames[bi+1];
        let lowMax = -Infinity, highMin = Infinity;
        for (const ci of fingerVerts) {
            const label = assignment.get(ci);
            const z = positions[ci*3+2];
            if (label === fLow && z > lowMax) lowMax = z;
            if (label === fHigh && z < highMin) highMin = z;
        }
        if (lowMax > highMin) {
            overlapZones.push({ fLow, fHigh, zMin: highMin, zMax: lowMax, biLow: bi, biHigh: bi+1 });
            console.log(`  ${fLow}/${fHigh}: overlap Z=[${(highMin*1000).toFixed(1)}, ${(lowMax*1000).toFixed(1)}]mm`);
        }
    }

    // For each finger vert, check if it's in any overlap zone
    let overlapCount = 0, coreCount = 0;
    for (const ci of fingerVerts) {
        const label = assignment.get(ci);
        if (label === 'thumb' || label === 'palm') continue;
        const z = positions[ci*3+2];
        const fi = fingerIdx[label];

        let inOverlap = false;
        for (const oz of overlapZones) {
            if (z < oz.zMin || z > oz.zMax) continue;
            if (label !== oz.fLow && label !== oz.fHigh) continue;

            // Compute blend: how far into the overlap zone is this vert?
            // At the edge of overlap nearest to own finger's core → 1.0
            // At the edge nearest to neighbor → 0.5 (equal blend)
            const ozWidth = oz.zMax - oz.zMin;
            let t; // 0 = deep in own territory, 1 = at neighbor's edge
            if (label === oz.fLow) {
                // fLow finger, overlap is at its +Z edge
                t = (z - oz.zMin) / ozWidth; // 0 at low edge (near fHigh core), 1 at high edge (own core)
                t = 1 - t; // flip: 0 = near own core, 1 = near neighbor
            } else {
                // fHigh finger, overlap is at its -Z edge
                t = (z - oz.zMin) / ozWidth; // 0 at low edge (near fLow core), 1 at high edge (own core)
            }
            t = Math.max(0, Math.min(1, t));

            // Blend: own weight goes from 1.0 (at own core edge) to 0.5 (at neighbor edge)
            const ownWeight = 1.0 - 0.5 * t;
            const neighborName = label === oz.fLow ? oz.fHigh : oz.fLow;

            blendWeights.set(ci, { [label]: ownWeight, [neighborName]: 1.0 - ownWeight });
            inOverlap = true;
            overlapCount++;
            break;
        }
        if (!inOverlap) {
            blendWeights.set(ci, { [label]: 1.0 });
            coreCount++;
        }
    }
    console.log(`  Core verts: ${coreCount}, Overlap verts: ${overlapCount}`);
}

// Output — now includes blend weights for overlap zone
const output={left:[],right:[]};
for(let vi=0;vi<vertCount;vi++){
    const ci=canonMap[vi];
    if(assignment.has(ci)){
        const entry = {vi, x:positions[vi*3], y:positions[vi*3+1], z:positions[vi*3+2], finger:assignment.get(ci)};
        const bw = blendWeights.get(ci);
        if (bw && Object.keys(bw).length > 1) {
            entry.blend = bw; // e.g. {pinky: 0.7, ring: 0.3}
        }
        output.left.push(entry);
    }
}
fs.writeFileSync(path.join(BASE_DIR,'finger_assignment.json'),JSON.stringify(output));
const blendedCount = output.left.filter(v => v.blend).length;
console.log(`\nWrote ${output.left.length} verts (${blendedCount} with blend weights)`);

// Compare with GT
let gt;try{gt=JSON.parse(fs.readFileSync(path.join(BASE_DIR,'finger_assignment (1).json'),'utf8'));}catch(e){}
if(gt){
    const gtMap=new Map();for(const v of gt.left)gtMap.set(v.vi,v.finger);
    let total=0,agree=0,palmN=0;const conf={};
    for(const e of output.left){const g=gtMap.get(e.vi);if(!g)continue;total++;if(e.finger==='palm'){palmN++;continue;}if(e.finger===g){agree++;continue;}conf[`${e.finger}→${g}`]=(conf[`${e.finger}→${g}`]||0)+1;}
    const scored=total-palmN;
    console.log(`\n[SCORE] ${agree}/${scored} = ${(100*agree/scored).toFixed(1)}% (${palmN} palm excluded)`);
    for(const[k,v]of Object.entries(conf).sort((a,b)=>b[1]-a[1]).slice(0,8))console.log(`  ${k}: ${v}`);
    let rawA=0;for(const e of output.left){const g=gtMap.get(e.vi);if(g&&e.finger===g)rawA++;}
    console.log(`[RAW] ${rawA}/${total} = ${(100*rawA/total).toFixed(1)}%`);
    let fA=0,fT=0;for(const e of output.left){if(e.finger==='thumb'||e.finger==='palm')continue;const g=gtMap.get(e.vi);if(!g||g==='thumb')continue;fT++;if(e.finger===g)fA++;}
    console.log(`[FINGER-ONLY] ${fA}/${fT} = ${(100*fA/fT).toFixed(1)}%`);
}
console.log('\nDone! Refresh http://localhost:8000/finger_editor.html');
