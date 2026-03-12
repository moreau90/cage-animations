# Cage Walk — 3D Humanoid Skeleton Placement Project

## What This Is
A system for placing skeleton joints inside a 3D humanoid mesh. The mesh is a Mixamo-compatible humanoid.
The skeleton drives deformation (LBS skinning). Joints must be anatomically correct INSIDE the mesh.

## Key Data Files (in this directory)
- `mesh.glb` — the humanoid mesh (GLB/glTF format with vertex positions and face indices)
- `placed_joints.json` — current joint positions: `{ "P": { "joint_name": [x, y, z], ... }, "primary": {...}, "bone_lengths": {...} }`
- `cage.json` — cage geometry: vertex positions (`V`), face indices (`F`), per-vertex data
- `regions.json` — body region assignments per vertex (0=torso, 1/2=thighs, 3/4=shins, 5/6=feet, 7/8=upper arms, 9/10=forearms, 11=head, 12/13=hands)
- `bind.json` — binding data between mesh and skeleton
- `index.html` — 18k line Three.js app that loads/renders everything

## Coordinate System
- Y is UP (height)
- X is LEFT-RIGHT (mesh faces -Z)
- Z is FRONT-BACK
- Units: METERS (joint positions in placed_joints.json are in meters)
- The mesh is roughly 1.7m tall

## Joint Placement Rules
- Joints go INSIDE the mesh, not on the surface
- **Hips** (l_hip, r_hip): femoral head position — ball-and-socket, laterally offset from body center, NOT at torso centroid
- **Shoulders** (l_shoulder, r_shoulder): glenohumeral joint — where arm connects to torso, sits POSTERIOR to torso cross-section centroid
- **Knees/Elbows**: hinge joints at the bend axis of the limb
- **Spine joints**: along the spinal column, roughly centered in torso
- **Wrists/Ankles**: centered in the limb cross-section at the joint

## CRITICAL: Centering Metrics Are NOT Correction Vectors
The QA pipeline measures `centering_{joint}_total_mm` — distance from joint to mesh cross-section centroid.
This is a MEASUREMENT, not a correction recipe. Cross-section centroids are geometric centers of the mesh slice.
For anatomically asymmetric areas (shoulders, hips, torso), the joint should NOT be at the centroid.
You MUST analyze the actual mesh geometry to determine correct placement.

## cage-core Library (cage-core/src/)
TypeScript geometry library with mesh analysis:
- `geometry/plane-slice.ts` — `sliceMeshAtPlane()` slices mesh with a plane, returns cross-section contour and centroid
- `geometry/cross-section.ts` — cross-section shape analysis
- `geometry/mesh-surface.ts` — mesh surface utilities
- `qa/skeleton-qa.ts` — skeleton quality checks
- `skeleton/` — bone hierarchy, FK computation

## cage-pipeline (cage-pipeline/src/)
The optimization pipeline:
- `workers/joint-centering.worker.ts` — computes centering scores (how far each joint is from cross-section centroid)
- `workers/finger-alignment.worker.ts` — measures finger skeleton alignment within finger mesh tubes
- `lib/mutation-engine.ts` — generates joint position candidates
- `lib/ai-workers.ts` — spawns this AI agent

## How To Analyze The Mesh
1. Read `cage.json` — it has vertex positions in `V` array and face indices in `F` array
2. Read `regions.json` — tells you which vertices belong to which body part
3. Use the vertex data to understand mesh geometry at any joint location
4. For cross-sections: slice the mesh at the joint's Y height, look at the resulting contour shape
5. For arm/leg joints: the joint should be at the axis of rotation, which you can determine from the limb geometry above and below the joint
