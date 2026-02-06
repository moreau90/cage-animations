import json, math
import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:
    import fast_simplification
except Exception as e:
    raise RuntimeError("fast_simplification import failed. Try: pip install fast-simplification") from e


# ----------------------------
# Tunables
# ----------------------------
CAGE_TARGET_FACES = 800          # you said ~800, good
INFLATE = 0.015                  # your offset along normals
K = 16                           # IMPORTANT: 16 (or 24) to stop spiky binding
SIGMA_FRAC = 0.12                # gaussian width as fraction of mesh height (0.08–0.18 typical)


def simplify_to_target(v, f, target_faces=800, max_iter=10):
    v = v.astype(np.float64)
    f = f.astype(np.int32)
    for _ in range(max_iter):
        if len(f) <= target_faces:
            break
        # reduce progressively
        target_reduction = min(0.85, 1.0 - (target_faces / max(len(f), 1)))
        v, f = fast_simplification.simplify(v, f, target_reduction=target_reduction)
    return v, f


def gaussian_weights(dists, sigma):
    # dists shape (K,)
    w = np.exp(-(dists / max(1e-9, sigma)) ** 2)
    s = w.sum()
    if s < 1e-12:
        w[:] = 1.0 / len(w)
    else:
        w /= s
    return w


def classify_regions_cage(cage_v):
    """
    Robust-ish region assignment using normalized height + torso width measurement.
    Returns regionId per cage vertex and metadata.
    region ids:
      0 torso/core
      1 l_thigh
      2 r_thigh
      3 l_shin
      4 r_shin
      5 l_foot
      6 r_foot
      7 l_upperarm
      8 r_upperarm
      9 l_forearm
      10 r_forearm
      11 head/neck
    """
    v = cage_v
    miny = float(v[:,1].min())
    maxy = float(v[:,1].max())
    H = max(1e-9, maxy - miny)
    ny = (v[:,1] - miny) / H

    cx = float(np.median(v[:,0]))

    # Measure torso half-width at mid-torso (ny ~ 0.50–0.55)
    band = (ny > 0.48) & (ny < 0.58)
    if band.sum() < 20:
        band = (ny > 0.45) & (ny < 0.60)
    torso_half = float(np.percentile(np.abs(v[band,0] - cx), 85)) if band.sum() else float(np.percentile(np.abs(v[:,0]-cx), 60))
    arm_thresh = torso_half * 1.15  # arms are outside torso width

    region = np.zeros(len(v), dtype=np.int32)

    # Feet: lowest
    is_foot = ny < 0.06
    region[is_foot & (v[:,0] < cx)] = 5
    region[is_foot & (v[:,0] >= cx)] = 6

    # Legs: ny < 0.40 excluding feet
    is_leg = (ny >= 0.06) & (ny < 0.40)
    # Split thigh/shin by ny threshold (works better than distance thresholds for cages)
    is_thigh = is_leg & (ny >= 0.22)
    is_shin  = is_leg & (ny < 0.22)

    region[is_thigh & (v[:,0] < cx)] = 1
    region[is_thigh & (v[:,0] >= cx)] = 2
    region[is_shin & (v[:,0] < cx)] = 3
    region[is_shin & (v[:,0] >= cx)] = 4

    # Head/neck
    is_head = ny > 0.82
    region[is_head] = 11

    # Arms: mid-high band, outside torso half-width
    is_arm_band = (ny > 0.55) & (ny < 0.80) & (np.abs(v[:,0] - cx) > arm_thresh)
    # upperarm vs forearm by |x-cx| (forearm tends to be farther)
    arm_dist = np.abs(v[:,0] - cx)
    # split by percentile within arm band
    if is_arm_band.sum() > 10:
        split = np.percentile(arm_dist[is_arm_band], 55)
    else:
        split = arm_thresh * 1.4

    is_upperarm = is_arm_band & (arm_dist <= split)
    is_forearm  = is_arm_band & (arm_dist > split)

    region[is_upperarm & (v[:,0] < cx)] = 7
    region[is_upperarm & (v[:,0] >= cx)] = 8
    region[is_forearm & (v[:,0] < cx)] = 9
    region[is_forearm & (v[:,0] >= cx)] = 10

    meta = {
        "miny": miny,
        "maxy": maxy,
        "height": H,
        "cx": cx,
        "torso_half": torso_half,
        "arm_thresh": arm_thresh,
    }
    return region, meta


def main():
    mesh = trimesh.load("mesh.glb", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        # pick largest geometry
        geoms = list(mesh.geometry.values())
        mesh = max(geoms, key=lambda g: len(g.vertices))

    V = mesh.vertices.view(np.ndarray).astype(np.float64)
    F = mesh.faces.view(np.ndarray).astype(np.int32)

    # Mesh bounds (for sigma)
    miny = float(V[:,1].min())
    maxy = float(V[:,1].max())
    H = max(1e-9, maxy - miny)

    # Build cage via simplification + inflate
    v_s, f_s = simplify_to_target(V, F, target_faces=CAGE_TARGET_FACES, max_iter=12)
    cage = trimesh.Trimesh(vertices=v_s, faces=f_s, process=True)

    # Inflate outward
    cage.vertices = cage.vertices + cage.vertex_normals * INFLATE
    cage_v = cage.vertices.view(np.ndarray).astype(np.float64)
    cage_f = cage.faces.view(np.ndarray).astype(np.int32)

    # Region classify cage (for hierarchical driving)
    regionId, regionMeta = classify_regions_cage(cage_v)

    # Build binding (IMPORTANT: NOT region-constrained)
    tree = cKDTree(cage_v)
    dists, idx = tree.query(V, k=K, workers=-1)  # shape (nMesh, K)
    sigma = SIGMA_FRAC * H

    w = np.zeros_like(dists, dtype=np.float32)
    for i in range(len(V)):
        w[i] = gaussian_weights(dists[i], sigma).astype(np.float32)

    # Save files
    with open("cage.json", "w") as f:
        json.dump({"verts": cage_v.tolist(), "faces": cage_f.tolist()}, f)

    with open("bind.json", "w") as f:
        json.dump({"k": K, "idx": idx.astype(int).ravel().tolist(), "w": w.ravel().tolist(),
                   "sigma": sigma, "sigma_frac": SIGMA_FRAC}, f)

    with open("regions.json", "w") as f:
        json.dump({"regionId": regionId.astype(int).tolist(), "meta": regionMeta}, f)

    print("Wrote cage.json, bind.json, regions.json")
    print(f"mesh verts: {len(V)} | cage verts: {len(cage_v)} | cage faces: {len(cage_f)} | K: {K} | sigma: {sigma:.4f}")

if __name__ == "__main__":
    main()
