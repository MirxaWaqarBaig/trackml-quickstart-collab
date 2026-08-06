"""
Coplanarity-based quirk detection utility.

Two implementations are provided:

1. SIMPLE (our original) — centred covariance + global SVD.
   Fast, good for clean events, gives a dimensionless score in [0, 1/3].

2. PAPER (Knapen, Lou, Papucci, Setford 2017, arXiv:1708.02243) — matches
   the exact algorithm proposed for ATLAS/CMS:
   - Uncentred tensor T_ij = (1/(N-1)) * sum x_a^i * x_a^j  (from IP)
   - Two separate variables: delta_s (thickness) and delta_w (width) in mm
   - Two-stage algorithm: seed from outer layers -> iterative inside-out fit
   - Final cuts: delta_s < 0.1 mm, delta_w < 10.0 mm, >=2 hits per layer
"""

import numpy as np

# ---------------------------------------------------------------------------
# TrackML barrel layer definitions (radius boundaries in mm)
# Layer index 1 = innermost, 6 = outermost (as seen in our feature store)
# ---------------------------------------------------------------------------
LAYER_BINS_MM = [
    (0.0,   52.0,  1),   # ~32 mm  pixel layer 1
    (52.0,  95.0,  2),   # ~72 mm  pixel layer 2
    (95.0,  145.0, 3),   # ~116 mm pixel layer 3
    (145.0, 215.0, 4),   # ~172 mm pixel layer 4
    (215.0, 310.0, 5),   # ~260 mm strip layer 5
    (310.0, 450.0, 6),   # ~360 mm strip layer 6
    (450.0, 590.0, 7),   # ~500 mm strip layer 7
    (590.0, 9999., 8),   # ~660 mm strip layer 8
]

N_LAYERS = 8   # paper uses 8 layers

# Paper cut values (in mm; paper quotes cm)
PAPER_DS_SEED_MM   = 0.5    # 0.05 cm — seeding stage thickness cut
PAPER_DW_SEED_MM   = 10.0   # 1.0  cm — seeding stage width cut
PAPER_DS_ITER_MM   = 0.5    # 0.05 cm — iterative stage plane tolerance
PAPER_DW_ITER_MM   = 10.0   # 1.0  cm — iterative stage width tolerance
PAPER_DS_FINAL_MM  = 0.1    # 0.01 cm — paper final selection thickness cut
PAPER_DW_FINAL_MM  = 10.0   # 1.0  cm — final selection width cut
PAPER_GROW_FACTOR  = 3.0    # max allowed growth of delta_s / delta_w when adding a hit
PAPER_DPHI_SEED    = 0.1    # rad — max Δφ between seeding pair hits
PAPER_DZ_SEED_MM   = 20.0   # mm  — max Δz between seeding pair hits (paper: 2 cm)


# ===========================================================================
# ── COORDINATE UTILITIES ────────────────────────────────────────────────────
# ===========================================================================

def reconstruct_xyz(x_tensor):
    """
    Convert feature-store tensor [r/1000, phi/pi, z/1000] to Cartesian mm.

    Parameters
    ----------
    x_tensor : array-like, shape (N, 3)

    Returns
    -------
    xyz : np.ndarray, shape (N, 3)  [mm]
    """
    x    = np.asarray(x_tensor, dtype=np.float64)
    r_mm = x[:, 0] * 1000.0
    phi  = x[:, 1] * np.pi
    z_mm = x[:, 2] * 1000.0
    return np.stack([r_mm * np.cos(phi),
                     r_mm * np.sin(phi),
                     z_mm], axis=1)


def assign_layers(hits_xyz):
    """
    Assign each hit to a barrel layer (1=innermost … 8=outermost) based on
    its transverse radius r = sqrt(x^2 + y^2).

    Hits outside all defined bins are assigned layer 0.

    Parameters
    ----------
    hits_xyz : array-like, shape (N, 3)  [mm]

    Returns
    -------
    layer_ids : np.ndarray of int, shape (N,)
    """
    hits  = np.asarray(hits_xyz, dtype=np.float64)
    r_mm  = np.sqrt(hits[:, 0] ** 2 + hits[:, 1] ** 2)
    layer_ids = np.zeros(len(hits), dtype=np.int32)
    for r_lo, r_hi, lyr in LAYER_BINS_MM:
        mask = (r_mm >= r_lo) & (r_mm < r_hi)
        layer_ids[mask] = lyr
    return layer_ids


def build_geometry_layer_map(detector_df, max_radial_spread_mm=10.0,
                             min_z_span_mm=500.0):
    """Map real TrackML barrel ``(volume_id, layer_id)`` pairs by radius.

    The barrel/endcap distinction is derived from the geometry: a barrel has
    nearly constant radius and a broad z span.  No volume IDs or radii are
    hard-coded.  Values in the returned mapping run from 1 (innermost) to N
    (outermost).
    """
    required = {"volume_id", "layer_id", "cx", "cy", "cz"}
    missing = required.difference(detector_df.columns)
    if missing:
        raise ValueError(f"Detector table is missing columns: {sorted(missing)}")
    layers = []
    for key, group in detector_df.groupby(["volume_id", "layer_id"], sort=False):
        radius = np.hypot(group["cx"].to_numpy(dtype=np.float64),
                          group["cy"].to_numpy(dtype=np.float64))
        z = group["cz"].to_numpy(dtype=np.float64)
        if ((radius.max() - radius.min()) <= float(max_radial_spread_mm)
                and (z.max() - z.min()) >= float(min_z_span_mm)):
            layers.append((float(np.median(radius)), (int(key[0]), int(key[1]))))
    layers.sort()
    if len(layers) < 2:
        raise ValueError("Could not identify at least two barrel layers")
    return {key: rank for rank, (_, key) in enumerate(layers, start=1)}


def assign_layers_from_geometry(module_indices, detector_df, layer_map=None):
    """Assign feature-store hits using their exact detector module indices."""
    if layer_map is None:
        layer_map = build_geometry_layer_map(detector_df)
    modules = np.asarray(module_indices, dtype=np.int64)
    assigned = np.zeros(len(modules), dtype=np.int32)
    if np.any(modules < 0) or np.any(modules >= len(detector_df)):
        raise IndexError("Feature-store module index is outside detectors.csv")
    selected = detector_df.iloc[modules][["volume_id", "layer_id"]]
    for i, row in enumerate(selected.itertuples(index=False)):
        assigned[i] = int(layer_map.get((int(row.volume_id), int(row.layer_id)), 0))
    return assigned


# ===========================================================================
# ── SIMPLE METHOD (centred covariance + global SVD) ─────────────────────────
# ===========================================================================

def compute_coplanarity_score(hits_xyz):
    """
    Simple coplanarity score via centred covariance SVD.

    Returns
    -------
    score     : float — lambda3/(lambda1+lambda2+lambda3), near 0 = coplanar
    normal    : np.ndarray (3,) — unit normal of best-fit plane
    centroid  : np.ndarray (3,)
    residuals : np.ndarray (N,) — signed distance of each hit from plane [mm]
    """
    hits = np.asarray(hits_xyz, dtype=np.float64)
    if len(hits) < 3:
        return 1.0, np.array([0., 0., 1.]), np.zeros(3), np.zeros(len(hits))

    centroid = hits.mean(axis=0)
    centred  = hits - centroid
    _, S, Vt = np.linalg.svd(centred, full_matrices=False)
    eigvals  = (S ** 2) / len(hits)
    total    = eigvals.sum()
    score    = float(eigvals[2] / total) if total > 1e-12 else 1.0
    normal   = Vt[2]
    residuals = centred @ normal
    return score, normal, centroid, residuals


def ransac_plane_fit(hits_xyz, n_iter=2000, epsilon_mm=15.0,
                     min_inliers=6, rng_seed=42):
    """
    RANSAC plane finder for mixed-background events.

    Returns
    -------
    best_inliers : np.ndarray of int
    best_normal  : np.ndarray (3,)
    best_score   : float (coplanarity score of inlier set)
    n_inliers    : int
    """
    hits = np.asarray(hits_xyz, dtype=np.float64)
    N    = len(hits)
    if N < 3:
        return np.arange(N), np.array([0., 0., 1.]), 1.0, N

    rng          = np.random.default_rng(rng_seed)
    best_inliers = np.array([], dtype=np.int64)
    best_normal  = np.array([0., 0., 1.])
    best_score   = 1.0

    for _ in range(n_iter):
        idx      = rng.choice(N, 3, replace=False)
        p0, p1, p2 = hits[idx[0]], hits[idx[1]], hits[idx[2]]
        v1, v2   = p1 - p0, p2 - p0
        n        = np.cross(v1, v2)
        nn       = np.linalg.norm(n)
        if nn < 1e-10:
            continue
        n /= nn
        dist    = np.abs((hits - p0) @ n)
        inliers = np.where(dist <= epsilon_mm)[0]
        if len(inliers) > len(best_inliers) and len(inliers) >= min_inliers:
            best_inliers = inliers
            best_normal  = n

    if len(best_inliers) >= 3:
        score, refined_normal, _, _ = compute_coplanarity_score(hits[best_inliers])
        best_score  = score
        best_normal = refined_normal

    return best_inliers, best_normal, best_score, len(best_inliers)


def event_coplanarity_features(hits_xyz, source_labels=None,
                                run_ransac=False, ransac_kwargs=None):
    """
    Extract simple coplanarity features for one event.

    Returns dict with: n_hits, coplanarity_all, rms_residual_mm,
    and (if run_ransac) coplanarity_ransac, n_ransac_inliers,
    ransac_inlier_frac, quirk_recall, sm_contamination.
    """
    hits = np.asarray(hits_xyz, dtype=np.float64)
    N    = len(hits)
    score_all, _, _, residuals = compute_coplanarity_score(hits)
    features = {
        "n_hits":          N,
        "coplanarity_all": score_all,
        "rms_residual_mm": float(np.sqrt(np.mean(residuals ** 2))),
    }
    if run_ransac:
        kw = ransac_kwargs or {}
        inliers, _, score_r, n_in = ransac_plane_fit(hits, **kw)
        features["coplanarity_ransac"] = score_r
        features["n_ransac_inliers"]   = n_in
        features["ransac_inlier_frac"] = n_in / max(N, 1)
        if source_labels is not None:
            labels   = np.asarray(source_labels)
            in_lbl   = labels[inliers]
            qmask    = (labels == 1) | (labels == 2)
            n_q      = qmask.sum()
            n_q_in   = ((in_lbl == 1) | (in_lbl == 2)).sum()
            n_sm_in  = (in_lbl == 0).sum()
            features["quirk_recall"]      = float(n_q_in / max(n_q, 1))
            features["sm_contamination"]  = float(n_sm_in / max(n_in, 1))
    return features


# ===========================================================================
# ── PAPER METHOD (Knapen et al. 2017) ───────────────────────────────────────
# ===========================================================================

def compute_T_tensor(hits_xyz):
    """
    Compute the uncentred tensor T_ij = (1/(N-1)) * sum_a x_a^i * x_a^j
    and return delta_s, delta_w and all three eigenvectors.

    This is Eq. 11 of arXiv:1708.02243. Positions are measured from the
    primary vertex (assumed at origin).

    Parameters
    ----------
    hits_xyz : array-like, shape (N, 3)  [mm]

    Returns
    -------
    delta_s : float  — sqrt(smallest eigenvalue) = plane thickness [mm]
    delta_w : float  — sqrt(second eigenvalue)   = oscillation width [mm]
    n1      : np.ndarray (3,) — normal to quirk plane
    n2      : np.ndarray (3,) — within-plane direction splitting Q / Qbar
    n3      : np.ndarray (3,) — COM motion direction
    success : bool   — False if N < 4 or numerically degenerate
    """
    hits = np.asarray(hits_xyz, dtype=np.float64)
    N    = len(hits)
    if N < 4:
        return 0.0, 0.0, np.array([0.,0.,1.]), np.array([0.,1.,0.]), \
               np.array([1.,0.,0.]), False

    # T = X^T X / (N-1)  where X is (N,3) matrix of hit positions
    T = (hits.T @ hits) / (N - 1)

    # np.linalg.eigh returns eigenvalues in ascending order for symmetric matrix
    eigvals, eigvecs = np.linalg.eigh(T)

    # Safety clip against numerical noise giving tiny negatives
    eigvals = np.maximum(eigvals, 0.0)

    delta_s = float(np.sqrt(eigvals[0]))   # smallest = thickness
    delta_w = float(np.sqrt(eigvals[1]))   # middle   = width
    n1 = eigvecs[:, 0]                      # normal to plane
    n2 = eigvecs[:, 1]                      # splits Q / Qbar
    n3 = eigvecs[:, 2]                      # COM direction (largest spread)

    return delta_s, delta_w, n1, n2, n3, True


def _ensure_forward(hits_xyz, n3):
    """
    Flip n3 so that the majority of hits satisfy n3·x > 0 (forward hemisphere).
    Returns the (possibly flipped) n3.
    """
    projections = hits_xyz @ n3
    if projections.mean() < 0:
        return -n3
    return n3


def _seed_pairs(hits_xyz, layer_mask, dphi_cut=PAPER_DPHI_SEED,
                dz_cut_mm=PAPER_DZ_SEED_MM, max_pairs=50):
    """
    Return pairs of hit indices within one layer satisfying
    Δφ < dphi_cut and Δz < dz_cut_mm, sorted by Δφ (closest first).
    At most max_pairs pairs are returned to keep seeding tractable.
    """
    idx  = np.where(layer_mask)[0]
    if len(idx) < 2:
        return []
    xy   = hits_xyz[idx, :2]
    z    = hits_xyz[idx, 2]
    phi  = np.arctan2(xy[:, 1], xy[:, 0])
    pairs = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dphi = abs(phi[i] - phi[j])
            if dphi > np.pi:
                dphi = 2.0 * np.pi - dphi
            if dphi < dphi_cut and abs(z[i] - z[j]) < dz_cut_mm:
                pairs.append((dphi, idx[i], idx[j]))
    pairs.sort()
    return [(a, b) for (_, a, b) in pairs[:max_pairs]]


def paper_plane_finding(hits_xyz, layer_ids=None,
                        ds_seed=PAPER_DS_SEED_MM,
                        dw_seed=PAPER_DW_SEED_MM,
                        ds_iter=PAPER_DS_ITER_MM,
                        dw_iter=PAPER_DW_ITER_MM,
                        ds_final=PAPER_DS_FINAL_MM,
                        dw_final=PAPER_DW_FINAL_MM,
                        grow_factor=PAPER_GROW_FACTOR,
                        min_hits_per_layer=2,
                        min_layers_with_2hits=None,
                        max_seed_pairs=50,
                        max_seeds=500,
                        dphi_seed=PAPER_DPHI_SEED,
                        dz_seed_mm=PAPER_DZ_SEED_MM):
    """
    Full two-stage plane-finding algorithm from Knapen et al. 2017.

    Stage 1 — Seeding:
        Form pairs of hits in the two outermost layers (Δφ < 0.1, Δz < 2 cm).
        Combine pairs from the two outermost layers into 4-hit seeds.
        Compute T tensor; apply delta_s, delta_w, and n3·x > 0 cuts.

    Stage 2 — Iterative inside-out fitting:
        For each seed, go layer by layer from second-outermost inward.
        Collect hits with |n1·x| < ds_iter and |n2·x| < dw_iter and n3·x > 0.
        Add a hit only if delta_s and delta_w do not grow by more than
        grow_factor compared to the current values.

    Final selection:
        delta_s < ds_final, delta_w < dw_final.
        At least min_hits_per_layer hits in enough layers.

    Parameters
    ----------
    hits_xyz   : array-like, shape (N, 3)  [mm]
    layer_ids  : array-like, shape (N,)   int  (1=innermost … 8=outermost)
                 If None, computed automatically from hit radius.

    Returns
    -------
    result : dict with keys:
        found          : bool
        delta_s        : float [mm]
        delta_w        : float [mm]
        n1, n2, n3     : np.ndarray (3,)
        plane_hit_idx  : np.ndarray of int  (indices into hits_xyz)
        n_plane_hits   : int
        layer_counts   : dict {layer_id: count}
        n_layers_2hits : int  (number of layers with >= 2 hits)
        seed_idx       : list of 4 ints (the winning seed)
    """
    hits = np.asarray(hits_xyz, dtype=np.float64)
    N    = len(hits)

    if layer_ids is None:
        layer_ids = assign_layers(hits)
    layer_ids = np.asarray(layer_ids, dtype=np.int32)

    present_layers = sorted(np.unique(layer_ids[layer_ids > 0]))
    n_total_layers = len(present_layers)

    if n_total_layers < 2 or N < 4:
        return _empty_result()

    if min_layers_with_2hits is None:
        min_layers_with_2hits = max(1, n_total_layers - 1)

    # ── Stage 1: seeding from the two outermost layers ──────────────────────
    outer_layer  = present_layers[-1]
    second_layer = present_layers[-2]

    outer_pairs  = _seed_pairs(hits, layer_ids == outer_layer,
                               dphi_cut=dphi_seed, dz_cut_mm=dz_seed_mm,
                               max_pairs=max_seed_pairs)
    second_pairs = _seed_pairs(hits, layer_ids == second_layer,
                               dphi_cut=dphi_seed, dz_cut_mm=dz_seed_mm,
                               max_pairs=max_seed_pairs)

    best_result  = None
    best_candidate = None
    best_ds      = np.inf
    n_seeds_tried = 0

    for (i1, i2) in outer_pairs:
        if n_seeds_tried >= max_seeds:
            break
        for (i3, i4) in second_pairs:
            if n_seeds_tried >= max_seeds:
                break
            n_seeds_tried += 1
            seed_idx  = [i1, i2, i3, i4]
            seed_xyz  = hits[seed_idx]
            ds, dw, n1, n2, n3, ok = compute_T_tensor(seed_xyz)
            if not ok:
                continue
            n3 = _ensure_forward(seed_xyz, n3)

            # Seed cuts
            if ds > ds_seed or dw > dw_seed:
                continue
            if not np.all((seed_xyz @ n3) > 0):
                continue

            # ── Stage 2: iterative inside-out fitting ────────────────────
            accepted   = list(seed_idx)
            cur_ds     = ds
            cur_dw     = dw
            cur_n1, cur_n2, cur_n3 = n1, n2, n3

            fit_layers = [l for l in present_layers
                          if l not in (outer_layer, second_layer)]
            fit_layers = sorted(fit_layers, reverse=True)   # outside-in

            for lyr in fit_layers:
                lyr_mask   = layer_ids == lyr
                lyr_idx    = np.where(lyr_mask)[0]
                candidates = []
                for hi in lyr_idx:
                    x = hits[hi]
                    if (x @ cur_n3) <= 0:
                        continue
                    if abs(x @ cur_n1) > ds_iter:
                        continue
                    if abs(x @ cur_n2) > dw_iter:
                        continue
                    candidates.append((abs(x @ cur_n1), hi))

                # Sort by distance to plane (closest first)
                candidates.sort()

                for _, hi in candidates:
                    trial = accepted + [hi]
                    t_ds, t_dw, t_n1, t_n2, t_n3, ok2 = compute_T_tensor(
                        hits[trial])
                    if not ok2:
                        continue
                    t_n3 = _ensure_forward(hits[trial], t_n3)
                    ds_limit = grow_factor * max(cur_ds, np.finfo(float).eps)
                    dw_limit = grow_factor * max(cur_dw, np.finfo(float).eps)
                    if t_ds > ds_limit or t_dw > dw_limit:
                        continue
                    accepted = trial
                    cur_ds, cur_dw = t_ds, t_dw
                    cur_n1, cur_n2, cur_n3 = t_n1, t_n2, t_n3

            plane_idx   = np.array(accepted, dtype=np.int64)
            lyr_counts  = {}
            for hi in plane_idx:
                l = int(layer_ids[hi])
                lyr_counts[l] = lyr_counts.get(l, 0) + 1
            n_layers_2h = sum(1 for c in lyr_counts.values()
                              if c >= min_hits_per_layer)

            candidate = dict(
                found=False, delta_s=cur_ds, delta_w=cur_dw,
                n1=cur_n1, n2=cur_n2, n3=cur_n3,
                plane_hit_idx=plane_idx, n_plane_hits=len(plane_idx),
                layer_counts=lyr_counts, n_layers_2hits=n_layers_2h,
                seed_idx=seed_idx, outer_layer=int(outer_layer),
                second_layer=int(second_layer), n_seeds_tried=n_seeds_tried,
            )
            if best_candidate is None or cur_ds < best_candidate["delta_s"]:
                best_candidate = candidate

            # ── Evaluate this candidate plane ────────────────────────────
            if cur_ds > ds_final or cur_dw > dw_final:
                continue

            if n_layers_2h < min_layers_with_2hits:
                continue

            if cur_ds < best_ds:
                best_ds = cur_ds
                best_result = dict(
                    found=True,
                    delta_s=cur_ds,
                    delta_w=cur_dw,
                    n1=cur_n1, n2=cur_n2, n3=cur_n3,
                    plane_hit_idx=plane_idx,
                    n_plane_hits=len(plane_idx),
                    layer_counts=lyr_counts,
                    n_layers_2hits=n_layers_2h,
                    seed_idx=seed_idx,
                    outer_layer=int(outer_layer),
                    second_layer=int(second_layer),
                    n_seeds_tried=n_seeds_tried,
                )

    if best_result is not None:
        return best_result
    if best_candidate is not None:
        return best_candidate
    return _empty_result(outer_layer=outer_layer, second_layer=second_layer,
                         n_seeds_tried=n_seeds_tried)


def _empty_result(outer_layer=0, second_layer=0, n_seeds_tried=0):
    return dict(
        found=False, delta_s=np.nan, delta_w=np.nan,
        n1=None, n2=None, n3=None,
        plane_hit_idx=np.array([], dtype=np.int64),
        n_plane_hits=0, layer_counts={},
        n_layers_2hits=0, seed_idx=[],
        outer_layer=int(outer_layer), second_layer=int(second_layer),
        n_seeds_tried=int(n_seeds_tried),
    )


def paper_classify_event(hits_xyz, layer_ids=None, **kwargs):
    """
    Run the paper algorithm on one event and return a flat feature dict.

    Returns
    -------
    dict with: found, delta_s, delta_w, n_plane_hits, n_layers_2hits
    """
    result = paper_plane_finding(hits_xyz, layer_ids=layer_ids, **kwargs)
    return {
        "found":         result["found"],
        "delta_s":       result["delta_s"],
        "delta_w":       result["delta_w"],
        "n_plane_hits":  result["n_plane_hits"],
        "n_layers_2hits": result["n_layers_2hits"],
    }


# ===========================================================================
# ── EVALUATION UTILITIES (shared by both methods) ───────────────────────────
# ===========================================================================

def threshold_classify(scores, threshold):
    """Below threshold -> signal (1), above -> background (0)."""
    return (np.asarray(scores, dtype=np.float64) < threshold).astype(np.int32)


def evaluate_classifier(y_pred, y_true):
    """
    Precision, recall, F1, accuracy from binary predictions.

    Returns dict: precision, recall, f1, accuracy, tp, fp, tn, fn.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy  = (tp + tn) / max(len(y_pred), 1)
    return dict(precision=precision, recall=recall, f1=f1,
                accuracy=accuracy, tp=tp, fp=fp, tn=tn, fn=fn)


def roc_curve_points(scores_signal, scores_background, n_thresholds=200):
    """
    ROC curve by sweeping threshold over coplanarity scores (lower = more signal-like).

    Returns thresholds, tpr, fpr, auc.
    """
    scores_signal = np.asarray(scores_signal, dtype=np.float64)
    scores_background = np.asarray(scores_background, dtype=np.float64)
    all_s = np.concatenate([scores_signal, scores_background])
    if not len(all_s):
        raise ValueError("ROC calculation requires at least one score")
    unique = np.unique(all_s)
    if len(unique) <= n_thresholds - 2:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
    else:
        midpoints = np.linspace(all_s.min(), all_s.max(), n_thresholds - 2)
    # Explicit endpoints make tied/constant classifiers produce AUC=0.5.
    thrs = np.r_[-np.inf, midpoints, np.inf]
    tpr_l, fpr_l = [], []
    for thr in thrs:
        tp = (scores_signal    < thr).sum()
        fn = (scores_signal    >= thr).sum()
        fp = (scores_background < thr).sum()
        tn = (scores_background >= thr).sum()
        tpr_l.append(tp / max(tp + fn, 1))
        fpr_l.append(fp / max(fp + tn, 1))
    tpr = np.array(tpr_l)
    fpr = np.array(fpr_l)
    order = np.lexsort((tpr, fpr))
    auc   = float(np.trapz(tpr[order], fpr[order]))
    return thrs, tpr, fpr, auc
