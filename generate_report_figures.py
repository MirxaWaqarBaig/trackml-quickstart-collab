"""Generate figures for the quirk-module-intersection branch report."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Pipelines.TrackML_Example.LightningModules.Processing.utils.quirk_intersections import (
    build_detector_module_table,
    intersect_track_with_modules,
    compute_before_after_stats,
)
from Pipelines.TrackML_Example.LightningModules.Processing.utils.quirk_trajectory import (
    simulate_quirk_pair_tracks,
)

OUT = Path('/home/waqar/rafia_clone/docs_from_june')
OUT.mkdir(parents=True, exist_ok=True)

print("Loading detector...")
det_df = pd.read_csv('trackml_raw/detectors.csv')
mods   = build_detector_module_table(det_df)
print(f"  {len(mods)} modules, half_u: {mods.half_u.min():.1f}-{mods.half_u.max():.1f} mm")

# -----------------------------------------------------------------------
# Figure 1: Outward helix test — XY and RZ views with exact hits
# -----------------------------------------------------------------------
print("Figure 1: helix test track...")
N = 3000
t  = np.linspace(0, 1, N)
r  = 900.0 * t
ph = 2.5 * t
z  = 200.0 * t
traj_helix = np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=1)

diag = []
df_new = intersect_track_with_modules(traj_helix, mods, max_hits=150, diagnostic_log=diag)
stats  = compute_before_after_stats(traj_helix, mods, max_hits=150)
print(f"  Old (proximity): {stats['old_hits']}  New (exact): {stats['new_hits']}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Exact Geometry vs Proximity Method — Outward Helix Test Track',
    fontsize=13, fontweight='bold'
)
ax = axes[0]
ax.plot(traj_helix[:, 0], traj_helix[:, 1], 'b-', lw=0.6, alpha=0.4, label='Trajectory')
if len(df_new):
    ax.scatter(df_new['x'], df_new['y'], c='red', s=60, zorder=5,
               label=f'Exact geometry hits ({len(df_new)})')
ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_title('XY View')
ax.legend(fontsize=9); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

ax = axes[1]
r_traj = np.sqrt(traj_helix[:, 0]**2 + traj_helix[:, 1]**2)
ax.plot(traj_helix[:, 2], r_traj, 'b-', lw=0.6, alpha=0.4, label='Trajectory')
if len(df_new):
    ax.scatter(df_new['z'], df_new['r'], c='red', s=60, zorder=5,
               label=f'Exact geometry hits ({len(df_new)})')
ax.set_xlabel('z [mm]'); ax.set_ylabel('r [mm]'); ax.set_title('RZ View')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT / 'helix_exact_hits.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved helix_exact_hits.png")

# -----------------------------------------------------------------------
# Figure 2: local u/v scatter — accepted vs rejected candidates
# -----------------------------------------------------------------------
print("Figure 2: local u/v scatter...")
inside_u  = [rec['local_u_mm'] for rec in diag if rec['inside']]
inside_v  = [rec['local_v_mm'] for rec in diag if rec['inside']]
outside_u = [rec['local_u_mm'] for rec in diag if not rec['inside']]
outside_v = [rec['local_v_mm'] for rec in diag if not rec['inside']]

fig, ax = plt.subplots(figsize=(7, 6))
if outside_u:
    ax.scatter(outside_u, outside_v, c='salmon', s=70, alpha=0.8,
               label=f'Rejected (outside boundary): {len(outside_u)}', zorder=3)
if inside_u:
    ax.scatter(inside_u, inside_v, c='green', s=100, marker='*',
               label=f'Accepted (inside boundary): {len(inside_u)}', zorder=4)
ax.axvline(0, color='gray', lw=0.5, ls='--')
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.set_xlabel('local_u [mm]', fontsize=11)
ax.set_ylabel('local_v [mm]', fontsize=11)
ax.set_title(
    'Module Local Coordinates of Intersection Candidates\n'
    '(first 20 candidates from helix test, accepted green / rejected red)',
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT / 'local_uv_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved local_uv_scatter.png")

# -----------------------------------------------------------------------
# Figure 3: Old vs New hit counts on simulated quirk tracks
# -----------------------------------------------------------------------
print("Figure 3: simulated quirk old vs new...")
old_hits_list, new_hits_list, labels = [], [], []
for i in range(6):
    result = simulate_quirk_pair_tracks(
        event_id=i, n_steps=3000, pair_pt=6.0, pair_pz=3.0,
        b_field=2.0, string_tension=0.2, velocity_scale=1200.0,
    )
    for lbl, key in [('quirk', 'xyz_q'), ('anti-quirk', 'xyz_aq')]:
        traj = np.asarray(result[key])
        s    = compute_before_after_stats(traj, mods, max_hits=150)
        old_hits_list.append(s['old_hits'])
        new_hits_list.append(s['new_hits'])
        labels.append(f'ev{i} {lbl}')

x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(14, 5))
bars1 = ax.bar(x - w / 2, old_hits_list, w,
               label='Old: proximity (tol=1000mm) — FAKE HITS', color='#e07070')
bars2 = ax.bar(x + w / 2, new_hits_list, w,
               label='New: exact geometry — REAL crossings only', color='#5aaa7a')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Hits recorded', fontsize=11)
ax.set_title(
    'Old vs New Hit Counts on Simulated Quirk Tracks\n'
    '(sep0=900mm: quirks start at r=450mm, path length ~45mm — never cross module planes)',
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2, str(int(h)),
                ha='center', va='bottom', fontsize=8, color='#c04040')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2, str(int(h)),
            ha='center', va='bottom', fontsize=8, color='#2a8a5a')
plt.tight_layout()
plt.savefig(str(OUT / 'old_vs_new_hits.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved old_vs_new_hits.png")

# -----------------------------------------------------------------------
# Figure 4: Module geometry diagram — visual explanation
# -----------------------------------------------------------------------
print("Figure 4: geometry diagram...")
fig, ax = plt.subplots(figsize=(8, 7))
Cm   = np.array([70.0, 14.0])
hu   = 8.4
hv   = 36.0
u_ax2 = np.array([0.97, -0.24])
v_ax2 = np.array([0.24,  0.97])
X1 = Cm + hu * u_ax2 + hv * v_ax2
X2 = Cm - hu * u_ax2 + hv * v_ax2
X3 = Cm - hu * u_ax2 - hv * v_ax2
X4 = Cm + hu * u_ax2 - hv * v_ax2
corners = np.array([X1, X2, X3, X4, X1])
ax.fill(corners[:-1, 0], corners[:-1, 1], alpha=0.25, color='steelblue')
ax.plot(corners[:, 0], corners[:, 1], 'b-', lw=2.5, label='Module active surface (X1-X4)')
ax.plot(*Cm, 'bs', ms=10, label='Module center C', zorder=5)
# label corners
for lbl, pt in [('X1', X1), ('X2', X2), ('X3', X3), ('X4', X4)]:
    ax.text(pt[0] + 1, pt[1] + 1, lbl, fontsize=9, color='steelblue', fontweight='bold')
# local axes
ax.annotate('', xy=Cm + 12 * u_ax2, xytext=Cm,
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
ax.text(*(Cm + 13 * u_ax2), 'u-axis', fontsize=9, color='navy')
ax.annotate('', xy=Cm + 12 * v_ax2, xytext=Cm,
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
ax.text(*(Cm + 13 * v_ax2), 'v-axis', fontsize=9, color='navy')
# track segment
A_ = np.array([55.0, -25.0])
B_ = np.array([82.0,  48.0])
n_mod = v_ax2
denom_ = float(np.dot(n_mod, B_ - A_))
t_     = float(np.dot(n_mod, Cm - A_)) / denom_
F_     = A_ + t_ * (B_ - A_)
ax.annotate('', xy=B_, xytext=A_,
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2.5))
ax.plot(*A_, 'o', color='darkorange', ms=10, zorder=5)
ax.plot(*B_, 'o', color='darkorange', ms=10, zorder=5)
ax.text(A_[0] - 8, A_[1] - 2, 'A  (step i)', fontsize=10, color='darkorange')
ax.text(B_[0] + 1, B_[1] + 1, 'B  (step i+1)', fontsize=10, color='darkorange')
ax.plot(*F_, 'r*', ms=18, zorder=6, label=f'Hit F  (t = {t_:.2f})')
ax.annotate(
    'F = A + t*(B-A)\n'
    'Check: |local_u| <= half_u\n'
    '          |local_v| <= half_v',
    xy=F_, xytext=(F_[0] + 8, F_[1] - 16),
    fontsize=9, color='darkred',
    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.3),
)
ax.set_xlabel('x [mm]', fontsize=11)
ax.set_ylabel('y [mm]', fontsize=11)
ax.set_title(
    "Professor's Method: Segment-Plane Intersection\n"
    "with Strict Module Boundary Check",
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(str(OUT / 'module_geometry_diagram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved module_geometry_diagram.png")

print("\nAll 4 figures generated successfully.")
