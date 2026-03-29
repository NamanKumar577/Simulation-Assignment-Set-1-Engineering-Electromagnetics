"""
EE1204: Engineering Electromagnetics
Simulation Assignment Set 1 - Question 3
Simulation of Electric Field Enhancement at a Sharp Conductor (Lightning Rod)

Method: Finite Difference Method (FDM) — Gauss-Seidel nearest-neighbour
        averaging to iteratively solve Laplace's equation:
            d²V/dx² + d²V/dy² = 0

Grid: 100 × 100
Boundary conditions:
  - Bottom row  (y = 0):         V = 0    (ground plane)
  - Top    row  (y = N-1):       V = 100  (charged cloud)
  - Left/Right columns:          linearly interpolated (natural BC)
  - Needle conductor (V = 0):    vertical line at x = 50,
                                 from row 0 (ground) up to row 49 (mid-grid)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

# ─────────────────────────────────────────────────────────────
# 1.  GRID AND BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────
N = 100                          # grid size N × N

# Initialise with a linear interpolation between V=0 (bottom) and V=100 (top)
# This gives a warm start that speeds up convergence
row_idx = np.arange(N)
V = np.zeros((N, N))
for i in range(N):
    V[i, :] = (i / (N - 1)) * 100.0   # rows 0→0V, rows N-1→100V

# ── Fixed-node mask ──────────────────────────────────────────
fixed = np.zeros((N, N), dtype=bool)

# Bottom boundary: V = 0  (ground)
V[0,  :] = 0.0
fixed[0, :] = True

# Top boundary: V = 100  (charged cloud)
V[-1, :] = 100.0
fixed[-1, :] = True

# Left and right boundaries — held at linear interpolation
V[:,  0] = row_idx / (N - 1) * 100.0
V[:, -1] = row_idx / (N - 1) * 100.0
fixed[:,  0] = True
fixed[:, -1] = True

# ── Needle conductor ─────────────────────────────────────────
# Vertical line at centre column (x = 49, 0-indexed)
# Extends from the ground (row 0) to the middle of the grid (row 49)
needle_col       = N // 2        # column 50  (x-centre)
needle_row_start = 0             # ground end
needle_row_end   = N // 2        # mid-grid tip  (row 49 → row 50 exclusive)

needle_rows = np.arange(needle_row_start, needle_row_end)
V[needle_rows, needle_col]     = 0.0
fixed[needle_rows, needle_col] = True

print(f"Grid: {N}×{N}")
print(f"Needle: column {needle_col}, rows {needle_row_start}–{needle_row_end-1}")
print(f"Needle tip at grid index (row={needle_row_end-1}, col={needle_col})\n")

# ─────────────────────────────────────────────────────────────
# 2.  ITERATIVE SOLUTION — convergence-checked Gauss-Seidel
# ─────────────────────────────────────────────────────────────
max_iter  = 20000
tol       = 1e-5      # convergence threshold (max change per iteration)

print("Solving Laplace's equation …")
for iteration in range(1, max_iter + 1):
    V_old = V.copy()

    # Nearest-neighbour averaging for all interior nodes
    V[1:-1, 1:-1] = 0.25 * (
        V[2:,   1:-1] +   # above
        V[:-2,  1:-1] +   # below
        V[1:-1, 2:]   +   # right
        V[1:-1, :-2]      # left
    )

    # Re-enforce all fixed boundaries
    V[fixed] = V_old[fixed]

    # Check convergence
    max_change = np.max(np.abs(V - V_old))
    if iteration % 2000 == 0:
        print(f"  Iter {iteration:6d} | max ΔV = {max_change:.2e}")
    if max_change < tol:
        print(f"\nConverged after {iteration} iterations  (max ΔV = {max_change:.2e})\n")
        break
else:
    print(f"\nReached max iterations ({max_iter}).\n")

# ─────────────────────────────────────────────────────────────
# 3.  ELECTRIC FIELD  E = -∇V
# ─────────────────────────────────────────────────────────────
# Using numpy central differences (unit grid spacing)
grad_y, grad_x = np.gradient(V)   # [dV/dy, dV/dx]  (axis 0 = y, axis 1 = x)
Ex = -grad_x
Ey = -grad_y
E_mag = np.sqrt(Ex**2 + Ey**2)

# Coordinate arrays (in grid-index units, 0–99)
x_arr = np.arange(N)
y_arr = np.arange(N)
X, Y  = np.meshgrid(x_arr, y_arr)

# Tip of the needle
tip_row = needle_row_end - 1      # row 49
tip_col = needle_col              # col 50
E_tip   = E_mag[tip_row, tip_col]
print(f"Electric field magnitude at needle tip: {E_tip:.4f}  (V/grid-unit)")

# For comparison, reference field far from needle (mid-height, left side)
E_ref = E_mag[N//2, 5]
print(f"Reference field (mid-height, away from needle): {E_ref:.4f}")
if E_ref > 0:
    print(f"Field enhancement factor at tip: {E_tip/E_ref:.2f}×\n")

# ─────────────────────────────────────────────────────────────
# 4.  PLOTS
# ─────────────────────────────────────────────────────────────

# ── PLOT 1: Potential heat map + E-field vectors ──────────────
fig1, ax1 = plt.subplots(figsize=(8, 8))

cf = ax1.imshow(V, origin='lower', cmap='hot', aspect='equal',
                extent=[0, N-1, 0, N-1])
cbar = fig1.colorbar(cf, ax=ax1, pad=0.02, label='Electric Potential V (volts)')
cbar.set_label('Electric Potential  V  (volts)', fontsize=11)

# Overlay electric field vectors (quiver), downsampled
step = 5
ax1.quiver(X[::step, ::step], Y[::step, ::step],
           Ex[::step, ::step], Ey[::step, ::step],
           color='cyan', scale=400, width=0.003,
           headwidth=3, headlength=4, alpha=0.85)

# Draw the needle
ax1.plot([needle_col, needle_col],
         [needle_row_start, needle_row_end - 1],
         color='white', lw=3, label='Needle conductor (V=0)', zorder=5)
ax1.plot(needle_col, tip_row, 'w^', ms=10, zorder=6, label='Needle tip')

ax1.set_xlabel('x  (grid units)', fontsize=12)
ax1.set_ylabel('y  (grid units)', fontsize=12)
ax1.set_title('Potential Heat Map & Electric Field Vectors\n'
              '(Lightning Rod Simulation)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9, framealpha=0.7)
ax1.text(2, 95, 'Cloud  (V = 100 V)', color='white', fontsize=9,
         va='top', fontweight='bold')
ax1.text(2, 2,  'Ground (V = 0 V)',   color='yellow', fontsize=9,
         va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('q3_plot1_heatmap_vectors.png', dpi=150)
print("Saved: q3_plot1_heatmap_vectors.png")

# ── PLOT 2: Equipotential contours ────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 8))

cf2 = ax2.contourf(X, Y, V, levels=50, cmap='plasma', alpha=0.85)
ct2 = ax2.contour(X, Y, V, levels=20, colors='white',
                  linewidths=0.6, alpha=0.8)
ax2.clabel(ct2, fmt='%d V', fontsize=7, inline=True)
fig2.colorbar(cf2, ax=ax2, label='V (volts)', pad=0.02)

ax2.plot([needle_col, needle_col],
         [needle_row_start, needle_row_end - 1],
         color='cyan', lw=3, label='Needle (V=0)', zorder=5)
ax2.plot(needle_col, tip_row, 'c^', ms=10, zorder=6, label='Tip')

ax2.set_xlabel('x  (grid units)', fontsize=12)
ax2.set_ylabel('y  (grid units)', fontsize=12)
ax2.set_title('Equipotential Contour Map\n(Lightning Rod Simulation)',
              fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.set_aspect('equal')
plt.tight_layout()
plt.savefig('q3_plot2_equipotential.png', dpi=150)
print("Saved: q3_plot2_equipotential.png")

# ── PLOT 3: |E| heat map highlighting tip enhancement ─────────
fig3, ax3 = plt.subplots(figsize=(8, 8))

# Cap E_mag for display so the tip doesn't swamp the colour scale
E_display = np.clip(E_mag, 0, np.percentile(E_mag, 99))
cf3 = ax3.imshow(E_display, origin='lower', cmap='inferno',
                 aspect='equal', extent=[0, N-1, 0, N-1])
fig3.colorbar(cf3, ax=ax3, label='|E|  (V/grid-unit)  [capped at 99th pct]',
              pad=0.02)

ax3.plot([needle_col, needle_col],
         [needle_row_start, needle_row_end - 1],
         color='white', lw=3, label='Needle (V=0)', zorder=5)
ax3.plot(needle_col, tip_row, 'w^', ms=12, zorder=6,
         label=f'Tip  |E| = {E_tip:.2f}')

# Annotate tip
ax3.annotate(f'|E| = {E_tip:.2f}\n(tip enhancement)',
             xy=(needle_col, tip_row),
             xytext=(needle_col + 8, tip_row + 8),
             color='white', fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

ax3.set_xlabel('x  (grid units)', fontsize=12)
ax3.set_ylabel('y  (grid units)', fontsize=12)
ax3.set_title('Electric Field Magnitude  |E(x,y)|\n'
              '(Showing Enhancement at Needle Tip)',
              fontsize=13, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9, framealpha=0.7)
plt.tight_layout()
plt.savefig('q3_plot3_Emag.png', dpi=150)
print("Saved: q3_plot3_Emag.png")

# ── PLOT 4: |E| along the vertical centre-line (x = needle_col) ─
fig4, ax4 = plt.subplots(figsize=(9, 5))

E_centreline = E_mag[:, needle_col]

ax4.plot(y_arr, E_centreline, color='darkorange', lw=2.5,
         label='|E| at x = centre (needle column)')
ax4.axvline(tip_row, color='red', ls='--', lw=1.8,
            label=f'Needle tip  (row {tip_row})')
ax4.axvspan(needle_row_start, needle_row_end - 1,
            alpha=0.15, color='blue', label='Needle body')

ax4.set_xlabel('y  (grid row index)', fontsize=12)
ax4.set_ylabel('|E|  (V/grid-unit)', fontsize=12)
ax4.set_title('Electric Field Magnitude Along Vertical Centre-Line\n'
              '(Needle Column,  x = 50)',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, ls=':', lw=0.5)
plt.tight_layout()
plt.savefig('q3_plot4_centreline.png', dpi=150)
print("Saved: q3_plot4_centreline.png")

# ── PLOT 5: Summary 2×2 panel ────────────────────────────────
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 13))
fig5.suptitle('EE1204 – Q3: Lightning Rod Field Enhancement Simulation\n'
              '(100×100 FDM, Laplace\'s Equation)',
              fontsize=13, fontweight='bold')

# (0,0) Potential heatmap
ax = axes5[0, 0]
im = ax.imshow(V, origin='lower', cmap='hot', aspect='equal',
               extent=[0, N-1, 0, N-1])
fig5.colorbar(im, ax=ax, label='V (volts)', shrink=0.9)
ax.plot([needle_col]*2, [needle_row_start, needle_row_end-1],
        'w-', lw=3)
ax.plot(needle_col, tip_row, 'w^', ms=9)
ax.set_title('Potential Heat Map', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')

# (0,1) Equipotential + E vectors
ax = axes5[0, 1]
cf = ax.contourf(X, Y, V, levels=40, cmap='plasma', alpha=0.8)
ax.contour(X, Y, V, levels=15, colors='white', linewidths=0.5, alpha=0.6)
step2 = 6
ax.quiver(X[::step2, ::step2], Y[::step2, ::step2],
          Ex[::step2, ::step2], Ey[::step2, ::step2],
          color='cyan', scale=500, width=0.003, alpha=0.9)
ax.plot([needle_col]*2, [needle_row_start, needle_row_end-1],
        'w-', lw=3, label='Needle')
ax.plot(needle_col, tip_row, 'w^', ms=9)
ax.set_title('Equipotentials + E Vectors', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_aspect('equal')

# (1,0) |E| magnitude map
ax = axes5[1, 0]
im2 = ax.imshow(E_display, origin='lower', cmap='inferno', aspect='equal',
                extent=[0, N-1, 0, N-1])
fig5.colorbar(im2, ax=ax, label='|E| (V/unit)', shrink=0.9)
ax.plot([needle_col]*2, [needle_row_start, needle_row_end-1],
        'w-', lw=3)
ax.plot(needle_col, tip_row, 'w^', ms=9,
        label=f'Tip |E|={E_tip:.2f}')
ax.annotate(f'|E|={E_tip:.2f}',
            xy=(needle_col, tip_row),
            xytext=(needle_col+7, tip_row+7),
            color='white', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='white'))
ax.set_title('|E| Magnitude (Field Enhancement)', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')

# (1,1) Centre-line profile
ax = axes5[1, 1]
ax.plot(y_arr, E_centreline, color='darkorange', lw=2)
ax.axvline(tip_row, color='red', ls='--', lw=1.5, label='Needle tip')
ax.axvspan(0, needle_row_end-1, alpha=0.1, color='blue', label='Needle')
ax.set_xlabel('y (row index)'); ax.set_ylabel('|E| (V/unit)')
ax.set_title('|E| Along Centre-Line (x=50)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, ls=':', lw=0.4)

plt.tight_layout()
plt.savefig('q3_plot5_summary.png', dpi=150)
print("Saved: q3_plot5_summary.png")

plt.show()
print("\nAll Q3 plots saved successfully.")

# ─────────────────────────────────────────────────────────────
# 5.  PHYSICAL INTERPRETATION  (printed to console)
# ─────────────────────────────────────────────────────────────
print("""
════════════════════════════════════════════════════════
  PHYSICAL INTERPRETATION — NEEDLE TIP FIELD ENHANCEMENT
════════════════════════════════════════════════════════

1. UNIFORM FIELD (no needle):
   Without the needle, V varies linearly from 0V (ground)
   to 100V (cloud), giving a uniform vertical E-field.

2. DISTORTION BY THE NEEDLE:
   The grounded conducting needle (V=0) disrupts the uniform
   field. Because the needle is a conductor, equipotential
   lines are forced to be perpendicular to its surface.
   They bunch tightly near the sharp tip.

3. FIELD ENHANCEMENT AT THE TIP:
   Dense equipotential packing → large |∇V| → large |E|.
   This is the "lightning rod effect": the radius of curvature
   at the tip is very small, so the surface charge density and
   hence E diverges as r → 0 (theoretically ∝ 1/√r for a
   paraboloidal tip).

4. LIGHTNING ROD MECHANISM:
   The intense field at the tip ionises the surrounding air,
   creating a conducting plasma channel. This provides a
   preferred low-resistance path for lightning discharge,
   safely directing the current to ground.
════════════════════════════════════════════════════════
""")
