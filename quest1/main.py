"""
EE1204: Engineering Electromagnetics
Simulation Assignment Set 1 - Question 1
2D Simulation of a Parallel-Plate Capacitor with Finite Dimensions

Method: Finite Difference Method (FDM) using nearest-neighbor averaging
        to iteratively solve Laplace's equation:
            d²V/dx² + d²V/dy² = 0

Domain: 5mm x 5mm square, discretized into 120x120 grid
Plate A (Top):    V = +5V at y = +5/3 mm, x in [-2.5, 2.5] mm
Plate B (Bottom): V = -5V at y = -5/3 mm, x in [-2.5, 2.5] mm
Outer boundary:   V = 0 (grounded)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────
# 1. GRID / DOMAIN SETUP
# ──────────────────────────────────────────────
N        = 120                      # grid points per side
L        = 5e-3                     # domain side length  [m]  (5 mm)
dx       = L / (N - 1)              # grid spacing in x   [m]
dy       = L / (N - 1)              # grid spacing in y   [m]

# Physical coordinates of each grid point
x = np.linspace(-L/2, L/2, N)      # x ∈ [-2.5 mm, +2.5 mm]
y = np.linspace(-L/2, L/2, N)      # y ∈ [-2.5 mm, +2.5 mm]
X, Y = np.meshgrid(x, y)           # shape (N, N); row i → y[i], col j → x[j]

# ──────────────────────────────────────────────
# 2. IDENTIFY PLATE NODE INDICES
# ──────────────────────────────────────────────
V_A =  5.0     # [V]  top    plate potential
V_B = -5.0     # [V]  bottom plate potential

y_A =  5e-3 / 3          # +5/3 mm  ≈ +1.6667 mm
y_B = -5e-3 / 3          # -5/3 mm  ≈ -1.6667 mm

# Find the row index closest to each plate y-position
row_A = np.argmin(np.abs(y - y_A))   # row for top    plate
row_B = np.argmin(np.abs(y - y_B))   # row for bottom plate

# Plate spans the full x-extent of the domain → all column indices
plate_cols = np.arange(N)           # columns 0 … N-1

print(f"Grid spacing dx = dy = {dx*1e3:.4f} mm")
print(f"Top    plate: row {row_A}  (y = {y[row_A]*1e3:.4f} mm)  V = {V_A} V")
print(f"Bottom plate: row {row_B}  (y = {y[row_B]*1e3:.4f} mm)  V = {V_B} V")

# ──────────────────────────────────────────────
# 3. INITIALISE POTENTIAL ARRAY
# ──────────────────────────────────────────────
V = np.zeros((N, N))                # all nodes start at 0 V

# Apply Dirichlet boundary conditions on the plates
V[row_A, plate_cols] = V_A
V[row_B, plate_cols] = V_B

# Outer edges are already 0 (grounded) — no extra action needed

# Boolean mask: True where a node is fixed (plate or outer boundary)
fixed = np.zeros((N, N), dtype=bool)
fixed[0,  :]  = True      # bottom outer boundary  (y = -2.5 mm)
fixed[-1, :]  = True      # top    outer boundary  (y = +2.5 mm)
fixed[:,  0]  = True      # left   outer boundary  (x = -2.5 mm)
fixed[:, -1]  = True      # right  outer boundary  (x = +2.5 mm)
fixed[row_A, plate_cols] = True
fixed[row_B, plate_cols] = True

# ──────────────────────────────────────────────
# 4. ITERATIVE SOLUTION — NEAREST-NEIGHBOUR AVERAGING
#    (Gauss-Seidel / Jacobi finite-difference update)
#    V[i,j] = 0.25 * (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])
# ──────────────────────────────────────────────
n_iter = 5000       # number of iterations (≥ 1000 as required)

print(f"\nRunning {n_iter} iterations …")

for iteration in range(n_iter):
    V_new = V.copy()

    # Update all interior (non-fixed) nodes
    V_new[1:-1, 1:-1] = 0.25 * (
        V[2:,   1:-1] +   # node above   (i+1, j)
        V[:-2,  1:-1] +   # node below   (i-1, j)
        V[1:-1, 2:]   +   # node right   (i, j+1)
        V[1:-1, :-2]      # node left    (i, j-1)
    )

    # Re-enforce fixed boundary conditions (plates + outer edges)
    V_new[fixed] = V[fixed]

    V = V_new

    if (iteration + 1) % 1000 == 0:
        print(f"  Iteration {iteration+1} complete")

print("Iteration complete.\n")

# ──────────────────────────────────────────────
# 5. COMPUTE ELECTRIC FIELD  E = -∇V
#    Using central differences for interior nodes
# ──────────────────────────────────────────────
# np.gradient returns [dV/dy (axis 0), dV/dx (axis 1)]
grad_y, grad_x = np.gradient(V, dy, dx)

Ex = -grad_x     # E_x = -dV/dx
Ey = -grad_y     # E_y = -dV/dy
E_mag = np.sqrt(Ex**2 + Ey**2)    # |E|

# ──────────────────────────────────────────────
# 6. PLOTS
# ──────────────────────────────────────────────
x_mm = x * 1e3          # convert to mm for axis labels
y_mm = y * 1e3
X_mm, Y_mm = X * 1e3, Y * 1e3

# ── 6a. Equipotential contour map ──────────────
fig1, ax1 = plt.subplots(figsize=(7, 7))

contourf = ax1.contourf(X_mm, Y_mm, V,
                        levels=50, cmap='RdBu_r', alpha=0.85)
contour  = ax1.contour(X_mm, Y_mm, V,
                       levels=20, colors='black', linewidths=0.7)
ax1.clabel(contour, fmt='%.1f V', fontsize=7, inline=True)

cbar = fig1.colorbar(contourf, ax=ax1, pad=0.02)
cbar.set_label('Electric Potential V (volts)', fontsize=11)

# Draw plate markers
ax1.axhline(y[row_A]*1e3, color='red',  linewidth=2.5,
            label=f'Plate A  V = +5 V  (y = {y[row_A]*1e3:.2f} mm)')
ax1.axhline(y[row_B]*1e3, color='blue', linewidth=2.5,
            label=f'Plate B  V = −5 V  (y = {y[row_B]*1e3:.2f} mm)')

ax1.set_xlabel('x  (mm)', fontsize=12)
ax1.set_ylabel('y  (mm)', fontsize=12)
ax1.set_title('Equipotential Contour Map  V(x, y)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig('plot1_equipotential.png', dpi=150)
print("Saved: plot1_equipotential.png")

# ── 6b. Electric field lines (streamplot) ──────
fig2, ax2 = plt.subplots(figsize=(7, 7))

# Background: potential colour map
cf2 = ax2.contourf(X_mm, Y_mm, V, levels=50, cmap='RdBu_r', alpha=0.6)
cbar2 = fig2.colorbar(cf2, ax=ax2, pad=0.02)
cbar2.set_label('Electric Potential V (volts)', fontsize=11)

# Streamlines of E — seed density proportional to field strength
speed = np.sqrt(Ex**2 + Ey**2)
lw    = 1.5 * speed / (speed.max() + 1e-20)   # variable line width

ax2.streamplot(X_mm, Y_mm, Ex, Ey,
               color='black', linewidth=lw,
               density=1.4, arrowsize=1.2, arrowstyle='->')

# Plate markers
ax2.axhline(y[row_A]*1e3, color='red',  linewidth=2.5,
            label=f'Plate A  V = +5 V')
ax2.axhline(y[row_B]*1e3, color='blue', linewidth=2.5,
            label=f'Plate B  V = −5 V')

ax2.set_xlabel('x  (mm)', fontsize=12)
ax2.set_ylabel('y  (mm)', fontsize=12)
ax2.set_title('Electric Field Lines  E(x, y)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_aspect('equal')
ax2.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig('plot2_efield_lines.png', dpi=150)
print("Saved: plot2_efield_lines.png")

# ── 6c. |E| along vertical centre-line x = 0 ──
fig3, ax3 = plt.subplots(figsize=(8, 5))

mid_col  = np.argmin(np.abs(x))          # column index for x ≈ 0
E_centre = E_mag[:, mid_col]             # |E| along x = 0 for all y

ax3.plot(y_mm, E_centre, color='darkorange', linewidth=2)

# Shade the inter-plate region
ax3.axvspan(y[row_B]*1e3, y[row_A]*1e3,
            alpha=0.15, color='green', label='Between plates')
ax3.axvline(y[row_A]*1e3, color='red',  linestyle='--', linewidth=1.5,
            label=f'Plate A  (y = {y[row_A]*1e3:.2f} mm)')
ax3.axvline(y[row_B]*1e3, color='blue', linestyle='--', linewidth=1.5,
            label=f'Plate B  (y = {y[row_B]*1e3:.2f} mm)')

ax3.set_xlabel('y  (mm)', fontsize=12)
ax3.set_ylabel('|E|  (V/m)', fontsize=12)
ax3.set_title('Electric Field Magnitude Along Vertical Centre-Line (x = 0)',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig('plot3_E_centreline.png', dpi=150)
print("Saved: plot3_E_centreline.png")

# ── 6d. Combined summary figure ────────────────
fig4, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: equipotential
ax = axes[0]
cf = ax.contourf(X_mm, Y_mm, V, levels=50, cmap='RdBu_r', alpha=0.85)
ax.contour(X_mm, Y_mm, V, levels=15, colors='black', linewidths=0.6)
fig4.colorbar(cf, ax=ax, label='V (volts)', pad=0.02)
ax.axhline(y[row_A]*1e3, color='red',  lw=2)
ax.axhline(y[row_B]*1e3, color='blue', lw=2)
ax.set_title('Equipotential Map', fontweight='bold')
ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
ax.set_aspect('equal')

# Middle: field lines
ax = axes[1]
cf2 = ax.contourf(X_mm, Y_mm, V, levels=50, cmap='RdBu_r', alpha=0.5)
ax.streamplot(X_mm, Y_mm, Ex, Ey,
              color='black', linewidth=1.0, density=1.2,
              arrowsize=1.0, arrowstyle='->')
ax.axhline(y[row_A]*1e3, color='red',  lw=2, label='Plate A (+5V)')
ax.axhline(y[row_B]*1e3, color='blue', lw=2, label='Plate B (−5V)')
ax.set_title('Electric Field Lines', fontweight='bold')
ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
ax.legend(fontsize=8)
ax.set_aspect('equal')

# Right: |E| along centre-line
ax = axes[2]
ax.plot(y_mm, E_centre, color='darkorange', lw=2)
ax.axvspan(y[row_B]*1e3, y[row_A]*1e3, alpha=0.15, color='green',
           label='Between plates')
ax.axvline(y[row_A]*1e3, color='red',  ls='--', lw=1.5)
ax.axvline(y[row_B]*1e3, color='blue', ls='--', lw=1.5)
ax.set_title('|E| Along Centre-Line (x = 0)', fontweight='bold')
ax.set_xlabel('y (mm)'); ax.set_ylabel('|E|  (V/m)')
ax.legend(fontsize=8); ax.grid(True, ls=':')

fig4.suptitle(
    'EE1204 – 2D Parallel-Plate Capacitor Simulation\n'
    '(Laplace FDM, N=120, 5000 iterations)',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot4_summary.png', dpi=150)
print("Saved: plot4_summary.png")

plt.show()
print("\nAll plots saved successfully.")
