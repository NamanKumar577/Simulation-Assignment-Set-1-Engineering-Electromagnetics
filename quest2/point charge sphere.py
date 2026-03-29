"""
EE1204: Engineering Electromagnetics
Simulation Assignment Set 1 - Question 2
2D Simulation of a Point Charge Near a Grounded Conducting Sphere

Method: Method of Images (analytical)
  A grounded conducting sphere of radius R centred at origin, with an external
  point charge Q at distance d from the centre, is exactly equivalent to:
    - The original charge  Q  at position (d, 0)
    - An image charge      Q' = -Q * R/d   at position (R²/d, 0)

  This guarantees V = 0 on the sphere surface and satisfies Laplace's equation
  everywhere outside the sphere.

Physical setup (simulation units):
  R  = 2          sphere radius
  Q  = 10 µC      point charge magnitude
  d  = 4          default distance of Q from origin  →  placed at (4, 0)
  Domain: -6 ≤ x ≤ 6,  -6 ≤ y ≤ 6
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
eps0 = 8.854187817e-12      # permittivity of free space [F/m]
k_e  = 1.0 / (4 * np.pi * eps0)   # Coulomb constant [N·m²/C²]

# Simulation uses dimensionless units (1 unit = 1 m for field calc)
# Charge given in µC → convert to C
Q_base = 10e-6       # [C]
R      = 2.0         # sphere radius  [sim units]

# ─────────────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────────────
Ngrid = 400
x_arr = np.linspace(-6, 6, Ngrid)
y_arr = np.linspace(-6, 6, Ngrid)
X, Y  = np.meshgrid(x_arr, y_arr)

# ─────────────────────────────────────────────────────────────
# HELPER: compute V, Ex, Ey and surface charge density σ
#         for a given Q and charge position (d, 0)
# ─────────────────────────────────────────────────────────────
def compute_fields(Q, d):
    """
    Returns V, Ex, Ey on the grid and σ on the sphere surface.

    Image charge:
        Q'  = -Q * R / d          (magnitude)
        pos = (R² / d, 0)         (location inside sphere)

    Surface charge density (method of images result):
        σ(θ) = -Q(d²-R²) / (4π R (d²+R²-2Rd cosθ)^(3/2))
    where θ is the polar angle on the sphere.
    """
    # Image charge properties
    Q_img  = -Q * R / d
    x_img  =  R**2 / d         # always inside sphere (x_img < R when d > R)

    # ── Potential V = V_real + V_image ──────────────────────
    # Distance from every grid point to real charge at (d, 0)
    r_real = np.sqrt((X - d)**2 + Y**2)
    # Distance to image charge at (x_img, 0)
    r_img  = np.sqrt((X - x_img)**2 + Y**2)

    # Avoid division by zero very close to charges
    r_real = np.where(r_real < 1e-9, 1e-9, r_real)
    r_img  = np.where(r_img  < 1e-9, 1e-9, r_img)

    V = k_e * (Q / r_real + Q_img / r_img)

    # ── Electric field E = -∇V  (analytical) ────────────────
    # Contribution from real charge
    Ex_real = k_e * Q * (X - d)   / r_real**3
    Ey_real = k_e * Q *  Y        / r_real**3
    # Contribution from image charge
    Ex_img  = k_e * Q_img * (X - x_img) / r_img**3
    Ey_img  = k_e * Q_img *  Y          / r_img**3

    Ex = Ex_real + Ex_img
    Ey = Ey_real + Ey_img

    # ── Mask interior of sphere  (set to NaN — not physical) ─
    inside = (X**2 + Y**2) <= R**2
    V[inside]  = np.nan
    Ex[inside] = np.nan
    Ey[inside] = np.nan

    # ── Surface charge density σ(θ) ──────────────────────────
    # θ = 0 is toward the point charge
    theta = np.linspace(0, 2 * np.pi, 1000)
    denom = (d**2 + R**2 - 2*R*d*np.cos(theta))**1.5
    sigma = -Q * (d**2 - R**2) / (4 * np.pi * R * denom)

    return V, Ex, Ey, theta, sigma


# ─────────────────────────────────────────────────────────────
# ── CASE 1: DEFAULT  Q=10µC, d=4 ─────────────────────────────
# ─────────────────────────────────────────────────────────────
Q1, d1 = Q_base, 4.0
V1, Ex1, Ey1, theta1, sigma1 = compute_fields(Q1, d1)

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
fig.suptitle(
    f'Point Charge Near Grounded Conducting Sphere\n'
    f'Q = {Q1*1e6:.0f} µC  at  (d=4, 0),   R = {R}',
    fontsize=13, fontweight='bold')

# ── (A) Equipotential contours ────────────────────────────────
ax = axes[0]
# Clamp V for clean contours (near charges it diverges)
V_plot = np.clip(V1, -2e6, 2e6)
norm   = TwoSlopeNorm(vmin=-2e6, vcenter=0, vmax=2e6)
cf = ax.contourf(X, Y, V_plot, levels=60, cmap='RdBu_r',
                 norm=norm, alpha=0.85)
ax.contour(X, Y, V_plot, levels=30, colors='black',
           linewidths=0.5, alpha=0.7)
fig.colorbar(cf, ax=ax, label='V  (volts)', pad=0.02, shrink=0.85)

sphere = plt.Circle((0, 0), R, color='silver', ec='black', lw=1.5, zorder=5)
ax.add_patch(sphere)
ax.plot(d1, 0, 'r*', ms=14, zorder=6, label=f'Q = {Q1*1e6:.0f} µC')
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.set_xlabel('x  (units)'); ax.set_ylabel('y  (units)')
ax.set_title('Equipotential Contours', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, ls=':', lw=0.4, alpha=0.5)

# ── (B) Electric field vector distribution ────────────────────
ax = axes[1]
cf2 = ax.contourf(X, Y, V_plot, levels=40, cmap='RdBu_r',
                  norm=norm, alpha=0.5)
# Downsample for quiver
step = 16
ax.quiver(X[::step, ::step], Y[::step, ::step],
          Ex1[::step, ::step], Ey1[::step, ::step],
          color='black', scale=5e8, width=0.003,
          headwidth=3, headlength=4)

sphere2 = plt.Circle((0, 0), R, color='silver', ec='black', lw=1.5, zorder=5)
ax.add_patch(sphere2)
ax.plot(d1, 0, 'r*', ms=14, zorder=6)
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.set_xlabel('x  (units)'); ax.set_ylabel('y  (units)')
ax.set_title('Electric Field Vectors  E(x,y)', fontweight='bold')
ax.grid(True, ls=':', lw=0.4, alpha=0.5)

# ── (C) Induced surface charge density σ(θ) ──────────────────
ax = axes[2]
ax.plot(np.degrees(theta1), sigma1 * 1e6,
        color='darkgreen', lw=2)
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.fill_between(np.degrees(theta1), sigma1 * 1e6, 0,
                where=(sigma1 < 0), alpha=0.3, color='blue',
                label='Negative σ (attracted)')
ax.fill_between(np.degrees(theta1), sigma1 * 1e6, 0,
                where=(sigma1 > 0), alpha=0.3, color='red',
                label='Positive σ')
ax.set_xlabel('θ  (degrees,  θ=0 toward charge)', fontsize=10)
ax.set_ylabel('σ  (µC/m²)', fontsize=10)
ax.set_title('Induced Surface Charge Density', fontweight='bold')
ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
ax.legend(fontsize=9); ax.grid(True, ls=':', lw=0.5)

plt.tight_layout()
plt.savefig('q2_plot1_default.png', dpi=150)
print("Saved: q2_plot1_default.png")


# ─────────────────────────────────────────────────────────────
# ── CASE 2: VARYING CHARGE MAGNITUDE  (d fixed = 4) ──────────
# ─────────────────────────────────────────────────────────────
Q_values = [5e-6, 10e-6, 20e-6, 40e-6]   # 5, 10, 20, 40 µC
fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))
fig2.suptitle('Effect of Varying Charge Magnitude  (d = 4, R = 2)',
              fontsize=13, fontweight='bold')

for col, Q_val in enumerate(Q_values):
    V_c, Ex_c, Ey_c, th_c, sig_c = compute_fields(Q_val, d=4.0)
    V_p = np.clip(V_c, -2e6, 2e6)
    norm_c = TwoSlopeNorm(vmin=-2e6, vcenter=0, vmax=2e6)

    # Top row: equipotential
    ax = axes2[0, col]
    cf  = ax.contourf(X, Y, V_p, levels=40, cmap='RdBu_r',
                      norm=norm_c, alpha=0.85)
    ax.contour(X, Y, V_p, levels=20, colors='black',
               linewidths=0.5, alpha=0.6)
    sp = plt.Circle((0,0), R, color='silver', ec='black', lw=1.5, zorder=5)
    ax.add_patch(sp)
    ax.plot(4, 0, 'r*', ms=12, zorder=6)
    ax.set_xlim(-6,6); ax.set_ylim(-6,6); ax.set_aspect('equal')
    ax.set_title(f'Q = {Q_val*1e6:.0f} µC', fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, ls=':', lw=0.3, alpha=0.5)

    # Bottom row: surface charge density
    ax = axes2[1, col]
    ax.plot(np.degrees(th_c), sig_c * 1e6, lw=2,
            label=f'Q={Q_val*1e6:.0f}µC')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.fill_between(np.degrees(th_c), sig_c*1e6, 0,
                    where=(sig_c<0), alpha=0.25, color='blue')
    ax.fill_between(np.degrees(th_c), sig_c*1e6, 0,
                    where=(sig_c>0), alpha=0.25, color='red')
    ax.set_xlabel('θ (°)'); ax.set_ylabel('σ (µC/m²)')
    ax.set_title(f'σ  for Q = {Q_val*1e6:.0f} µC', fontweight='bold')
    ax.set_xticks([0,90,180,270,360])
    ax.grid(True, ls=':', lw=0.5); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('q2_plot2_vary_Q.png', dpi=150)
print("Saved: q2_plot2_vary_Q.png")


# ─────────────────────────────────────────────────────────────
# ── CASE 3: VARYING DISTANCE  (Q fixed = 10 µC) ──────────────
# ─────────────────────────────────────────────────────────────
d_values = [2.5, 4.0, 6.0, 10.0]    # must be > R=2
fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10))
fig3.suptitle('Effect of Varying Charge Distance  (Q = 10 µC, R = 2)',
              fontsize=13, fontweight='bold')

for col, d_val in enumerate(d_values):
    V_d, Ex_d, Ey_d, th_d, sig_d = compute_fields(Q_base, d=d_val)
    V_p = np.clip(V_d, -2e6, 2e6)
    norm_d = TwoSlopeNorm(vmin=-2e6, vcenter=0, vmax=2e6)

    # Top row: equipotentials
    ax = axes3[0, col]
    cf = ax.contourf(X, Y, V_p, levels=40, cmap='RdBu_r',
                     norm=norm_d, alpha=0.85)
    ax.contour(X, Y, V_p, levels=20, colors='black',
               linewidths=0.5, alpha=0.6)
    sp = plt.Circle((0,0), R, color='silver', ec='black', lw=1.5, zorder=5)
    ax.add_patch(sp)
    # Clamp marker inside domain
    x_marker = min(d_val, 5.9)
    ax.plot(x_marker, 0, 'r*', ms=12, zorder=6)
    ax.set_xlim(-6,6); ax.set_ylim(-6,6); ax.set_aspect('equal')
    ax.set_title(f'd = {d_val}', fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, ls=':', lw=0.3, alpha=0.5)

    # Bottom row: surface charge density
    ax = axes3[1, col]
    ax.plot(np.degrees(th_d), sig_d * 1e6, lw=2, label=f'd={d_val}')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.fill_between(np.degrees(th_d), sig_d*1e6, 0,
                    where=(sig_d<0), alpha=0.25, color='blue')
    ax.fill_between(np.degrees(th_d), sig_d*1e6, 0,
                    where=(sig_d>0), alpha=0.25, color='red')
    ax.set_xlabel('θ (°)'); ax.set_ylabel('σ (µC/m²)')
    ax.set_title(f'σ  for d = {d_val}', fontweight='bold')
    ax.set_xticks([0,90,180,270,360])
    ax.grid(True, ls=':', lw=0.5); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('q2_plot3_vary_d.png', dpi=150)
print("Saved: q2_plot3_vary_d.png")


# ─────────────────────────────────────────────────────────────
# ── CASE 4: COMBINED σ comparison curves ─────────────────────
# ─────────────────────────────────────────────────────────────
fig4, (ax_Q, ax_d) = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle('Induced Surface Charge Density Comparison', fontsize=13,
              fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, Q_val in enumerate(Q_values):
    _, _, _, th, sig = compute_fields(Q_val, d=4.0)
    ax_Q.plot(np.degrees(th), sig*1e6,
              label=f'Q = {Q_val*1e6:.0f} µC', color=colors[i], lw=2)
ax_Q.axhline(0, color='gray', lw=0.8, ls='--')
ax_Q.set_xlabel('θ  (degrees)', fontsize=11)
ax_Q.set_ylabel('σ  (µC/m²)', fontsize=11)
ax_Q.set_title('Varying Q  (d = 4 fixed)', fontweight='bold')
ax_Q.legend(); ax_Q.grid(True, ls=':', lw=0.5)
ax_Q.set_xticks([0,60,120,180,240,300,360])

for i, d_val in enumerate(d_values):
    _, _, _, th, sig = compute_fields(Q_base, d=d_val)
    ax_d.plot(np.degrees(th), sig*1e6,
              label=f'd = {d_val}', color=colors[i], lw=2)
ax_d.axhline(0, color='gray', lw=0.8, ls='--')
ax_d.set_xlabel('θ  (degrees)', fontsize=11)
ax_d.set_ylabel('σ  (µC/m²)', fontsize=11)
ax_d.set_title('Varying d  (Q = 10 µC fixed)', fontweight='bold')
ax_d.legend(); ax_d.grid(True, ls=':', lw=0.5)
ax_d.set_xticks([0,60,120,180,240,300,360])

plt.tight_layout()
plt.savefig('q2_plot4_sigma_comparison.png', dpi=150)
print("Saved: q2_plot4_sigma_comparison.png")

plt.show()
print("\nAll Q2 plots saved successfully.")
