import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
ax.set_xlim(-5.0, 9.2)
ax.set_ylim(-0.6, 5.5)
ax.set_aspect('equal')
ax.axis('off')

def draw_poly(pts, facecolor='none', edgecolor='black', lw=1.5, alpha=1.0, ls='-', hatch=None, zorder=2):
    poly = patches.Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                           linewidth=lw, alpha=alpha, linestyle=ls, hatch=hatch, zorder=zorder)
    ax.add_patch(poly)

# 1. Camera Planes & Windows
cam3 = [[-4.3, 0.9], [-2.2, 0.55], [-2.0, 2.05], [-4.1, 2.4]]
win3 = [[-3.35, 1.35], [-3.00, 1.29], [-2.95, 1.58], [-3.30, 1.64]]

cam2 = [[-0.8, 0.0], [1.4, 0.0], [1.4, 1.45], [-0.8, 1.45]]
win2 = [[0.2, 0.55], [0.5, 0.55], [0.5, 0.82], [0.2, 0.82]]

cam1 = [[2.2, 0.1], [4.3, 0.35], [4.5, 1.75], [2.4, 1.5]]
win1 = [[3.25, 0.75], [3.55, 0.78], [3.60, 1.05], [3.30, 1.02]]

for cam, win in [(cam1, win1), (cam2, win2)]:
    draw_poly(cam, facecolor='white', edgecolor='#222222', lw=1.5, zorder=3)
    draw_poly(win, facecolor='#d9d9d9', edgecolor='#222222', lw=1.2, zorder=4)

draw_poly(cam3, facecolor='white', edgecolor='#222222', lw=1.5, zorder=3)
draw_poly(win3, facecolor='#d62728', edgecolor='#8c0800', lw=1.2, zorder=4)

# 2. 3D Volume F_k Vertices
F = np.array([[0.0, 3.2], [1.3, 3.2], [1.3, 4.4], [0.0, 4.4]])
shift = np.array([0.55, 0.42])
B = F + shift

# Rays from Cam 1 and Cam 2
for w_pt, f_pt in zip(win1, [F[1], B[1], B[2], F[3]]):
    ax.plot([w_pt[0], f_pt[0]], [w_pt[1], f_pt[1]], color='#888888', lw=0.9, ls='--', zorder=1)
for w_pt, f_pt in zip(win2, [F[0], F[1], B[2], B[3]]):
    ax.plot([w_pt[0], f_pt[0]], [w_pt[1], f_pt[1]], color='#888888', lw=0.9, ls='--', zorder=1)

# 3. Red Slice S_k inside F_k
SF = np.array([[0.0, 3.50], [0.75, 3.45], [0.75, 3.98], [0.0, 4.02]])
SB = np.array([[0.45, 3.82], [1.15, 3.78], [1.15, 4.25], [0.45, 4.28]])

draw_poly(SF, facecolor='#ff4d4d', edgecolor='#990000', lw=1.3, alpha=0.55, hatch='///', zorder=5)
draw_poly([SF[3], SF[2], SB[2], SB[3]], facecolor='#e60000', edgecolor='#990000', lw=1.3, alpha=0.45, hatch='///', zorder=5)
draw_poly([SF[1], SB[1], SB[2], SF[2]], facecolor='#cc0000', edgecolor='#990000', lw=1.3, alpha=0.40, zorder=5)

# Rays from Cam 3 (Red Beam)
for w_pt, s_pt in zip(win3, SF):
    ax.plot([w_pt[0], s_pt[0]], [w_pt[1], s_pt[1]], color='#b30000', lw=1.4, zorder=4)

# 4. Draw F_k Wireframe
ax.plot([F[0,0], B[0,0]], [F[0,1], B[0,1]], color='gray', ls='--', lw=1.2, zorder=2)
ax.plot([B[0,0], B[1,0]], [B[0,1], B[1,1]], color='gray', ls='--', lw=1.2, zorder=2)
ax.plot([B[0,0], B[3,0]], [B[0,1], B[3,1]], color='gray', ls='--', lw=1.2, zorder=2)

draw_poly(F, facecolor='none', edgecolor='black', lw=2.0, zorder=6)
draw_poly([F[3], B[3], B[2], F[2]], facecolor='none', edgecolor='black', lw=2.0, zorder=6)
draw_poly([F[1], B[1], B[2], F[2]], facecolor='none', edgecolor='black', lw=2.0, zorder=6)

# Landmark center point k
ax.scatter([0.85], [3.95], color='black', s=35, zorder=7)
ax.text(0.95, 4.95, r"$k(\hat{x}, \hat{y}, \hat{z})$", fontsize=12, ha='center')

# Labels
ax.text(-3.2, 0.25, r"$\theta_3 = (R_3, t_3)$", fontsize=12, ha='center')
ax.text(0.3, -0.4, r"$\theta_2 = (R_2, t_2)$", fontsize=12, ha='center')
ax.text(3.3, -0.3, r"$\theta_1 = (R_1, t_1)$", fontsize=12, ha='center')

ax.text(-2.25, 1.8, "3", fontsize=11, fontweight='bold')
ax.text(1.15, 1.15, "2", fontsize=11, fontweight='bold')
ax.text(4.65, 1.55, "1", fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor='black'))

# Explanatory Box
note_text = (
    "For example, the red volume\n"
    r"is $S_k = \mathcal{F}_k \cap \mathcal{K}((R_3, t_3), W)$." + "\n\n"
    r"$\mathcal{F}_k$ is the original volume" + "\n"
    "obtained by image 1 and 2."
)
ax.text(4.8, 3.4, note_text, fontsize=11.5, va='center', ha='left',
        bbox=dict(boxstyle='square,pad=0.7', facecolor='#fafafa', edgecolor='black', lw=1.3))

plt.tight_layout()
plt.savefig("backprojected_slice_diagram.pdf")
plt.savefig("backprojected_slice_diagram.png", dpi=300)
plt.show()