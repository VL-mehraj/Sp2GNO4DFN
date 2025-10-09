import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


Frac1 = np.array([[-162.354, -500.000, 206.177],
                  [138.978, -30.511, -61.861],
                  [-229.987, 500.000, -10.007],
                  [-417.374, 500.000, 83.687],
                  [-500.000, 475.093, 131.227],
                  [-500.00, -446.814, 361.704],
                  [-367.607, -500.000, 308.800]])

Frac2 = np.array([[26.316, -500.000, -431.524],
                  [23.472, -445.963, -500.000],
                  [-21.371, 406.045, -500.000],
                  [-26.316, 500.000, -397.831],
                  [-26.316, 500.000, 392.163],
                  [-25.312, 480.928, 500.000],
                  [23.070, -438.334, 500.000],
                  [26.316, -500.000, 261.673]])

Frac3 = np.array([[500.000, -500.000, 275.000],
                  [500.000, 500.000, 25.000],
                  [382.626, 500.000, 83.687],
                  [-53.977, 368.388, 334.891],
                  [-18.109, -319.021, 488.810],
                  [432.393, -500.000, 308.804]])

Frac4 = np.array([[500.000, -500.000, -325.000],
                  [500.000, 200.000, -500.000],
                  [354.273, 491.453, -500.000],
                  [-53.977, 368.388, -265.109],
                  [-18.109, -319.021, -111.190],
                  [432.393, -500.000, -291.196]])

# Store all fractures and colors
fractures = [Frac1, Frac2, Frac3, Frac4]
colors = ['red', 'blue', 'green', 'orange']

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, (frac, color) in enumerate(zip(fractures, colors), 1):
    # Plot fracture vertices
    ax.scatter(frac[:, 0], frac[:, 1], frac[:, 2], s=30, label=f'Fracture {i}', color=color)
    
    # Optionally, make a polygon surface (if roughly planar)
    try:
        verts = [frac]
        poly = Poly3DCollection(verts, alpha=0.3, facecolor=color)
        ax.add_collection3d(poly)
    except Exception as e:
        print(f"Could not create surface for Frac{i}: {e}")

# Axis labels and legend
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Fracture Network Visualization')

ax.legend()
ax.view_init(elev=20, azim=60)  # adjust viewing angle
plt.tight_layout()
plt.show()
