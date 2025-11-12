import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt

# -------------------------
# Fracture vertex definitions
# -------------------------
Frac1 = np.array([
    [-162.354, -500.000, 206.177],
    [138.978,  -30.511, -61.861],
    [-229.987, 500.000, -10.007],
    [-417.374, 500.000,  83.687],
    [-500.000, 475.093, 131.227],
    [-500.000, -446.814, 361.704],
    [-367.607, -500.000, 308.800]
])

Frac2 = np.array([
    [26.316, -500.000, -431.524],
    [23.472, -445.963, -500.000],
    [-21.371, 406.045, -500.000],
    [-26.316, 500.000, -397.831],
    [-26.316, 500.000, 392.163],
    [-25.312, 480.928, 500.000],
    [23.070, -438.334, 500.000],
    [26.316, -500.000, 261.673]
])

Frac3 = np.array([
    [500.000, -500.000, 275.000],
    [500.000, 500.000, 25.000],
    [382.626, 500.000, 83.687],
    [-53.977, 368.388, 334.891],
    [-18.109, -319.021, 488.810],
    [432.393, -500.000, 308.804]
])

Frac4 = np.array([
    [500.000, -500.000, -325.000],
    [500.000, 200.000, -500.000],
    [354.273, 491.453, -500.000],
    [-53.977, 368.388, -265.109],
    [-18.109, -319.021, -111.190],
    [432.393, -500.000, -291.196]
])

fractures = [Frac1, Frac2, Frac3, Frac4]

#Fracture normals
frac_normals = np.array([[ 0.436435780471985, 0.218217890235992, 0.872871560943969],
                      [0.99861782933251, 0.0525588331227637, 0],
                      [0.436435780471985, 0.218217890235992, 0.872871560943969],
                      [ 0.436435780471985, 0.218217890235992, 0.872871560943969]])

permeability = np.array([8.333e-8,8.333e-8,8.333e-8,2.083e-8])
Apertures = np.array([1.0e-3,1.0e-3,1.0e-3,5.0e-4])

def points_on_plane(A, B, C, point, unknown):
    """
    Generate a grid of points on a plane Ax + By + Cz = D passing through 'point'.
    'unknown' specifies which coordinate to solve for:
        1 = x, 2 = y, 3 = z
    """
    x0, y0, z0 = point
    # Compute D from the point
    D = A*x0 + B*y0 + C*z0

    # Create grid
    Ls = np.arange(-500, 501, 25)
    Ms = np.arange(-500, 501, 25)
    L, M = np.meshgrid(Ls, Ms, indexing='ij')

    if unknown == 1:  # solve for x
        X = (D - B*L - C*M)/A
        points = np.vstack([X.ravel(), L.ravel(), M.ravel()]).T
    elif unknown == 2:  # solve for y
        Y = (D - A*L - C*M)/B
        points = np.vstack([L.ravel(), Y.ravel(), M.ravel()]).T
    elif unknown == 3:  # solve for z
        Z = (D - A*L - B*M)/C
        points = np.vstack([L.ravel(), M.ravel(), Z.ravel()]).T
    else:
        raise ValueError("Unknown must be 1, 2, or 3")
    
    return points


def points_inside_plane(points, polygon, normal):
    """
    Extract points lying inside a 3D polygon on a plane.
    
    points: Nx3 array of points to test
    polygon: Mx3 array of polygon vertices
    normal: 3-element plane normal vector
    """
    p0 = polygon[0]  # reference point on the plane
    n = normal / np.linalg.norm(normal)

    # Step 1: Create local 2D coordinate system on the plane
    if abs(n[0]) < 0.9:
        v1 = np.cross(n, [1, 0, 0])
    else:
        v1 = np.cross(n, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    v2 /= np.linalg.norm(v2)

    # Step 2: Project points and polygon vertices onto the 2D plane
    def project_to_2d(P):
        return np.array([[(p - p0) @ v1, (p - p0) @ v2] for p in P])

    points_2d = project_to_2d(points)
    polygon_2d = project_to_2d(polygon)

    # Step 3: Check which points are inside polygon
    poly_path = Path(polygon_2d)
    inside_mask = poly_path.contains_points(points_2d)
    return points[inside_mask]

def compute_linear_pressure(points, P_west, P_east, x_min, x_max):
    """
    Compute pressure at each point linearly varying from west (x_min) to east (x_max).
    """
    x = points[:, 0]  # extract x-coordinates
    P = P_west + (x - x_min) / (x_max - x_min) * (P_east - P_west)
    return P

def normalize_points(points, x_min=-500, x_max=500):
    """
    Normalize 3D points so that x, y, z coordinates range from -1 to 1.
    Assumes input domain ranges between x_min and x_max (default: -500 to 500).
    """
    # Apply normalization elementwise
    return 2 * (points - x_min) / (x_max - x_min) - 1


F1_points = points_on_plane(frac_normals[0,0], frac_normals[0,1], frac_normals[0,2], Frac1[0,:], 3)
F2_points = points_on_plane(frac_normals[1,0], frac_normals[1,1], frac_normals[1,2], Frac2[0,:], 1)
F3_points = points_on_plane(frac_normals[2,0], frac_normals[2,1], frac_normals[2,2], Frac3[0,:], 3)
F4_points = points_on_plane(frac_normals[3,0], frac_normals[3,1], frac_normals[3,2], Frac4[0,:], 3)

a = len(np.arange(-500, 501, 25))

# Extract points inside each fracture polygon
F1_inside = points_inside_plane(F1_points, Frac1, frac_normals[0])
F2_inside = points_inside_plane(F2_points, Frac2, frac_normals[1])
F3_inside = points_inside_plane(F3_points, Frac3, frac_normals[2])
F4_inside = points_inside_plane(F4_points, Frac4, frac_normals[3])

print(F1_inside.shape, F2_inside.shape, F3_inside.shape, F4_inside.shape)

F1_inside_normal = normalize_points(F1_inside)
F2_inside_normal = normalize_points(F2_inside)
F3_inside_normal = normalize_points(F3_inside)
F4_inside_normal = normalize_points(F4_inside)


Pressure_West = 1.001e6
Pressure_East = 1.000e6
x_min, x_max = -500, 500

F1_pressure = compute_linear_pressure(F1_inside, Pressure_West, Pressure_East, x_min, x_max)
F2_pressure = compute_linear_pressure(F2_inside, Pressure_West, Pressure_East, x_min, x_max)
F3_pressure = compute_linear_pressure(F3_inside, Pressure_West, Pressure_East, x_min, x_max)
F4_pressure = compute_linear_pressure(F4_inside, Pressure_West, Pressure_East, x_min, x_max)
print(F1_pressure.shape, F2_pressure.shape, F3_pressure.shape, F4_pressure.shape)

F1_permeability = np.full_like(F1_pressure,permeability[0])
F2_permeability = np.full_like(F2_pressure,permeability[1])
F3_permeability = np.full_like(F3_pressure,permeability[2])
F4_permeability = np.full_like(F4_pressure,permeability[3])
print(F1_permeability.shape, F2_permeability.shape, F3_permeability.shape, F4_permeability.shape)

F1_aperture = np.full_like(F1_pressure,Apertures[0])
F2_aperture = np.full_like(F2_pressure,Apertures[1])
F3_aperture = np.full_like(F3_pressure,Apertures[2])
F4_aperture = np.full_like(F4_pressure,Apertures[3])
print(F1_aperture.shape, F2_aperture.shape, F3_aperture.shape, F4_aperture.shape)


# %%


# Stack the 3D coordinates (x, y, z) for all points into a single array
all_points_3d = np.vstack([F1_inside, F2_inside, F3_inside, F4_inside])

all_points_normal = np.vstack([F1_inside_normal, F2_inside_normal, F3_inside_normal, F4_inside_normal])
all_pressure = np.hstack([F1_pressure, F2_pressure, F3_pressure, F4_pressure])
all_permeability = np.hstack([F1_permeability, F2_permeability, F3_permeability, F4_permeability])
all_aperture = np.hstack([F1_aperture, F2_aperture, F3_aperture, F4_aperture])

# You might want to combine the normalized points (Input Features)
# into a single array for a machine learning model:
# First, reshape the 1D arrays to be 2D columns (N, 1)
all_pressure_col = all_pressure.reshape(-1, 1)
all_permeability_col = all_permeability.reshape(-1, 1)
all_aperture_col = all_aperture.reshape(-1, 1)

# Combine the normalized 3D coordinates and the physical properties into one feature matrix
# The shape of this final feature array will be (Total_Points, 6)
all_input_features = np.hstack([
    all_points_normal,      # Normalized x, y, z coordinates (3 columns)
    all_permeability_col,   # Permeability (1 column)
    all_aperture_col,        # Aperture (1 column)
    all_pressure_col
])

# Combine the normalized points and concentration into a single array for training (features and target)
# The shape will be (Total_Points, 6) if you use just coordinates and pressure
# features_and_pressure = np.hstack([all_points_normal, all_pressure_col])

# Print the shape of the main stacked arrays for verification
print("-" * 40)
print(f"Total number of points: {all_points_3d.shape[0]}")
print(f"Shape of all_points_3d: {all_points_3d.shape}")
print(f"Shape of all_points_normal: {all_points_normal.shape}")
print(f"Shape of all_input_features (x_norm, y_norm, z_norm, K, a, p): {all_input_features.shape}")
print("-" * 40)


# %%


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

fractures_inside = [F1_inside, F2_inside, F3_inside, F4_inside]
fractures_colors = ['blue', 'green', 'orange', 'purple']

for i, pts_inside in enumerate(fractures_inside):
    ax.scatter(pts_inside[:, 0], pts_inside[:, 1], pts_inside[:, 2],
               c=fractures_colors[i], s=10, alpha=0.6, label=f'Frac{i+1} points inside')
    # Also plot the polygon edges
    frac = fractures[i]
    ax.plot(frac[:, 0], frac[:, 1], frac[:, 2], 'r-o', linewidth=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Points inside all fractures')
ax.legend()
plt.show()

plt.figure(figsize=(7, 4))
plt.scatter(F1_inside[:, 0], F1_pressure, s=10, label='Frac1')
plt.scatter(F2_inside[:, 0], F2_pressure, s=10, label='Frac2')
plt.scatter(F3_inside[:, 0], F3_pressure, s=10, label='Frac3')
plt.scatter(F4_inside[:, 0], F4_pressure, s=10, label='Frac4')
plt.xlabel('X coordinate (m)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure from East to west face across fractures')
plt.legend()
plt.show()


# %%

import pandas as pd
import numpy as np
from scipy.io import savemat

file_name = "U_sol.csv"

# Read the file to get the team names from the first row (header row index 0 in the file)
df_teams = pd.read_csv(file_name, header=0)

# Dictionary mapping the starting column index of the 'Time' data
# to the column index of the actual Team Name in the header row (index 0).
# The data is structured in blocks of 3 columns: [Time, Concentration, Unnamed_Empty]
column_map = {
    0: 1,   # BGR data starts at index 0, name at index 1
    3: 4,   # KAERI data starts at index 3, name at index 4
    6: 7,   # NARI data starts at index 6, name at index 7
    9: 10,  # NWMO data starts at index 9, name at index 10
    12: 13, # SSM-UU data starts at index 12, name at index 13
    15: 16, # SURAO data starts at index 15, name at index 16
    18: 19  # CNSC data starts at index 18, name at index 19
}
df_data = pd.read_csv(file_name, header=6)
all_arrays = {}

for data_start_col, team_name_col in column_map.items():
    team_name = df_teams.columns[team_name_col]
    df_pair = df_data.iloc[:, data_start_col:data_start_col+2].copy()

    for col in df_pair.columns:
        df_pair[col] = pd.to_numeric(df_pair[col], errors='coerce')

    df_numeric = df_pair.dropna()
    numpy_array = df_numeric.to_numpy()
    all_arrays[team_name] = numpy_array

# Assuming 'all_arrays' is the dictionary containing the team data
# Example structure: all_arrays = {'BGR': np.array([[t1, c1], [t2, c2], ...]), ...}

# Initialize the maximum time found so far
max_time_found = 0.0
team_with_max_time = ""

print("Checking maximum time for each simulation...")
print("-" * 40)

# Iterate through the dictionary items (team name and their data array)
for team, data_array in all_arrays.items():
    if data_array.size > 0:
        # The time data is in the first column (index 0)
        current_max_time = np.max(data_array[:, 0])

        print(f"Max time for {team}: {current_max_time:.4f} years")

        # Check if this is the global maximum
        if current_max_time > max_time_found:
            max_time_found = current_max_time
            team_with_max_time = team
            
print("-" * 40)
print(f"The **Maximum Time** among all cases is **{max_time_found:.4f} years**.")
print(f"This maximum time was recorded in the simulation data for the **{team_with_max_time}** team.")

Times = np.linspace(0, max_time_found,20) 
# print('Times',Times)

nearest_time_concentration_data = {}

# Iterate through each team's data
for team, data_array in all_arrays.items():
    if data_array.size > 0:
        team_times = data_array[:, 0]
        team_concentrations = data_array[:, 1]
        
        # Initialize an array to store the resulting (Time, Concentration) pairs
        result_array = np.zeros((len(Times), 2))
        
        # Iterate through each target time in the 'Times' array
        for i, target_time in enumerate(Times):
            # 1. Calculate the absolute difference between the target time and all recorded team times
            time_diff = np.abs(team_times - target_time)
            
            # 2. Find the index of the minimum difference (the nearest time point)
            nearest_index = np.argmin(time_diff)
            
            # 3. Retrieve the recorded time and concentration at that nearest index
            nearest_recorded_time = team_times[nearest_index]
            nearest_recorded_concentration = team_concentrations[nearest_index]
            
            # 4. Store the pair in the result array
            result_array[i, 0] = nearest_recorded_time
            result_array[i, 1] = nearest_recorded_concentration
        
        # Store the final NumPy array for the team
        nearest_time_concentration_data[team] = result_array

Teams = list(nearest_time_concentration_data.keys())
output_features_at_times_for_teams = np.zeros((len(Teams),all_points_normal.shape[0],1,len(Times)))
input_features_at_times = np.zeros((all_input_features.shape[0],6,len(Times)))
input_features_for_teams = np.zeros((len(Teams),all_points_normal.shape[0],6, len(Times)))


for T in range(len(Teams)):
    for i in range(len(Times)):
        concentration = np.full_like(all_pressure_col,nearest_time_concentration_data[Teams[T]][i,1])
        output_features_at_times_for_teams[T,:,:,i] = concentration
        input_features_at_times[:,:, i] = all_input_features
    input_features_for_teams[T,:,:,:] = input_features_at_times[:,:,:]
    
print('input features with times',input_features_at_times.shape)
print('input features for teams',input_features_for_teams.shape)
print('output features with times',output_features_at_times_for_teams.shape)

mat_data = {}
mat_data['coeff'] = input_features_for_teams
mat_data['sol'] = output_features_at_times_for_teams

output_mat_filename = "dfn_4frac_prob.mat"
savemat(output_mat_filename, mat_data)

print(f"\nSuccessfully saved data to {output_mat_filename}")
print("The file contains two variables:")
print(f"  - 'coeff' (Input X): {input_features_for_teams.shape}")
print(f"  - 'sol' (Output Y): {output_features_at_times_for_teams.shape}")