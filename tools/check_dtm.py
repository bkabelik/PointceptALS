import os
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.interpolate import NearestNDInterpolator

from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

def get_ground_elevation(points, labels, grid_size=2.0):
    ground_mask = (labels == 1)
    
    if np.sum(ground_mask) < 100:
        target_points = points
        actual_grid_size = 30.0 
    else:
        target_points = points[ground_mask]
        actual_grid_size = grid_size

    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    
    nx = int((x_max - x_min) / actual_grid_size) + 1
    ny = int((y_max - y_min) / actual_grid_size) + 1
    
    grid = np.zeros((nx, ny)) + np.nan
    ix = ((target_points[:, 0] - x_min) / actual_grid_size).astype(int)
    iy = ((target_points[:, 1] - y_min) / actual_grid_size).astype(int)
    
    for i in range(nx):
        for j in range(ny):
            mask = (ix == i) & (iy == j)
            if np.any(mask):
                # 5th percentile is more stable than 2nd for small grids
                grid[i, j] = np.percentile(target_points[mask, 2], 5)
                
    valid_mask = ~np.isnan(grid)
    if not np.any(valid_mask):
        return x_min, y_min, actual_grid_size, np.full((nx, ny), np.min(points[:, 2]))
        
    coords_valid = np.array(np.where(valid_mask)).T
    values_valid = grid[valid_mask]
    
    # 1. Fill holes with Linear (Smooth Ramps under buildings)
    itp_linear = LinearNDInterpolator(coords_valid, values_valid)
    all_coords = np.array(np.where(~valid_mask)).T
    grid[~valid_mask] = itp_linear(all_coords)
    
    # 2. Fill remaining NaNs with Nearest (Only for edges where Linear fails)
    if np.any(np.isnan(grid)):
        itp_nearest = NearestNDInterpolator(coords_valid, values_valid)
        nan_mask = np.isnan(grid)
        grid[nan_mask] = itp_nearest(np.array(np.where(nan_mask)).T)
        
    return x_min, y_min, actual_grid_size, grid

def visualize_dtm(ply_path):
    print(f"Loading {ply_path}...")
    plydata = PlyData.read(ply_path)
    data = plydata.elements[0].data
    
    points = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
    labels = data['sem_class'].astype(np.int64)
    
    # Subsample points for speed in the viewer (1 in 10)
    points_sub = points[::10]
    labels_sub = labels[::10]
    
    print("Calculating Class-Aware DTM...")
    min_x, min_y, g_size, g_model = get_ground_elevation(points, labels)
    
    # Create grid for DTM visualization
    nx, ny = g_model.shape
    dtm_points = []
    lines = []
    
    for i in range(nx):
        for j in range(ny):
            x = min_x + i * g_size
            y = min_y + j * g_size
            z = g_model[i, j]
            dtm_points.append([x, y, z])
            
            # Grid connectivity
            if i < nx - 1:
                lines.append([i * ny + j, (i + 1) * ny + j])
            if j < ny - 1:
                lines.append([i * ny + j, i * ny + (j + 1)])

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sub)
    
    # Optional: Color points by class so we can see what's ground
    # Ground = Green, Others = Gray
    colors = np.zeros_like(points_sub)
    colors[:] = [0.4, 0.4, 0.4] # Default Gray
    colors[labels_sub == 1] = [0.0, 0.8, 0.0] # Ground Green
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create DTM Grid (Red)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(dtm_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.paint_uniform_color([1, 0, 0]) # Red
    
    print("\n[VIEWER CONTROLS]")
    print("-> RED GRID: Calculated DTM (Ground Model)")
    print("-> GREEN POINTS: Class 1 (Ground)")
    print("-> GRAY POINTS: Everything else (Buildings, Trees, etc.)")
    
    o3d.visualization.draw_geometries([pcd, line_set], 
                                      window_name="Class-Aware DTM Check",
                                      width=1280, height=720)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_file", help="Path to raw DALES .ply file")
    args = parser.parse_args()
    
    if not os.path.exists(args.ply_file):
        print(f"Error: {args.ply_file} not found.")
    else:
        visualize_dtm(args.ply_file)
