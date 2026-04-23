import os
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

def get_ground_elevation(points, labels, grid_size=2.0):
    """
    Vectorized High-Accuracy DTM logic.
    Identical to the preprocessing script for consistency.
    """
    ground_mask = (labels == 1)
    
    if np.sum(ground_mask) < 100:
        print("Warning: Low ground point count. Using fallback.")
        target_points = points
        actual_grid_size = 30.0 
    else:
        target_points = points[ground_mask]
        actual_grid_size = grid_size

    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    
    nx = int((x_max - x_min) / actual_grid_size) + 1
    ny = int((y_max - y_min) / actual_grid_size) + 1
    
    # --- Vectorized Binning (The Speed Hack) ---
    ix = ((target_points[:, 0] - x_min) / actual_grid_size).astype(int).clip(0, nx-1)
    iy = ((target_points[:, 1] - y_min) / actual_grid_size).astype(int).clip(0, ny-1)
    flat_indices = ix * ny + iy
    
    sort_idx = np.argsort(flat_indices)
    sorted_indices = flat_indices[sort_idx]
    sorted_z = target_points[sort_idx, 2]
    
    diffs = np.diff(sorted_indices)
    split_indices = np.where(diffs > 0)[0] + 1
    z_groups = np.split(sorted_z, split_indices)
    unique_bins = sorted_indices[np.append([0], split_indices)]
    
    grid = np.full((nx, ny), np.nan)
    for bin_idx, group in zip(unique_bins, z_groups):
        if len(group) > 0:
            grid[bin_idx // ny, bin_idx % ny] = np.percentile(group, 5)
                
    valid_mask = ~np.isnan(grid)
    if not np.any(valid_mask):
        return x_min, y_min, actual_grid_size, np.full((nx, ny), np.min(points[:, 2]))
        
    coords_valid = np.array(np.where(valid_mask)).T
    values_valid = grid[valid_mask]
    
    # Linear Interpolation (Smooth Slopes)
    itp_linear = LinearNDInterpolator(coords_valid, values_valid)
    all_coords = np.array(np.where(~valid_mask)).T
    grid[~valid_mask] = itp_linear(all_coords)
    
    # Nearest Fallback for edges
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
    
    points_sub = points[::10]
    labels_sub = labels[::10]
    
    print("Calculating Vectorized Linear DTM...")
    min_x, min_y, g_size, g_model = get_ground_elevation(points, labels)
    
    nx, ny = g_model.shape
    dtm_points = []
    lines = []
    
    # We still loop here for visualization geometry, but it's much faster 
    # because the heavy math (binning/percentile) is already done.
    for i in range(nx):
        for j in range(ny):
            x = min_x + i * g_size
            y = min_y + j * g_size
            z = g_model[i, j]
            dtm_points.append([x, y, z])
            if i < nx - 1: lines.append([i * ny + j, (i + 1) * ny + j])
            if j < ny - 1: lines.append([i * ny + j, i * ny + (j + 1)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sub)
    colors = np.zeros_like(points_sub)
    colors[:] = [0.3, 0.3, 0.3] # Gray
    colors[labels_sub == 1] = [0.0, 1.0, 0.0] # Ground Green
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(dtm_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.paint_uniform_color([1, 0, 0]) # Red
    
    print("Viewer Ready.")
    o3d.visualization.draw_geometries([pcd, line_set], 
                                      window_name="Fast Class-Aware DTM",
                                      width=1280, height=720)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_file", help="Path to raw DALES .ply file")
    args = parser.parse_args()
    if os.path.exists(args.ply_file):
        visualize_dtm(args.ply_file)
    else:
        print("File not found.")
