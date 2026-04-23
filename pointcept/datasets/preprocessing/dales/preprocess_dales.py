import os
import argparse
import numpy as np
import json
from plyfile import PlyData
from tqdm import tqdm
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def scan_single_file(file_path):
    try:
        plydata = PlyData.read(file_path)
        data = plydata.elements[0].data
        intensity_sample = np.array(data['intensity'][::100])
        z_min, z_max = np.min(data['z']), np.max(data['z'])
        return intensity_sample, (z_max - z_min)
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
        return None, None

def get_global_stats(input_path, split="train", max_workers=32):
    print(f"--- Pass 1: Scanning {split} set for Global Statistics ---")
    folder = os.path.join(input_path, split)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.ply')]
    
    all_intensities, max_hags = [], []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(scan_single_file, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), desc="Scanning tiles"):
            ints, hag = future.result()
            if ints is not None:
                all_intensities.append(ints)
                max_hags.append(hag)
                
    intensity_max = np.percentile(np.concatenate(all_intensities), 99.9)
    suggested_z_scale = np.percentile(max_hags, 95) 
    
    print(f"\n[SCAN COMPLETED]")
    print(f"-> Detected Max Intensity (99.9th %): {intensity_max:.2f}")
    print(f"-> Suggested Z-Scale (95th %): {suggested_z_scale:.2f} meters")
    
    print("\nFormat: [IntensityMax] [Z-Scale]")
    print(f"Example: '61430 80' or just press Enter for defaults ({intensity_max:.0f} {suggested_z_scale:.1f})")
    val = input(">> ").strip()
    
    if not val:
        return intensity_max, suggested_z_scale
    parts = val.split()
    if len(parts) == 1:
        return intensity_max, float(parts[0])
    return float(parts[0]), float(parts[1])

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
    
    # --- Vectorized Binning ---
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
    
    itp_linear = LinearNDInterpolator(coords_valid, values_valid)
    all_coords = np.array(np.where(~valid_mask)).T
    grid[~valid_mask] = itp_linear(all_coords)
    
    if np.any(np.isnan(grid)):
        itp_nearest = NearestNDInterpolator(coords_valid, values_valid)
        nan_mask = np.isnan(grid)
        grid[nan_mask] = itp_nearest(np.array(np.where(nan_mask)).T)
        
    return x_min, y_min, actual_grid_size, grid

def process_single_file(file_name, input_split_path, output_split_path, int_max, z_scale, voxel_size):
    try:
        plydata = PlyData.read(os.path.join(input_split_path, file_name))
        data = plydata.elements[0].data
        points = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
        intensity = (data['intensity'].astype(np.float32) / int_max).clip(0, 1).reshape(-1, 1)
        segments = data['sem_class'].astype(np.int64) - 1
        raw_segments = data['sem_class'].astype(np.int64) 

        min_x, min_y, g_size, g_model = get_ground_elevation(points, raw_segments)  
        tile_size = 50.0
        
        for x_s in np.arange(min_x, min_x + 500, tile_size):
            for y_s in np.arange(min_y, min_y + 500, tile_size):
                mask = (points[:, 0] >= x_s) & (points[:, 0] < x_s + tile_size) & \
                       (points[:, 1] >= y_s) & (points[:, 1] < y_s + tile_size)
                if np.sum(mask) < 500: continue
                
                c_p, c_i, c_s = points[mask], intensity[mask], segments[mask]
                
                # Voxelize
                g_c = np.floor(c_p / voxel_size).astype(np.int64)
                _, idx = np.unique(g_c, axis=0, return_index=True)
                c_p, c_i, c_s = c_p[idx], c_i[idx], c_s[idx]
                
                # Normalization
                ix = ((c_p[:, 0] - min_x) / g_size).astype(int).clip(0, g_model.shape[0]-1)
                iy = ((c_p[:, 1] - min_y) / g_size).astype(int).clip(0, g_model.shape[1]-1)
                z_ref = g_model[ix, iy]
                
                hag = (c_p[:, 2] - z_ref).clip(0, z_scale)
                norm_coords = np.zeros_like(c_p)
                norm_coords[:, 0] = c_p[:, 0] - (x_s + 25.0)  # Centered meters
                norm_coords[:, 1] = c_p[:, 1] - (y_s + 25.0)  # Centered meters
                norm_coords[:, 2] = hag                       # Height in meters
                
                tile_folder = os.path.join(output_split_path, f"{file_name[:-4]}_{int(x_s)}_{int(y_s)}")
                os.makedirs(tile_folder, exist_ok=True)
                np.save(os.path.join(tile_folder, "coord.npy"), norm_coords.astype(np.float32))
                np.save(os.path.join(tile_folder, "strength.npy"), c_i.astype(np.float32))
                np.save(os.path.join(tile_folder, "segment.npy"), c_s.astype(np.int32))
        return True
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return False

def process_split(split, input_path, output_path, int_max, z_scale, voxel_size, cores):
    print(f"\n--- Pass 2: Processing {split} split (Voxel: {voxel_size}m) ---")
    in_p, out_p = os.path.join(input_path, split), os.path.join(output_path, split)
    os.makedirs(out_p, exist_ok=True)
    files = [f for f in os.listdir(in_p) if f.endswith('.ply')]
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = [executor.submit(process_single_file, f, in_p, out_p, int_max, z_scale, voxel_size) for f in files]
        for _ in tqdm(as_completed(futures), total=len(files), desc=f"Processing {split}"): pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/home/fractal01/PointSSM/data/DALESObjects")
    parser.add_argument("--output_path", default="/home/fractal01/PointceptALS/data/DALESObjects_training_data")
    parser.add_argument("--voxel_size", type=float, default=0.15)
    parser.add_argument("--cores", type=int, default=32)
    args = parser.parse_args()
    
    int_max, z_scale = get_global_stats(args.input_path, "train", args.cores)
    process_split("train", args.input_path, args.output_path, int_max, z_scale, args.voxel_size, args.cores)
    process_split("test", args.input_path, args.output_path, int_max, z_scale, args.voxel_size, args.cores)
    
    with open(os.path.join(args.output_path, "meta.json"), "w") as f:
        json.dump({"int_max": float(int_max), "z_scale": float(z_scale), "voxel_size": args.voxel_size}, f)
