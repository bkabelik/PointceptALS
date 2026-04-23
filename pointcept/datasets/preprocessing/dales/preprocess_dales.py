import os
import argparse
import numpy as np
import torch
import json
from plyfile import PlyData
from tqdm import tqdm
from scipy.interpolate import NearestNDInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed

def scan_single_file(file_path):
    """Worker function for scanning global stats."""
    try:
        plydata = PlyData.read(file_path)
        data = plydata.elements[0].data
        
        # Sample intensity (every 100th point)
        intensity_sample = np.array(data['intensity'][::100])
        
        # Local HAG estimate
        z_min = np.min(data['z'])
        z_max = np.max(data['z'])
        return intensity_sample, (z_max - z_min)
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
        return None, None

def get_global_stats(input_path, split="train", max_workers=32):
    print(f"--- Pass 1: Scanning {split} set for Global Statistics (Parallel) ---")
    folder = os.path.join(input_path, split)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.ply')]
    
    all_intensities = []
    max_hags = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(scan_single_file, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), desc="Scanning tiles"):
            ints, hag = future.result()
            if ints is not None:
                all_intensities.append(ints)
                max_hags.append(hag)
                
    global_intensities = np.concatenate(all_intensities)
    intensity_max = np.percentile(global_intensities, 99.9)
    suggested_z_scale = np.max(max_hags)
    
    print(f"\n[SCAN COMPLETED]")
    print(f"-> Detected Max Intensity (99.9th %): {intensity_max:.2f}")
    print(f"-> Detected Max Height Difference: {suggested_z_scale:.2f} meters")
    
    val = input("\nPress Enter to use defaults, 'n' to exit, or enter custom values int z (e.g., '64000 100'): ").strip()
    
    if val.lower() == 'n':
        exit()
    if val.lower() == 'y' or val == "":
        return intensity_max, suggested_z_scale
    
    parts = val.split()
    if len(parts) == 1:
        # If only one number is provided, assume it's the Z-scale
        return intensity_max, float(parts[0])
    else:
        # If two numbers are provided, first is intensity, second is Z-scale
        return float(parts[0]), float(parts[1])

def get_ground_elevation(points, grid_size=5.0):
    """Calculates DTM for the 500m tile."""
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    nx, ny = int((x_max - x_min) / grid_size) + 1, int((y_max - y_min) / grid_size) + 1
    
    grid = np.zeros((nx, ny)) + np.nan
    ix = ((points[:, 0] - x_min) / grid_size).astype(int)
    iy = ((points[:, 1] - y_min) / grid_size).astype(int)
    
    for i in range(nx):
        for j in range(ny):
            mask = (ix == i) & (iy == j)
            if np.any(mask):
                grid[i, j] = np.percentile(points[mask, 2], 5)
                
    valid = ~np.isnan(grid)
    coords = np.array(np.where(valid)).T
    itp = NearestNDInterpolator(coords, grid[valid])
    
    # Fill holes
    invalid_coords = np.array(np.where(~valid)).T
    if len(invalid_coords) > 0:
        grid[~valid] = itp(invalid_coords)
    return x_min, y_min, grid_size, grid

def process_single_file(file_name, input_split_path, output_split_path, int_max, z_scale, voxel_size):
    """Worker function to process one 500m PLY file into many 50m tiles."""
    try:
        plydata = PlyData.read(os.path.join(input_split_path, file_name))
        data = plydata.elements[0].data
        points = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
        intensity = (data['intensity'].astype(np.float32) / int_max).clip(0, 1).reshape(-1, 1)
        segments = data['sem_class'].astype(np.int64) - 1
        
        min_x, min_y, g_size, g_model = get_ground_elevation(points)
        tile_size = 50.0
        
        # Iterate through 10x10 grid (for a 500m tile)
        for x_s in np.arange(min_x, min_x + 500, tile_size):
            for y_s in np.arange(min_y, min_y + 500, tile_size):
                mask = (points[:, 0] >= x_s) & (points[:, 0] < x_s + tile_size) & \
                       (points[:, 1] >= y_s) & (points[:, 1] < y_s + tile_size)
                
                if np.sum(mask) < 500: 
                    continue
                
                c_p, c_i, c_s = points[mask], intensity[mask], segments[mask]
                
                # Voxelize (using bit-shifting or string keys can be slow, 
                # but unique with axis=0 is often faster for small batches)
                l_min = np.min(c_p, axis=0)
                g_c = np.floor((c_p - l_min) / voxel_size).astype(np.int64)
                _, idx = np.unique(g_c, axis=0, return_index=True)
                c_p, c_i, c_s = c_p[idx], c_i[idx], c_s[idx]
                
                # Normalization
                ix = ((c_p[:, 0] - min_x) / g_size).astype(int).clip(0, g_model.shape[0]-1)
                iy = ((c_p[:, 1] - min_y) / g_size).astype(int).clip(0, g_model.shape[1]-1)
                z_ref = g_model[ix, iy]
                
                norm_coords = np.zeros_like(c_p)
                norm_coords[:, 0] = (c_p[:, 0] - (x_s + 25.0)) / 25.0
                norm_coords[:, 1] = (c_p[:, 1] - (y_s + 25.0)) / 25.0
                norm_coords[:, 2] = ((c_p[:, 2] - z_ref) / (z_scale / 2.0)) - 1.0
                
                # Save
                tile_folder_name = f"{file_name[:-4]}_{int(x_s)}_{int(y_s)}"
                tile_path = os.path.join(output_split_path, tile_folder_name)
                os.makedirs(tile_path, exist_ok=True)
                
                np.save(os.path.join(tile_path, "coord.npy"), norm_coords.astype(np.float32))
                np.save(os.path.join(tile_path, "strength.npy"), c_i.astype(np.float32))
                np.save(os.path.join(tile_path, "segment.npy"), c_s.astype(np.int32))
        return True
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return False

def process_split(split, input_path, output_path, int_max, z_scale, voxel_size, max_workers=32):
    print(f"\n--- Pass 2: Processing {split} split (Parallel) ---")
    input_split_path = os.path.join(input_path, split)
    output_split_path = os.path.join(output_path, split)
    os.makedirs(output_split_path, exist_ok=True)
    
    files = [f for f in os.listdir(input_split_path) if f.endswith('.ply')]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file, f, input_split_path, output_split_path, int_max, z_scale, voxel_size) 
            for f in files
        ]
        for _ in tqdm(as_completed(futures), total=len(files), desc=f"Processing {split}"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/home/fractal01/PointSSM/data/DALESObjects")
    parser.add_argument("--output_path", default="/home/fractal01/PointceptALS/data/dales_scientific")
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--cores", type=int, default=32)
    args = parser.parse_args()
    
    # 1. Parallel Scan
    int_max, z_scale = get_global_stats(args.input_path, split="train", max_workers=args.cores)
    
    # 2. Parallel Process
    process_split("train", args.input_path, args.output_path, int_max, z_scale, args.voxel_size, max_workers=args.cores)
    process_split("test", args.input_path, args.output_path, int_max, z_scale, args.voxel_size, max_workers=args.cores)
    
    # 3. Meta data
    with open(os.path.join(args.output_path, "meta.json"), "w") as f:
        json.dump({"int_max": float(int_max), "z_scale": float(z_scale)}, f)
    print("\nDone.")
