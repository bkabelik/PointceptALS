import os
import argparse
import numpy as np
import torch
import json
from plyfile import PlyData
from tqdm import tqdm
from scipy.interpolate import NearestNDInterpolator

def get_global_stats(input_path, split="train"):
    """Scans the training PLY files for global intensity and height scales."""
    print(f"--- Pass 1: Scanning {split} set for Global Statistics ---")
    files = [f for f in os.listdir(os.path.join(input_path, split)) if f.endswith('.ply')]
    
    all_intensities = []
    max_hags = []
    
    for f in tqdm(files, desc="Scanning tiles"):
        plydata = PlyData.read(os.path.join(input_path, split, f))
        data = plydata.elements[0].data
        
        # 1. Intensity Sample
        all_intensities.append(np.array(data['intensity'][::100]))
        
        # 2. HAG estimate
        z_min = np.min(data['z'])
        z_max = np.max(data['z'])
        max_hags.append(z_max - z_min)
        
    global_intensities = np.concatenate(all_intensities)
    intensity_max = np.percentile(global_intensities, 99.9)
    suggested_z_scale = np.max(max_hags)
    
    print(f"\n[SCAN COMPLETED]")
    print(f"-> Detected Max Intensity (99.9th %): {intensity_max:.2f}")
    print(f"-> Detected Max Height Difference: {suggested_z_scale:.2f} meters")
    
    val = input("\nUse these values? (y/n) or enter custom Z-scale: ")
    if val.lower() == 'y':
        return intensity_max, suggested_z_scale
    elif val.lower() == 'n':
        exit()
    else:
        return intensity_max, float(val)

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
    grid[~valid] = itp(np.array(np.where(~valid)).T)
    return x_min, y_min, grid_size, grid

def process_split(split, input_path, output_path, int_max, z_scale, voxel_size):
    print(f"\n--- Pass 2: Processing {split} split ---")
    files = [f for f in os.listdir(os.path.join(input_path, split)) if f.endswith('.ply')]
    
    for file_name in files:
        plydata = PlyData.read(os.path.join(input_path, split, file_name))
        data = plydata.elements[0].data
        points = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
        intensity = (data['intensity'].astype(np.float32) / int_max).clip(0, 1).reshape(-1, 1)
        segments = data['sem_class'].astype(np.int64) - 1 # Assuming 1-8 -> 0-7
        
        min_x, min_y, g_size, g_model = get_ground_elevation(points)
        tile_size = 50.0
        
        for x_s in tqdm(np.arange(min_x, min_x + 500, tile_size), desc=f"Tiling {file_name}"):
            for y_s in np.arange(min_y, min_y + 500, tile_size):
                mask = (points[:, 0] >= x_s) & (points[:, 0] < x_s + tile_size) & \
                       (points[:, 1] >= y_s) & (points[:, 1] < y_s + tile_size)
                if np.sum(mask) < 500: continue
                
                c_p, c_i, c_s = points[mask], intensity[mask], segments[mask]
                
                # Voxelize
                l_min = np.min(c_p, axis=0)
                g_c = np.floor((c_p - l_min) / voxel_size).astype(np.int64)
                keys = g_c[:, 0] * 10**12 + g_c[:, 1] * 10**6 + g_c[:, 2]
                _, idx = np.unique(keys, return_index=True)
                c_p, c_i, c_s = c_p[idx], c_i[idx], c_s[idx]
                
                # Normalization
                ix = ((c_p[:, 0] - min_x) / g_size).astype(int).clip(0, g_model.shape[0]-1)
                iy = ((c_p[:, 1] - min_y) / g_size).astype(int).clip(0, g_model.shape[1]-1)
                z_ref = g_model[ix, iy]
                
                norm_coords = np.zeros_like(c_p)
                norm_coords[:, 0] = (c_p[:, 0] - (x_s + 25.0)) / 25.0
                norm_coords[:, 1] = (c_p[:, 1] - (y_s + 25.0)) / 25.0
                norm_coords[:, 2] = ((c_p[:, 2] - z_ref) / (z_scale / 2.0)) - 1.0
                
                # --- NEW SAVING LOGIC FOR DEFAULT DATASET ---
                # Create a folder for this specific 50m tile
                tile_folder_name = f"{file_name[:-4]}_{int(x_s)}_{int(y_s)}"
                tile_path = os.path.join(output_path, split, tile_folder_name)
                os.makedirs(tile_path, exist_ok=True)
                
                # Save separate .npy files inside that folder
                np.save(os.path.join(tile_path, "coord.npy"), norm_coords.astype(np.float32))
                np.save(os.path.join(tile_path, "strength.npy"), c_i.astype(np.float32))
                np.save(os.path.join(tile_path, "segment.npy"), c_s.astype(np.int32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/home/fractal01/PointSSM/data/DALESObjects")
    parser.add_argument("--output_path", default="/home/fractal01/PointceptALS/data/dales_scientific")
    parser.add_argument("--voxel_size", type=float, default=0.1)
    args = parser.parse_args()
    
    int_max, z_scale = get_global_stats(args.input_path, split="train")
    process_split("train", args.input_path, args.output_path, int_max, z_scale, args.voxel_size)
    process_split("test", args.input_path, args.output_path, int_max, z_scale, args.voxel_size)
    
    with open(os.path.join(args.output_path, "meta.json"), "w") as f:
        json.dump({"int_max": float(int_max), "z_scale": float(z_scale)}, f)
