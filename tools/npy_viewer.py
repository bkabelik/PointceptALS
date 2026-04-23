# pip install open3d
import os
import numpy as np
import open3d as o3d
import glob

def visualize_tiles(base_path, split="train", num_tiles=1):
    """
    base_path: path to 'dales_scientific'
    num_tiles: how many random tiles to load together
    """
    tile_folders = glob.glob(os.path.join(base_path, split, "*/"))
    if not tile_folders:
        print("No tiles found! Check your path.")
        return

    # Select tiles
    selected_indices = np.random.choice(len(tile_folders), min(num_tiles, len(tile_folders)), replace=False)
    
    geometries = []
    
    # Color map for DALES (8 classes)
    colors = np.array([
        [0.5, 0.5, 0.5], # 0: Ground (Gray)
        [0.0, 1.0, 0.0], # 1: Veg (Green)
        [1.0, 0.0, 0.0], # 2: Cars (Red)
        [1.0, 0.5, 0.0], # 3: Trucks (Orange)
        [1.0, 1.0, 0.0], # 4: Powerlines (Yellow)
        [0.0, 0.0, 1.0], # 5: Fences (Blue)
        [0.0, 1.0, 1.0], # 6: Poles (Cyan)
        [1.0, 0.0, 1.0], # 7: Buildings (Pink)
    ])

    for idx in selected_indices:
        folder = tile_folders[idx]
        print(f"Loading tile: {os.path.basename(os.path.normpath(folder))}")
        
        # Load the attributes
        coords = np.load(os.path.join(folder, "coord.npy"))
        strength = np.load(os.path.join(folder, "strength.npy"))
        segments = np.load(os.path.join(folder, "segment.npy")).flatten()

        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # Apply colors based on segments
        # Handle ignore_index (-1) by making them black
        valid_mask = (segments >= 0) & (segments < 8)
        pt_colors = np.zeros((len(segments), 3))
        pt_colors[valid_mask] = colors[segments[valid_mask]]
        pcd.colors = o3d.utility.Vector3dVector(pt_colors)
        
        geometries.append(pcd)

    # Launch Viewer
    print("Launching Viewer... Close the window to exit.")
    o3d.visualization.draw_geometries(geometries, 
                                      window_name="PointceptALS Scientific Data Viewer",
                                      width=1280, height=720)

if __name__ == "__main__":
    # Update this path to your output directory
    DATA_PATH = "/home/fractal01/PointceptALS/data/dales_scientific"
    visualize_tiles(DATA_PATH, split="train", num_tiles=2)
