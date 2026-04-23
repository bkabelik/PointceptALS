import os
import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pathlib import Path

class TileViewer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        
        # Color map and labels for DALES
        self.class_info = [
            {"label": "Ground", "color": [0.5, 0.5, 0.5]},    # 0
            {"label": "Vegetation", "color": [0.0, 1.0, 0.0]}, # 1
            {"label": "Cars", "color": [1.0, 0.0, 0.0]},       # 2
            {"label": "Trucks", "color": [1.0, 0.5, 0.0]},     # 3
            {"label": "Powerlines", "color": [1.0, 1.0, 0.0]}, # 4
            {"label": "Fences", "color": [0.0, 0.0, 1.0]},     # 5
            {"label": "Poles", "color": [0.0, 1.0, 1.0]},      # 6
            {"label": "Buildings", "color": [1.0, 0.0, 1.0]},  # 7
        ]
        self.colors = np.array([c["color"] for c in self.class_info])

        # Initialize GUI
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("DALES Tile Explorer", 1280, 720)
        
        # Theme/Styling
        em = self.window.theme.font_size
        margin = gui.Margins(int(0.5 * em))
        
        # --- UI LAYOUT ---
        # The panel on the left
        self.panel = gui.Vert(0, margin)
        
        self.panel.add_child(gui.Label("Select a Tile:"))
        self.tile_list = gui.ListView()
        
        print(f"Scanning for tiles in {self.base_path}...")
        self.all_tiles = sorted([p.parent for p in self.base_path.glob("**/coord.npy")])
        
        if not self.all_tiles:
            self.tile_list.set_items(["No tiles found!"])
        else:
            self.tile_list.set_items([str(p.relative_to(self.base_path)) for p in self.all_tiles])
        
        self.tile_list.set_on_selection_changed(self._on_selection_changed)
        self.panel.add_child(self.tile_list)

        # Legend
        self.panel.add_fixed(em)
        self.panel.add_child(gui.Label("Legend:"))
        for info in self.class_info:
            c_label = gui.Label(info["label"])
            c_label.text_color = gui.Color(info["color"][0], info["color"][1], info["color"][2])
            self.panel.add_child(c_label)

        # The 3D Scene on the right
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.05, 0.05, 0.05, 1.0]) # Nearly black
        
        # Add components to window
        self.window.add_child(self.panel)
        self.window.add_child(self.scene_widget)

        # Define Layout function
        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        panel_width = 250
        self.panel.frame = gui.Rect(content_rect.x, content_rect.y, panel_width, content_rect.height)
        self.scene_widget.frame = gui.Rect(content_rect.x + panel_width, content_rect.y, 
                                           content_rect.width - panel_width, content_rect.height)

    def _on_selection_changed(self, new_val, is_double_click):
        idx = self.tile_list.selected_index
        if idx < 0 or idx >= len(self.all_tiles):
            return
        self.load_tile(self.all_tiles[idx])

    def load_tile(self, folder_path):
        print(f"Loading tile: {folder_path}")
        try:
            # Load data
            coords = np.load(folder_path / "coord.npy").astype(np.float32)
            segments = np.load(folder_path / "segment.npy").flatten()

            # Create Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            
            # Map colors
            pt_colors = np.zeros((len(segments), 3))
            valid_mask = (segments >= 0) & (segments < 8)
            pt_colors[valid_mask] = self.colors[segments[valid_mask]]
            pcd.colors = o3d.utility.Vector3dVector(pt_colors)

            # Update Scene
            self.scene_widget.scene.clear_geometry()
            
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 2.0  # Normalized coords are small, don't make points too big
            
            self.scene_widget.scene.add_geometry("tile_data", pcd, mat)
            
            # CRITICAL: Since coordinates are normalized (-1 to 1),
            # we must set the camera bound specifically for this range.
            bounds = pcd.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60, bounds, bounds.get_center())
            
            # Optional: Add a small coordinate axis at the center
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.scene_widget.scene.add_geometry("axis", axis, rendering.MaterialRecord())
            # Add this inside load_tile in npy_viewer.py
            print(f"Successfully loaded {len(coords)} points.")
            print(f"--- Tile Stats ---")
            print(f"Point count: {len(coords)}")
            print(f"X range: {coords[:,0].min():.3f} to {coords[:,0].max():.3f}")
            print(f"Z range: {coords[:,2].min():.3f} to {coords[:,2].max():.3f}")

            # Calculate theoretical max points for a 50m tile at 0.1m voxel (top-down)
            # 50/0.1 = 500. 500 * 500 = 250,000 max points per 'layer'
            
            
        except Exception as e:
            print(f"Failed to load tile: {e}")

    def run(self):
        gui.Application.instance.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/fractal01/PointceptALS/data/DALESObjects_training_data")
    args = parser.parse_args()
    
    if not Path(args.data_path).exists():
        print("Path does not exist.")
    else:
        app = TileViewer(args.data_path)
        app.run()
