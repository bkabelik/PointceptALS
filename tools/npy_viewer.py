import os
import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pathlib import Path
import time

class TileViewer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        
        self.class_info = [
            {"label": "Ground", "color": [0.5, 0.5, 0.5]},
            {"label": "Vegetation", "color": [0.0, 1.0, 0.0]},
            {"label": "Cars", "color": [1.0, 0.0, 0.0]},
            {"label": "Trucks", "color": [1.0, 0.5, 0.0]},
            {"label": "Powerlines", "color": [1.0, 1.0, 0.0]},
            {"label": "Fences", "color": [0.0, 0.0, 1.0]},
            {"label": "Poles", "color": [0.0, 1.0, 1.0]},
            {"label": "Buildings", "color": [1.0, 0.0, 1.0]},
        ]
        self.colors = np.array([c["color"] for c in self.class_info])

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("DALES Tile Explorer", 1280, 720)
        
        em = self.window.theme.font_size
        margin = gui.Margins(int(0.5 * em))
        
        self.panel = gui.Vert(0, margin)
        self.panel.add_child(gui.Label("Select a Tile:"))
        self.tile_list = gui.ListView()
        self.all_tiles = sorted([p.parent for p in self.base_path.glob("**/coord.npy")])
        
        if not self.all_tiles:
            self.tile_list.set_items(["No tiles found!"])
        else:
            self.tile_list.set_items([str(p.relative_to(self.base_path)) for p in self.all_tiles])
        
        self.tile_list.set_on_selection_changed(self._on_selection_changed)
        self.panel.add_child(self.tile_list)

        self.panel.add_fixed(em)
        self.panel.add_child(gui.Label("Cursor Position:"))
        self.coord_label = gui.Label("X: 0.00, Y: 0.00, Z: 0.00")
        self.coord_label.text_color = gui.Color(0.8, 0.8, 0.2)
        self.panel.add_child(self.coord_label)

        self.panel.add_fixed(em)
        self.panel.add_child(gui.Label("Legend:"))
        for info in self.class_info:
            c_label = gui.Label(info["label"])
            c_label.text_color = gui.Color(info["color"][0], info["color"][1], info["color"][2])
            self.panel.add_child(c_label)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.05, 0.05, 0.05, 1.0])
        
        self.last_hover_time = 0
        self.hover_delay = 0.05 
        self.scene_widget.set_on_mouse(self._on_mouse_widget_event)

        self.window.add_child(self.panel)
        self.window.add_child(self.scene_widget)
        self.window.set_on_layout(self._on_layout)

    def _update_label(self, text):
        self.coord_label.text = text

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        panel_width = 250
        self.panel.frame = gui.Rect(content_rect.x, content_rect.y, panel_width, content_rect.height)
        self.scene_widget.frame = gui.Rect(content_rect.x + panel_width, content_rect.y, 
                                           content_rect.width - panel_width, content_rect.height)

    def _on_mouse_widget_event(self, event):
        if event.type == gui.MouseEvent.Type.MOVE:
            current_time = time.time()
            if current_time - self.last_hover_time > self.hover_delay:
                self.last_hover_time = current_time
                
                def depth_callback(depth_image):
                    x = event.x - self.scene_widget.frame.x
                    y = event.y - self.scene_widget.frame.y
                    depth_array = np.asarray(depth_image)
                    if 0 <= x < depth_array.shape[1] and 0 <= y < depth_array.shape[0]:
                        depth = depth_array[y, x]
                        if depth < 1.0:
                            world_point = self.scene_widget.scene.camera.unproject(
                                x, y, depth,  # <--- Changed from event.x, event.y
                                self.scene_widget.frame.width, self.scene_widget.frame.height)
                            txt = f"X: {world_point[0]:.3f}, Y: {world_point[1]:.3f}, Z: {world_point[2]:.3f}"
                            gui.Application.instance.post_to_main_thread(self.window, lambda: self._update_label(txt))
                        else:
                            gui.Application.instance.post_to_main_thread(self.window, lambda: self._update_label("X: ---, Y: ---, Z: ---"))

                self.scene_widget.scene.scene.render_to_depth_image(depth_callback)
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_selection_changed(self, new_val, is_double_click):
        idx = self.tile_list.selected_index
        if 0 <= idx < len(self.all_tiles):
            self.load_tile(self.all_tiles[idx])

    def load_tile(self, folder_path):
        try:
            coords = np.load(folder_path / "coord.npy").astype(np.float32)
            segments = np.load(folder_path / "segment.npy").flatten()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pt_colors = np.zeros((len(segments), 3))
            valid_mask = (segments >= 0) & (segments < 8)
            pt_colors[valid_mask] = self.colors[segments[valid_mask]]
            pcd.colors = o3d.utility.Vector3dVector(pt_colors)
            self.scene_widget.scene.clear_geometry()
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 2.0
            self.scene_widget.scene.add_geometry("tile_data", pcd, mat)
            bounds = pcd.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60, bounds, bounds.get_center())
        except Exception as e:
            print(f"Failed to load: {e}")

    def run(self):
        gui.Application.instance.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    args = parser.parse_args()
    app = TileViewer(args.data_path)
    app.run()
