import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import laspy
import os
import sys
import threading

# Standard ASPRS/DALES Class Colors (R, G, B) mapping
CLASS_COLORS = {
    0: [0.5, 0.5, 0.5], # Created, never classified
    1: [0.5, 0.5, 0.5], # Unclassified
    2: [0.8, 0.4, 0.1], # Ground (Orange)
    3: [0.1, 0.8, 0.1], # Low Veg (Light Green)
    4: [0.0, 0.6, 0.0], # Med Veg (Green)
    5: [0.0, 0.4, 0.0], # High Veg (Dark Green)
    6: [0.1, 0.2, 0.8], # Building (Blue)
    7: [0.8, 0.1, 0.8], # Low Point/Noise (Purple)
    8: [1.0, 0.0, 0.0], # Key Point (Red)
    9: [0.1, 0.8, 0.8], # Water (Cyan)
    14: [0.8, 0.8, 0.1], # Wire/Conductor (Yellow)
    15: [0.5, 0.2, 0.1], # Transmission Tower
    19: [0.9, 0.5, 0.5], # Overhead Structure
    20: [0.6, 0.6, 0.6], # Ignored Ground
}

def get_class_colors(classes):
    colors = np.zeros((len(classes), 3))
    for k, v in CLASS_COLORS.items():
        colors[classes == k] = v
    known = np.isin(classes, list(CLASS_COLORS.keys()))
    colors[~known] = [0.7, 0.7, 0.7]
    return colors

class ViewerWindow:
    """A single viewer window with one SceneWidget."""
    def __init__(self, title, x, y, width, height):
        self.window = gui.Application.instance.create_window(
            title, width, height)
        
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.0, 0.0, 0.0, 1.0])
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.window.add_child(self.scene)
        self.window.set_on_layout(self._on_layout)
        self.pcd = None
        
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene.frame = gui.Rect(r.x, r.y, r.width, r.height)
    
    def set_geometry(self, pcd, mat, bbox):
        self.pcd = pcd  # Persist reference
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("points", pcd, mat)
        self.scene.setup_camera(60.0, bbox, bbox.get_center())
        self.window.post_redraw()


class CompareLasApp:
    def __init__(self):
        self.file1_path = ""
        self.file2_path = ""
        self.pcds = []
        self._bbox = None
        
        # Main control window
        self.window = gui.Application.instance.create_window(
            "Compare Two LAS Files", 400, 700)
        
        em = self.window.theme.font_size
        
        # Controls panel
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("Compare Two LAS Classifications"))
        
        # File 1 Selection
        self.btn_file1 = gui.Button("Select File 1...")
        self.btn_file1.set_on_clicked(lambda: self._on_select_file(1))
        self.label_file1 = gui.Label("None selected")
        self.panel.add_child(self.btn_file1)
        self.panel.add_child(self.label_file1)
        
        # File 2 Selection
        self.btn_file2 = gui.Button("Select File 2...")
        self.btn_file2.set_on_clicked(lambda: self._on_select_file(2))
        self.label_file2 = gui.Label("None selected")
        self.panel.add_child(self.btn_file2)
        self.panel.add_child(self.label_file2)
        
        self.panel.add_fixed(em)
        
        # Action
        self.btn_compare = gui.Button("Load and Compare")
        self.btn_compare.set_on_clicked(self._on_compare)
        self.btn_compare.horizontal_padding_em = 1.0
        self.btn_compare.vertical_padding_em = 0.5
        self.panel.add_child(self.btn_compare)
        
        self.panel.add_fixed(em)
        
        # Reset views button
        self.btn_reset = gui.Button("Reset All Views")
        self.btn_reset.set_on_clicked(self._on_reset_views)
        self.btn_reset.horizontal_padding_em = 1.0
        self.btn_reset.vertical_padding_em = 0.3
        self.panel.add_child(self.btn_reset)
        
        self.panel.add_fixed(em)
        
        # Stats
        self.stats_panel = gui.ScrollableVert(
            0.2 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.label_stats = gui.Label("Statistics will appear here.\n")
        self.stats_panel.add_child(self.label_stats)
        self.panel.add_child(self.stats_panel)
        
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_main_layout)
        
        # Create 3 separate viewer windows
        vw = 500
        vh = 600
        self.viewers = [
            ViewerWindow("File 1 - Classification", 410, 50, vw, vh),
            ViewerWindow("File 2 - Classification", 420 + vw, 50, vw, vh),
            ViewerWindow("Difference Map", 430 + 2 * vw, 50, vw, vh),
        ]
        
        # Camera sync state
        self._last_view_matrices = [None, None, None]
        self._pending_sync_source = -1
        self._sync_cooldown = 0
        self._post_sync_ignore = 0
        self.window.set_on_tick_event(self._on_tick)
    
    def _on_main_layout(self, layout_context):
        r = self.window.content_rect
        self.panel.frame = gui.Rect(r.x, r.y, r.width, r.height)

    def _on_tick(self):
        """Debounced camera sync across the 3 viewer windows."""
        if not self.pcds:
            return True
        
        # Post-sync cooldown: skip detection while look_at settles
        if self._post_sync_ignore > 0:
            self._post_sync_ignore -= 1
            for i in range(3):
                self._last_view_matrices[i] = np.array(
                    self.viewers[i].scene.scene.camera.get_view_matrix())
            return True
        
        # Detect which viewer's camera changed
        changed_idx = -1
        for i in range(3):
            mat = np.array(
                self.viewers[i].scene.scene.camera.get_view_matrix())
            if self._last_view_matrices[i] is not None:
                if not np.allclose(mat, self._last_view_matrices[i], atol=0.01):
                    changed_idx = i
            self._last_view_matrices[i] = mat
        
        if changed_idx != -1:
            # Camera is moving — keep resetting cooldown
            self._pending_sync_source = changed_idx
            self._sync_cooldown = 8
        elif self._sync_cooldown > 0:
            self._sync_cooldown -= 1
            if self._sync_cooldown == 0 and self._pending_sync_source >= 0:
                self._do_sync(self._pending_sync_source)
                self._pending_sync_source = -1
        
        return True

    def _do_sync(self, source_idx):
        """Sync all viewer windows to match source_idx's camera."""
        source = self.viewers[source_idx]
        cam = source.scene.scene.camera
        view_mat = np.array(cam.get_view_matrix(), dtype=np.float64)
        R = view_mat[:3, :3]
        t = view_mat[:3, 3]
        
        eye = (-R.T @ t).astype(np.float32)
        up = (R[1, :3]).astype(np.float32)
        center = np.array(source.scene.center_of_rotation, dtype=np.float32)
        
        # Validate
        if (np.any(~np.isfinite(eye)) or np.any(~np.isfinite(center))
                or np.any(~np.isfinite(up))):
            return
        up_len = np.linalg.norm(up)
        if up_len < 1e-6:
            return
        up = up / up_len
        
        for i in range(3):
            if i != source_idx:
                self.viewers[i].scene.look_at(center, eye, up)
                self.viewers[i].window.post_redraw()
        
        # Refresh stored matrices and pause detection
        for i in range(3):
            self._last_view_matrices[i] = np.array(
                self.viewers[i].scene.scene.camera.get_view_matrix())
        self._post_sync_ignore = 15

    def _on_reset_views(self):
        """Reset all viewers to the default camera position."""
        if self._bbox is not None:
            for viewer in self.viewers:
                viewer.scene.setup_camera(
                    60.0, self._bbox, self._bbox.get_center())
                viewer.window.post_redraw()

    def _on_select_file(self, file_idx):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, f"Select File {file_idx}", self.window.theme)
        dlg.add_filter(".las .laz", "LAS files (.las, .laz)")
        dlg.add_filter("", "All files")
        
        def on_done(path):
            if file_idx == 1:
                self.file1_path = path
                self.label_file1.text = os.path.basename(path)
            else:
                self.file2_path = path
                self.label_file2.text = os.path.basename(path)
            self.window.close_dialog()
            
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(on_done)
        self.window.show_dialog(dlg)

    def _on_compare(self):
        if not self.file1_path or not self.file2_path:
            self.label_stats.text = "Error: Please select both files."
            return
        if not os.path.exists(self.file1_path) or not os.path.exists(self.file2_path):
            self.label_stats.text = "Error: One or both files do not exist."
            return

        self.label_stats.text = "Loading and Processing (Please wait)..."
        self.btn_compare.enabled = False
        threading.Thread(target=self._load_and_process_thread).start()

    def _load_and_process_thread(self):
        try:
            print(f"Loading File 1: {self.file1_path}")
            las1 = laspy.read(self.file1_path)
            pts1 = np.array(las1.xyz)
            cls1 = np.array(las1.classification)
            
            print(f"Loading File 2: {self.file2_path}")
            las2 = laspy.read(self.file2_path)
            pts2 = np.array(las2.xyz)
            cls2 = np.array(las2.classification)
            
            print(f"Comparing {len(pts1):,} points...")
            if len(pts1) != len(pts2):
                def update_err():
                    self.label_stats.text = (
                        f"Error: Point counts differ!\n"
                        f"File 1: {len(pts1):,}\nFile 2: {len(pts2):,}")
                    self.btn_compare.enabled = True
                gui.Application.instance.post_to_main_thread(
                    self.window, update_err)
                return
            
            # Stats
            matches = (cls1 == cls2)
            total_points = len(pts1)
            num_matches = int(np.sum(matches))
            num_mismatches = total_points - num_matches
            accuracy = (num_matches / total_points) * 100.0
            
            stats_text = f"Total Points: {total_points:,}\n"
            stats_text += f"Matches: {num_matches:,} ({accuracy:.2f}%)\n"
            stats_text += f"Mismatches: {num_mismatches:,} ({100-accuracy:.2f}%)\n\n"
            
            if num_mismatches > 0:
                print(f"Calculating breakdown for {num_mismatches:,} mismatches...")
                stats_text += "Mismatch Breakdown (File 1 -> File 2):\n"
                mismatch_mask = ~matches
                cls1_miss = cls1[mismatch_mask]
                cls2_miss = cls2[mismatch_mask]
                
                class_names = {
                    2: "Ground", 3: "Low Veg", 4: "Med Veg", 5: "High Veg",
                    6: "Building", 7: "Noise", 9: "Water", 14: "Wire",
                    15: "Tower", 19: "Overhead", 20: "Ignored"
                }
                
                pairs, counts = np.unique(
                    np.column_stack((cls1_miss, cls2_miss)),
                    axis=0, return_counts=True)
                sort_idx = np.argsort(-counts)
                pairs = pairs[sort_idx]
                counts = counts[sort_idx]
                
                for (c1, c2), count in zip(pairs, counts):
                    name1 = class_names.get(c1, str(c1))
                    name2 = class_names.get(c2, str(c2))
                    stats_text += f"  {name1} -> {name2}: {count:,}\n"

            print("Centering and Preparing geometry...")
            center = np.mean(pts1, axis=0)
            pts_local = (pts1 - center).astype(np.float64)
            
            # Clamp Z outliers
            z_vals = pts_local[:, 2]
            z_lo = np.percentile(z_vals, 0.5)
            z_hi = np.percentile(z_vals, 99.5)
            pts_local[:, 2] = np.clip(z_vals, z_lo, z_hi)
            print(f"Z range after clamp: {z_lo:.1f} to {z_hi:.1f}")
            
            # Colors
            col1 = get_class_colors(cls1).astype(np.float64)
            col2 = get_class_colors(cls2).astype(np.float64)
            col_diff = np.full((total_points, 3), [0.2, 0.2, 0.2], dtype=np.float64)
            col_diff[~matches] = [1.0, 0.1, 0.1]
            
            print("Requesting UI Update...")
            def update_ui():
                try:
                    print("Updating UI and Scenes...")
                    self.label_stats.text = stats_text
                    
                    mat = rendering.MaterialRecord()
                    mat.shader = "defaultUnlit"
                    mat.point_size = 3.0
                    
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(pts_local)
                    pcd1.colors = o3d.utility.Vector3dVector(col1)
                    
                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector(pts_local)
                    pcd2.colors = o3d.utility.Vector3dVector(col2)
                    
                    pcd3 = o3d.geometry.PointCloud()
                    pcd3.points = o3d.utility.Vector3dVector(pts_local)
                    pcd3.colors = o3d.utility.Vector3dVector(col_diff)
                    
                    self.pcds = [pcd1, pcd2, pcd3]
                    
                    bbox = pcd1.get_axis_aligned_bounding_box()
                    self._bbox = bbox
                    print(f"BBox: {bbox}")
                    
                    # Push geometry to each viewer window
                    for viewer, pcd in zip(self.viewers, self.pcds):
                        viewer.set_geometry(pcd, mat, bbox)
                    
                    print("Update Complete. All 3 viewer windows are ready.")
                except Exception as e:
                    print(f"Error in UI update: {e}")
                    import traceback
                    traceback.print_exc()
                    self.label_stats.text += f"\nError: {str(e)}"
                finally:
                    self.btn_compare.enabled = True

            gui.Application.instance.post_to_main_thread(self.window, update_ui)
                
        except Exception as e:
            print(f"Error in processing thread: {e}")
            import traceback
            traceback.print_exc()
            def update_err():
                self.label_stats.text = f"Error:\n{str(e)}"
                self.btn_compare.enabled = True
            gui.Application.instance.post_to_main_thread(self.window, update_err)

def main():
    gui.Application.instance.initialize()
    app = CompareLasApp()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()
