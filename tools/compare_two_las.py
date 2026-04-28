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
    # Map each class to its color, default to grey if unknown
    for k, v in CLASS_COLORS.items():
        colors[classes == k] = v
    # Find unknown classes and set to grey
    known = np.isin(classes, list(CLASS_COLORS.keys()))
    colors[~known] = [0.7, 0.7, 0.7]
    return colors

class CompareLasApp:
    def __init__(self):
        self.file1_path = ""
        self.file2_path = ""
        
        self.window = gui.Application.instance.create_window("Compare Two LAS Files", 1600, 900)
        
        em = self.window.theme.font_size
        
        # Left Panel (Controls)
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
        
        # Stats
        self.stats_panel = gui.ScrollableVert(0.2 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.label_stats = gui.Label("Statistics will appear here.\n")
        self.stats_panel.add_child(self.label_stats)
        self.panel.add_child(self.stats_panel)
        
        # Right Panel (Viewers) - No container, added directly to window for manual layout
        self.scenes = []
        self.view_containers = []
        for title in ["File 1", "File 2", "Difference"]:
            v = gui.Vert()
            title_label = gui.Label(title)
            v.add_child(title_label)
            
            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(self.window.renderer)
            scene.scene.set_background([0.0, 0.0, 0.0, 1.0])
            scene.background_color = gui.Color(0, 0, 0, 1)
            scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)
            v.add_child(scene)
            
            self.window.add_child(v)
            self.scenes.append(scene)
            self.view_containers.append(v)
            
        self.window.add_child(self.panel)
        
        self.window.set_on_layout(self._on_layout)
        
        # Synchronization logic
        self.last_view_matrices = [None, None, None]
        self.window.set_on_tick_event(self._on_tick)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = self.window.theme.font_size
        panel_width = 350
        self.panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        
        view_x = r.x + panel_width
        view_width = r.width - panel_width
        
        # Manually size the 3 viewers to be equal width
        spacing = 0.2 * em
        child_width = (view_width - 2 * spacing) / 3.0
        
        label_height = em * 1.5
        for i, container in enumerate(self.view_containers):
            # 1. Set the container frame
            cont_x = view_x + i * (child_width + spacing)
            container.frame = gui.Rect(cont_x, r.y, child_width, r.height)
            
            # 2. Layout children of the Vert container (Label and SceneWidget)
            # Note: Child frames are relative to their parent container!
            children = container.get_children()
            if len(children) >= 2:
                title_label = children[0]
                scene_widget = children[1]
                
                # Title label at the top
                title_label.frame = gui.Rect(0, 0, child_width, label_height)
                # SceneWidget takes the rest of the height
                scene_widget.frame = gui.Rect(0, label_height, child_width, r.height - label_height)

    def _on_select_file(self, file_idx):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, f"Select File {file_idx}", self.window.theme)
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
                print(f"Error: Point count mismatch ({len(pts1)} vs {len(pts2)})")
                def update_err():
                    self.label_stats.text = f"Error: Point counts differ!\nFile 1: {len(pts1):,}\nFile 2: {len(pts2):,}"
                    self.btn_compare.enabled = True
                gui.Application.instance.post_to_main_thread(self.window, update_err)
                return
            
            # Stats calculation
            matches = (cls1 == cls2)
            total_points = len(pts1)
            num_matches = np.sum(matches)
            num_mismatches = total_points - num_matches
            accuracy = (num_matches / total_points) * 100.0
            
            stats_text = f"Total Points: {total_points:,}\n"
            stats_text += f"Matches: {num_matches:,} ({accuracy:.2f}%)\n"
            stats_text += f"Mismatches: {num_mismatches:,} ({(100-accuracy):.2f}%)\n\n"
            
            # Mismatch breakdown
            if num_mismatches > 0:
                print(f"Calculating breakdown for {num_mismatches:,} mismatches...")
                stats_text += "Mismatch Breakdown (File 1 -> File 2):\n"
                mismatch_mask = ~matches
                cls1_miss = cls1[mismatch_mask]
                cls2_miss = cls2[mismatch_mask]
                
                # Class name mapping
                class_names = {
                    2: "Ground", 3: "Low Veg", 4: "Med Veg", 5: "High Veg",
                    6: "Building", 7: "Noise", 9: "Water", 14: "Wire",
                    15: "Tower", 19: "Overhead", 20: "Ignored"
                }
                
                pairs, counts = np.unique(np.column_stack((cls1_miss, cls2_miss)), axis=0, return_counts=True)
                sort_idx = np.argsort(-counts)
                pairs = pairs[sort_idx]
                counts = counts[sort_idx]
                
                for (c1, c2), count in zip(pairs, counts):
                    name1 = class_names.get(c1, str(c1))
                    name2 = class_names.get(c2, str(c2))
                    stats_text += f"  {name1} -> {name2}: {count:,}\n"

            print("Centering and Preparing geometry...")
            # Localize coordinates for rendering
            center = np.mean(pts1, axis=0)
            pts_local = (pts1 - center).astype(np.float64)
            
            # Colors
            col1 = get_class_colors(cls1).astype(np.float64)
            col2 = get_class_colors(cls2).astype(np.float64)
            
            # Diff colors: Match = Grey, Mismatch = Red
            col_diff = np.full((total_points, 3), [0.2, 0.2, 0.2], dtype=np.float64)
            col_diff[~matches] = [1.0, 0.1, 0.1]
            
            print("Requesting UI Update...")
            def update_ui():
                try:
                    print("Updating UI and Scene...")
                    self.label_stats.text = stats_text
                    
                    mat = rendering.MaterialRecord()
                    mat.shader = "defaultUnlit"
                    mat.point_size = 5.0
                    
                    # Create PointClouds on the main thread for safety
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(pts_local)
                    pcd1.colors = o3d.utility.Vector3dVector(col1)
                    
                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector(pts_local)
                    pcd2.colors = o3d.utility.Vector3dVector(col2)
                    
                    pcd3 = o3d.geometry.PointCloud()
                    pcd3.points = o3d.utility.Vector3dVector(pts_local)
                    pcd3.colors = o3d.utility.Vector3dVector(col_diff)
                    
                    self.pcds = [pcd1, pcd2, pcd3] # Persist references
                    
                    bbox = pcd1.get_axis_aligned_bounding_box()
                    print(f"BBox: {bbox}")
                    for i, (scene, pcd) in enumerate(zip(self.scenes, self.pcds)):
                        scene.scene.clear_geometry()
                        scene.scene.add_geometry(f"points_{i}", pcd, mat)
                        scene.setup_camera(60.0, bbox, bbox.get_center())
                        self.last_view_matrices[i] = scene.scene.camera.get_model_matrix()
                        
                    self.window.post_redraw()
                    print("Update Complete.")
                except Exception as e:
                    print(f"Error in UI update: {e}")
                    self.label_stats.text += f"\nError in UI update: {str(e)}"
                finally:
                    self.btn_compare.enabled = True

            gui.Application.instance.post_to_main_thread(self.window, update_ui)
                
        except Exception as e:
            print(f"Error in processing thread: {e}")
            import traceback
            traceback.print_exc()
            def update_err():
                self.label_stats.text = f"Error processing files:\n{str(e)}"
                self.btn_compare.enabled = True
            gui.Application.instance.post_to_main_thread(self.window, update_err)

    def _on_tick(self):
        # Synchronization check
        changed_idx = -1
        current_matrices = []
        for i, scene in enumerate(self.scenes):
            cam = scene.scene.camera
            mat = cam.get_model_matrix()
            current_matrices.append(mat)
            if self.last_view_matrices[i] is not None:
                if not np.allclose(mat, self.last_view_matrices[i], atol=1e-3):
                    changed_idx = i
                    
        if changed_idx != -1:
            # Sync others to changed_idx
            source_cam = self.scenes[changed_idx].scene.camera
            fov = source_cam.get_field_of_view()
            fov_type = source_cam.get_field_of_view_type()
            near = source_cam.get_near()
            far = source_cam.get_far()
            
            for i, scene in enumerate(self.scenes):
                if i != changed_idx:
                    target_cam = scene.scene.camera
                    target_cam.copy_from(source_cam)
                    
                    # Fix: Re-apply projection with the source's FOV to maintain zoom
                    # and calculate aspect ratio for the target window
                    frame = scene.frame
                    aspect = float(frame.width) / float(max(1, frame.height))
                    target_cam.set_projection(fov, aspect, near, far, fov_type)
                    
                    self.last_view_matrices[i] = target_cam.get_model_matrix()
            
            self.last_view_matrices[changed_idx] = current_matrices[changed_idx]
            self.window.post_redraw()
            
        return True

def main():
    gui.Application.instance.initialize()
    app = CompareLasApp()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()
