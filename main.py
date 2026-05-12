import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading

# import modules
import image_processor as ip
from zoomable_canvas import ZoomableCanvas
from live_capture import LiveCaptureWindow

class AchtknotenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Achtknoten Analyse")
        self.root.geometry("1200x800")
        
        self.current_img_bgr = None
        self.params = {}
        
        self.setup_ui()

    def setup_ui(self):
        left_frame = tk.Frame(self.root, width=320, padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        btn_upload = tk.Button(left_frame, text="Upload Picture", command=self.load_image, bg="#4A90E2", fg="white", font=('Arial', 10, 'bold'))
        btn_upload.pack(fill=tk.X, pady=(0, 15))

        btn_live = tk.Button(left_frame, text="Live-Capture", command=self.open_live_capture, bg="#E24A7A", fg="white", font=('Arial', 10, 'bold'))
        btn_live.pack(fill=tk.X, pady=(0, 15))

        # --- PARAMETER Choice ---
        param_notebook = ttk.Notebook(left_frame)
        param_notebook.pack(fill=tk.X, pady=5)
        
        tab_geo = tk.Frame(param_notebook, padx=5, pady=5)
        tab_cnn = tk.Frame(param_notebook, padx=5, pady=5)
        
        param_notebook.add(tab_geo, text="Geometrie (Skelett)")
        param_notebook.add(tab_cnn, text="CNN Pipeline")

        # 1. Parameter for Geometrie
        self.add_param(tab_geo, "Close Kernel (Geo)", "11")
        self.add_param(tab_geo, "Open Kernel (Geo)", "6")
        self.add_param(tab_geo, "Close Iterations (Geo)", "9")
        self.add_param(tab_geo, "Border Margin", "30")
        self.add_param(tab_geo, "Region Merge", "5")
        self.add_param(tab_geo, "Min Merge px", "20")
        self.add_param(tab_geo, "Rel Diag Factor", "0.015")
        self.add_param(tab_geo, "Prune Length (px)", "40")

        # 2. Parameter for CNN
        self.add_param(tab_cnn, "Border Thickness", "10")

        self.btn_analyze = tk.Button(left_frame, text="Start!", command=self.start_analysis, bg="#50E3C2", fg="black", font=('Arial', 10, 'bold'))
        self.btn_analyze.pack(fill=tk.X, pady=(15, 5))

        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)

        tk.Label(left_frame, text="Full Result:", font=('Arial', 12, 'bold')).pack(pady=(10, 0))
        self.result_var = tk.StringVar(value="-")
        entry_result = tk.Entry(left_frame, textvariable=self.result_var, font=('Arial', 12, 'bold'), justify="center", state="readonly")
        entry_result.pack(fill=tk.X, pady=5)
        
        self.raw_stats_var = tk.StringVar(value="Details...")
        tk.Label(left_frame, textvariable=self.raw_stats_var, font=('Arial', 9), justify="left").pack(pady=5, anchor="w")

        tk.Label(left_frame, text="mousewheel = zoom\nleft click = move in picture", fg="gray").pack(pady=20)

        # --- Trailer-REITER ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.tab_orig = ttk.Frame(self.notebook)
        self.tab_mask = ttk.Frame(self.notebook)
        self.tab_overlay = ttk.Frame(self.notebook)
        self.tab_cnn_img = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_orig, text="Original")
        self.notebook.add(self.tab_mask, text="Mask")
        self.notebook.add(self.tab_overlay, text="Crossing Overlay")
        self.notebook.add(self.tab_cnn_img, text="CNN Result")

        self.canvas_orig = ZoomableCanvas(self.tab_orig)
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)

        self.canvas_mask = ZoomableCanvas(self.tab_mask)
        self.canvas_mask.pack(fill=tk.BOTH, expand=True)

        self.canvas_overlay = ZoomableCanvas(self.tab_overlay)
        self.canvas_overlay.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_cnn = ZoomableCanvas(self.tab_cnn_img)
        self.canvas_cnn.pack(fill=tk.BOTH, expand=True)

    def add_param(self, parent, label_text, default_val):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label_text, width=20, anchor="w").pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=8)
        entry.insert(0, default_val)
        entry.pack(side=tk.RIGHT)
        self.params[label_text] = entry

    def open_live_capture(self):
        LiveCaptureWindow(self.root, self)

    def load_from_capture(self, img_bgr):
        self.current_img_bgr = img_bgr
        self.canvas_orig.set_image(self.current_img_bgr)
        self.notebook.select(self.tab_orig)
        self.result_var.set("-")
        self.raw_stats_var.set("Details...")

        self.canvas_mask.canvas.delete("all")
        self.canvas_overlay.canvas.delete("all")
        self.canvas_cnn.canvas.delete("all")

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath: return

        self.current_img_bgr = cv2.imread(filepath)
        if self.current_img_bgr is None:
            messagebox.showerror("Fehler", "Bild konnte nicht geladen werden.")
            return

        self.canvas_orig.set_image(self.current_img_bgr)
        self.notebook.select(self.tab_orig)
        self.result_var.set("-")
        self.raw_stats_var.set("Details...")

        self.canvas_mask.original_image = None
        self.canvas_overlay.original_image = None
        self.canvas_cnn.original_image = None
        self.canvas_mask.canvas.delete("all")
        self.canvas_overlay.canvas.delete("all")
        self.canvas_cnn.canvas.delete("all")

    def start_analysis(self):
        if self.current_img_bgr is None:
            messagebox.showwarning("Achtung", "Bitte lade zuerst ein Bild hoch.")
            return

        try:
            # Geometrie-Parameter
            p_geo = {
                'c_k': int(self.params["Close Kernel (Geo)"].get()),
                'o_k': int(self.params["Open Kernel (Geo)"].get()),
                'c_iter': int(self.params["Close Iterations (Geo)"].get()),
                'b_margin': int(self.params["Border Margin"].get()),
                'r_merge': int(self.params["Region Merge"].get()),
                'min_merge': int(self.params["Min Merge px"].get()),
                'rel_diag': float(self.params["Rel Diag Factor"].get()),
                'prune_length': int(self.params["Prune Length (px)"].get()) 
            }
            # CNN-Parameter
            p_cnn = {
                'b_thick': int(self.params["Border Thickness"].get()),
            }
        except ValueError:
            messagebox.showerror("Fehler", "Bitte überprüfe, ob alle Parameter gültige Zahlen sind.")
            return

        self.btn_analyze.config(state=tk.DISABLED)
        self.progress.start(15)
        self.result_var.set("analyse...")

        threading.Thread(target=self.analysis_task, args=(p_geo, p_cnn), daemon=True).start()

    def analysis_task(self, p_geo, p_cnn):
        try:
            # 1. Geometrische Analyse
            mask_clean = ip.build_rope_mask(
                self.current_img_bgr, 
                close_k=(p_geo['c_k'], p_geo['c_k']), open_k=(p_geo['o_k'], p_geo['o_k']), 
                close_iter=p_geo['c_iter']
            )

            cross_result = ip.analyze_skeleton_crossings(
                mask_clean, 
                border_margin=p_geo['b_margin'], region_merge_size=p_geo['r_merge'], 
                min_merge_px=p_geo['min_merge'], rel_diag_factor=p_geo['rel_diag'],
                prune_length=p_geo['prune_length'] 
            )

            crossings = cross_result['crossing_centers_merged']
            skeleton = cross_result['seed_skeleton']
            overlay = ip.make_overlay(self.current_img_bgr, mask_clean, skeleton, crossings)
            
            # 2. CNN Prediction
            cnn_class, cnn_conf, debug_img = ip.predictor.predict_with_pipeline(self.current_img_bgr, p_cnn, mask_clean)

            self.root.after(0, self.finish_analysis, mask_clean, overlay, debug_img, len(crossings), cnn_class, cnn_conf, None)

        except Exception as e:
            self.root.after(0, self.finish_analysis, None, None, None, 0, "", 0.0, str(e))

    def finish_analysis(self, mask_clean, overlay, debug_img, count, cnn_class, cnn_conf, error_msg):
        self.progress.stop()
        self.btn_analyze.config(state=tk.NORMAL)

        if error_msg:
            messagebox.showerror("Fehler bei der Analyse", error_msg)
            self.result_var.set("Fehler")
            return

        self.canvas_mask.set_image(mask_clean)
        self.canvas_overlay.set_image(overlay)
        if debug_img is not None:
            self.canvas_cnn.set_image(debug_img)

        # KOMBINATION OF CNN AND GEOMETRIE
        self.raw_stats_var.set(f"Kreuzpunkte: {count}\nCNN: {cnn_class.upper()} ({cnn_conf:.1f}%)")
        
        has_correct_crossings = (4 <= count <= 6)
        
        if cnn_class == 'wrong' and cnn_conf > 95.0:
            self.result_var.set("WRONG (bad CNN)")
            self.notebook.select(self.tab_cnn_img)
            
        elif has_correct_crossings and cnn_class == 'correct' and cnn_conf > 85.0:
            self.result_var.set("CORRECT (double validation)")
            self.notebook.select(self.tab_overlay)
            
        elif has_correct_crossings:
            self.result_var.set("CORRECT (Topologie)")
            self.notebook.select(self.tab_overlay)
            
        else:
            self.result_var.set(f"WRONG ({count} CROSSINGS)")
            self.notebook.select(self.tab_overlay)

if __name__ == "__main__":
    root = tk.Tk()
    app = AchtknotenApp(root)
    root.mainloop()