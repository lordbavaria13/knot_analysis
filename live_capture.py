import tkinter as tk
import threading
import cv2
import PIL.Image
import PIL.ImageTk


class LiveCaptureWindow:
    def __init__(self, parent, app):
        self.app = app
        self.cap = None
        self.captured_frame = None
        self.is_streaming = False

        self.win = tk.Toplevel(parent)
        self.win.title("Live-Capture (DroidCam App)")
        self.win.geometry("700x600")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()

    def _build_ui(self):
        conn_frame = tk.LabelFrame(self.win, text="DroidCam Connection", padx=10, pady=8)
        conn_frame.pack(fill=tk.X, padx=15, pady=(15, 5))

        row_ip = tk.Frame(conn_frame)
        row_ip.pack(fill=tk.X, pady=3)
        tk.Label(row_ip, text="IP-Adress:", width=12, anchor="w").pack(side=tk.LEFT)
        self.entry_ip = tk.Entry(row_ip, font=('Arial', 11))
        self.entry_ip.insert(0, "192.168.0.")
        self.entry_ip.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row_port = tk.Frame(conn_frame)
        row_port.pack(fill=tk.X, pady=3)
        tk.Label(row_port, text="Port:", width=12, anchor="w").pack(side=tk.LEFT)
        self.entry_port = tk.Entry(row_port, font=('Arial', 11), width=10)
        self.entry_port.insert(0, "4747")
        self.entry_port.pack(side=tk.LEFT)
        tk.Label(row_port, text="  (Standard DroidCam Port: 4747)", fg="gray").pack(side=tk.LEFT)

        self.btn_connect = tk.Button(conn_frame, text="Connect", command=self.connect,
                                     bg="#4A90E2", fg="white", font=('Arial', 10, 'bold'))
        self.btn_connect.pack(fill=tk.X, pady=(8, 2))

        self.status_var = tk.StringVar(value="no connection")
        tk.Label(conn_frame, textvariable=self.status_var, fg="gray", font=('Arial', 9)).pack(anchor="w")

        preview_frame = tk.LabelFrame(self.win, text="preview", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.preview_label = tk.Label(preview_frame, bg="#1a1a1a",
                                      text="no picture - first connect",
                                      fg="#666666", font=('Arial', 10))
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=15, pady=10)

        self.btn_capture = tk.Button(btn_frame, text="capture", command=self.capture_frame,
                                     bg="#50E3C2", fg="black", font=('Arial', 10, 'bold'),
                                     state=tk.DISABLED)
        self.btn_capture.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.btn_retake = tk.Button(btn_frame, text="retake", command=self.retake,
                                    bg="#F5A623", fg="white", font=('Arial', 10, 'bold'),
                                    state=tk.DISABLED)
        self.btn_retake.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.btn_accept = tk.Button(btn_frame, text="accept", command=self.accept,
                                    bg="#7ED321", fg="white", font=('Arial', 10, 'bold'),
                                    state=tk.DISABLED)
        self.btn_accept.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        tk.Button(btn_frame, text="close", command=self.on_close,
                  bg="#D0021B", fg="white", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

    def connect(self):
        ip = self.entry_ip.get().strip()
        port = self.entry_port.get().strip()

        if not ip or not port:
            self.status_var.set("Fehler: IP oder Port leer!")
            return

        url = f"http://{ip}:{port}/video"
        self.status_var.set(f"connect with {url}...")
        self.btn_connect.config(state=tk.DISABLED)
        self.win.update()

        threading.Thread(target=self._try_connect, args=(url,), daemon=True).start()

    def _try_connect(self, url):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            self.win.after(0, lambda: self.status_var.set("Connection failed. Check IP/Port."))
            self.win.after(0, lambda: self.btn_connect.config(state=tk.NORMAL))
            return

        self.cap = cap
        self.is_streaming = True
        self.win.after(0, lambda: self.status_var.set("Connected! Stream is running."))
        self.win.after(0, lambda: self.btn_capture.config(state=tk.NORMAL))
        self.win.after(0, self._stream_loop)

    def _stream_loop(self):
        if not self.is_streaming or self.cap is None:
            return
        if self.captured_frame is not None:
            return

        ret, frame = self.cap.read()
        if ret:
            self._show_frame_in_preview(frame)

        self.win.after(33, self._stream_loop)

    def _show_frame_in_preview(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        preview_w = 650
        scale = preview_w / w
        preview_h = int(h * scale)
        resized = cv2.resize(frame_bgr, (preview_w, min(preview_h, 340)))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)
        tk_img = PIL.ImageTk.PhotoImage(pil_img)

        self.preview_label.config(image=tk_img, text="")
        self.preview_label.image = tk_img

    def capture_frame(self):
        if self.cap is None: return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Error: No frame received.")
            return

        self.captured_frame = frame
        self.is_streaming = False
        self._show_frame_in_preview(frame)

        self.status_var.set("captured! You can retake or accept the picture.")
        self.btn_capture.config(state=tk.DISABLED)
        self.btn_retake.config(state=tk.NORMAL)
        self.btn_accept.config(state=tk.NORMAL)

    def retake(self):
        self.captured_frame = None
        self.is_streaming = True
        self.btn_capture.config(state=tk.NORMAL)
        self.btn_retake.config(state=tk.DISABLED)
        self.btn_accept.config(state=tk.DISABLED)
        self.status_var.set("Stream is running again. Take a new picture.")
        self._stream_loop()

    def accept(self):
        if self.captured_frame is None: return
        self.app.load_from_capture(self.captured_frame)
        self.on_close()

    def on_close(self):
        self.is_streaming = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.win.destroy()