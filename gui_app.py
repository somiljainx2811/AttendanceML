import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import cv2
from PIL import Image, ImageTk

from face_recognizer import FaceRecognizer
from attendance_logger import AttendanceLogger

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_CSV = "attendance.csv"
RECOGNIZE_EVERY_N_FRAMES = 5     # run the (slower) recognition step every N frames
RESIZE_FACTOR = 0.25             # shrink frames before recognition for speed
CAPTURE_TARGET = 5                # photos to take when registering a new person
BOX_COLOR_KNOWN = (0, 200, 0)     # BGR
BOX_COLOR_UNKNOWN = (0, 0, 220)   # BGR


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("980x600")
        self.root.minsize(860, 540)

        self.recognizer = FaceRecognizer(KNOWN_FACES_DIR)
        self.logger = AttendanceLogger(ATTENDANCE_CSV)

        self.cap = None
        self.camera_running = False
        self.frame_count = 0
        self.last_results = []

        self.capturing_for = None
        self.capture_count = 0

        self._build_layout()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        video_frame = ttk.Frame(main)
        video_frame.pack(side="left", fill="both", expand=True)
        self.video_label = ttk.Label(video_frame, background="#222")
        self.video_label.pack(fill="both", expand=True)

        controls = ttk.Frame(main, padding=(15, 0, 0, 0))
        controls.pack(side="right", fill="y")

        ttk.Label(controls, text="Attendance System", font=("Segoe UI", 14, "bold")).pack(pady=(0, 15))

        self.start_btn = ttk.Button(controls, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(fill="x", pady=4)
        self.stop_btn = ttk.Button(controls, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(fill="x", pady=4)

        ttk.Separator(controls).pack(fill="x", pady=10)

        self.register_btn = ttk.Button(controls, text="Register New Person", command=self.start_registration)
        self.register_btn.pack(fill="x", pady=4)
        self.capture_btn = ttk.Button(controls, text="Capture Photo", command=self._capture_registration_photo, state="disabled")
        self.capture_btn.pack(fill="x", pady=4)
        self.cancel_register_btn = ttk.Button(controls, text="Cancel Registration", command=self._cancel_registration, state="disabled")
        self.cancel_register_btn.pack(fill="x", pady=4)

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Button(controls, text="View Today's Attendance", command=self.view_attendance).pack(fill="x", pady=4)
        ttk.Button(controls, text="Reload Known Faces", command=self.reload_faces).pack(fill="x", pady=4)

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Label(controls, text="Status:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.status_var = tk.StringVar(value="Camera stopped.")
        ttk.Label(controls, textvariable=self.status_var, wraplength=220, foreground="#444").pack(anchor="w", pady=(0, 10))

        ttk.Label(controls, text="Activity log:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.log_box = tk.Listbox(controls, height=14, width=32)
        self.log_box.pack(fill="both", expand=True, pady=(0, 10))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------ camera lifecycle --
    def start_camera(self):
        if self.camera_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Could not access the webcam. Check that it's connected and "
                "not already in use by another application.",
            )
            self.cap = None
            return

        self.camera_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Camera running -- looking for faces...")
        self._update_frame()

    def stop_camera(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Camera stopped.")

    def _on_close(self):
        self.stop_camera()
        self.root.destroy()

    def _update_frame(self):
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._log("Failed to read from camera.")
            self.root.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)

        if self.capturing_for is not None:
            self._draw_registration_overlay(frame)
            self._last_raw_frame = frame
        else:
            self._process_recognition(frame)

        self._render_frame(frame)
        self.root.after(20, self._update_frame)

    def _process_recognition(self, frame):
        self.frame_count += 1
        small = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if self.frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            self.last_results = self.recognizer.recognize_faces(small_rgb)
            for name, _location, _conf in self.last_results:
                if name != "Unknown":
                    newly_marked = self.logger.mark_attendance(name)
                    if newly_marked:
                        self._log(f"Marked present: {name} at {time.strftime('%H:%M:%S')}")
                        self.status_var.set(f"Marked present: {name}")

        scale = int(1 / RESIZE_FACTOR)
        for name, (top, right, bottom, left), confidence in self.last_results:
            top, right, bottom, left = top * scale, right * scale, bottom * scale, left * scale
            color = BOX_COLOR_KNOWN if name != "Unknown" else BOX_COLOR_UNKNOWN
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({confidence:.0%})" if name != "Unknown" else "Unknown"
            cv2.rectangle(frame, (left, bottom - 22), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 4, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _render_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        label_w = self.video_label.winfo_width() or 640
        label_h = self.video_label.winfo_height() or 480
        img.thumbnail((max(label_w, 320), max(label_h, 240)))
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = photo
        self.video_label.config(image=photo)

    def start_registration(self):
        if not self.camera_running:
            messagebox.showinfo("Camera not running", "Start the camera first, then register a new person.")
            return

        name = simpledialog.askstring("Register New Person", "Enter the person's name:")
        if not name:
            return
        name = name.strip().replace(" ", "_")
        if not name:
            return

        self.capturing_for = name
        self.capture_count = 0
        os.makedirs(os.path.join(KNOWN_FACES_DIR, name), exist_ok=True)

        self.register_btn.config(state="disabled")
        self.capture_btn.config(state="normal")
        self.cancel_register_btn.config(state="normal")

        self._log(f"Registering '{name}': click 'Capture Photo' {CAPTURE_TARGET} times (vary angle/expression a little).")
        self.status_var.set(f"Registering {name}: 0/{CAPTURE_TARGET} photos.")

    def _draw_registration_overlay(self, frame):
        text = f"Registering: {self.capturing_for}  ({self.capture_count}/{CAPTURE_TARGET})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    def _capture_registration_photo(self):
        if self.capturing_for is None or not hasattr(self, "_last_raw_frame"):
            return

        person_dir = os.path.join(KNOWN_FACES_DIR, self.capturing_for)
        filename = os.path.join(person_dir, f"{self.capture_count + 1}.jpg")
        cv2.imwrite(filename, self._last_raw_frame)
        self.capture_count += 1
        self.status_var.set(f"Registering {self.capturing_for}: {self.capture_count}/{CAPTURE_TARGET} photos.")

        if self.capture_count >= CAPTURE_TARGET:
            finished_name = self.capturing_for
            self._log(f"Finished capturing photos for '{finished_name}'. Reloading known faces...")
            self.capturing_for = None
            self.capture_btn.config(state="disabled")
            self.cancel_register_btn.config(state="disabled")
            self.register_btn.config(state="normal")
            self.reload_faces()

    def _cancel_registration(self):
        if self.capturing_for:
            self._log(f"Registration cancelled for '{self.capturing_for}'.")
        self.capturing_for = None
        self.capture_count = 0
        self.capture_btn.config(state="disabled")
        self.cancel_register_btn.config(state="disabled")
        self.register_btn.config(state="normal")
        self.status_var.set("Registration cancelled.")

    def reload_faces(self):
        self.recognizer.load_known_faces()
        names = sorted(set(self.recognizer.known_names))
        self._log(f"Loaded {len(self.recognizer.known_encodings)} photos for {len(names)} people.")

    def view_attendance(self):
        records = self.logger.get_today_records()
        win = tk.Toplevel(self.root)
        win.title("Today's Attendance")
        win.geometry("360x400")

        tree = ttk.Treeview(win, columns=("Name", "Time"), show="headings")
        tree.heading("Name", text="Name")
        tree.heading("Time", text="Time")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        if not records:
            tree.insert("", "end", values=("No one marked yet", ""))
        else:
            for row in records:
                tree.insert("", "end", values=(row["Name"], row["Time"]))

    def _log(self, message):
        self.log_box.insert(0, message)
        if self.log_box.size() > 200:
            self.log_box.delete(200, "end")
