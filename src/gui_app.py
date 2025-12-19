import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

# Drag & Drop
from tkinterdnd2 import DND_FILES, TkinterDnD

# -------------------------------
# Load Model (FAST: ONNX)
# -------------------------------
MODEL_PATH = "best.onnx"   # must be in same folder as exe
model = YOLO(MODEL_PATH)

# -------------------------------
# App Window
# -------------------------------
root = TkinterDnD.Tk()
root.title("Military Object Detection System | Innovatrix")
root.geometry("1000x720")
root.configure(bg="#1e1e2e")

# -------------------------------
# Styling
# -------------------------------
style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TButton",
    font=("Segoe UI", 14),
    padding=10,
    background="#4f46e5",
    foreground="white"
)

style.map(
    "TButton",
    background=[("active", "#4338ca")]
)

# -------------------------------
# Title
# -------------------------------
title = tk.Label(
    root,
    text="🛡 Military Object Detection System",
    font=("Segoe UI", 24, "bold"),
    fg="white",
    bg="#1e1e2e"
)
title.pack(pady=15)

subtitle = tk.Label(
    root,
    text="Drag & Drop an image or click Select Image",
    font=("Segoe UI", 13),
    fg="#c7c7d1",
    bg="#1e1e2e"
)
subtitle.pack()

# -------------------------------
# Image Display Frame
# -------------------------------
frame = tk.Frame(root, bg="#2a2a40", bd=2, relief="ridge")
frame.pack(pady=20, padx=30, fill="both", expand=True)

panel = tk.Label(
    frame,
    text="Drop Image Here",
    font=("Segoe UI", 18),
    fg="#9ca3af",
    bg="#2a2a40"
)
panel.pack(expand=True)

# -------------------------------
# Status Bar
# -------------------------------
status = tk.Label(
    root,
    text="Status: Waiting for input",
    font=("Segoe UI", 11),
    fg="#a5b4fc",
    bg="#1e1e2e",
    anchor="w"
)
status.pack(fill="x", padx=20, pady=5)

# -------------------------------
# Detection Function
# -------------------------------
def detect_image(image_path):
    status.config(text="Status: Running detection...")
    root.update_idletasks()

    results = model(
        image_path,
        imgsz=640,
        conf=0.25,
        device="cpu"
    )

    annotated = results[0].plot()

    h, w, _ = annotated.shape
    scale = min(900 / w, 520 / h)
    annotated = cv2.resize(
        annotated,
        (int(w * scale), int(h * scale))
    )

    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    panel.config(image=img_tk, text="")
    panel.image = img_tk

    status.config(text=f"Status: Detection complete ✔ ({os.path.basename(image_path)})")

# -------------------------------
# File Picker
# -------------------------------
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        detect_image(file_path)

# -------------------------------
# Drag & Drop Handler
# -------------------------------
def drop_event(event):
    file_path = event.data.strip("{}")
    if file_path.lower().endswith((".jpg", ".png", ".jpeg")):
        detect_image(file_path)

panel.drop_target_register(DND_FILES)
panel.dnd_bind("<<Drop>>", drop_event)

# -------------------------------
# Buttons
# -------------------------------
btn_frame = tk.Frame(root, bg="#1e1e2e")
btn_frame.pack(pady=15)

select_btn = ttk.Button(
    btn_frame,
    text="📂 Select Image",
    command=open_file
)
select_btn.pack()

# -------------------------------
# Start App
# -------------------------------
root.mainloop()
