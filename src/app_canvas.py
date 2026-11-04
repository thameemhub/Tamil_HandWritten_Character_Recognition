"""
src/app_canvas.py – Tkinter Drawing Canvas & Live Few-Shot Recognition App.

Features:
  - Interactive drawing canvas for handwritten Tamil characters
  - Real-time Top-5 prediction using class prototypes
  - Rejection threshold indicator (flags unknown/not-enrolled characters)
  - Zero-retraining letter enrollment:
      Draw a letter, type the Tamil label (e.g. ச, சா, ழ, ணே), and click 'Enroll'.
      The sample is immediately saved to data/support_set/<label>/ and prototypes
      are recalculated on-the-fly without retraining!
"""

from __future__ import annotations

import logging
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.config import get_config
from src.preprocessing import preprocess_image
from src.support_registry import SupportRegistry
from src.tamil_text import find_tamil_font

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app_canvas")


class TamilCanvasApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tamil Handwritten Character Recognition – Few-Shot Canvas")
        self.root.geometry("820x620")
        self.root.configure(bg="#1e1e1e")

        self.cfg = get_config()
        self.registry = SupportRegistry(cfg=self.cfg)

        # Canvas settings
        self.canvas_size = 360
        self.brush_size = 18
        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None

        # Pillow image mirror for canvas drawing (white ink on black background)
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self._build_ui()

    def _build_ui(self):
        # Header
        header = tk.Label(
            self.root,
            text="தமிழ் கையெழுத்து எழுத்துணர்தல் (Tamil Few-Shot HTR)",
            font=("Nirmala UI", 16, "bold"),
            fg="#60a5fa",
            bg="#1e1e1e"
        )
        header.pack(pady=10)

        # Main horizontal split
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Column: Canvas & Controls
        left_col = tk.Frame(main_frame, bg="#1e1e1e")
        left_col.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(
            left_col,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            cursor="crosshair",
            highlightthickness=2,
            highlightbackground="#3b82f6"
        )
        self.canvas.pack(pady=5)

        # Canvas mouse bindings
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        # Canvas buttons
        btn_frame = tk.Frame(left_col, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, pady=8)

        pred_btn = tk.Button(
            btn_frame,
            text="Recognize (கணி)",
            font=("Nirmala UI", 11, "bold"),
            bg="#22c55e",
            fg="white",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            command=self.predict_canvas
        )
        pred_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(
            btn_frame,
            text="Clear (அழி)",
            font=("Nirmala UI", 11),
            bg="#ef4444",
            fg="white",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            command=self.clear_canvas
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Brush size slider
        slider_frame = tk.Frame(left_col, bg="#1e1e1e")
        slider_frame.pack(fill=tk.X, pady=5)
        tk.Label(slider_frame, text="Brush Size:", fg="white", bg="#1e1e1e", font=("Nirmala UI", 9)).pack(side=tk.LEFT)
        self.brush_slider = tk.Scale(slider_frame, from_=8, to_=32, orient=tk.HORIZONTAL, bg="#1e1e1e", fg="white", highlightthickness=0)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # Right Column: Predictions & New Letter Enrollment
        right_col = tk.Frame(main_frame, bg="#1e1e1e")
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Predictions box
        pred_label = tk.Label(right_col, text="Top Predictions:", fg="#cbd5e1", bg="#1e1e1e", font=("Nirmala UI", 12, "bold"))
        pred_label.pack(anchor=tk.W, pady=2)

        self.pred_listbox = tk.Listbox(
            right_col,
            height=7,
            font=("Nirmala UI", 14),
            bg="#2d3748",
            fg="white",
            selectbackground="#4f46e5",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#4b5563"
        )
        self.pred_listbox.pack(fill=tk.X, pady=5)

        self.status_label = tk.Label(
            right_col,
            text="Draw a character and click 'Recognize'",
            fg="#94a3b8",
            bg="#1e1e1e",
            font=("Nirmala UI", 10, "italic")
        )
        self.status_label.pack(anchor=tk.W, pady=4)

        # Enrollment Section
        enroll_box = tk.LabelFrame(
            right_col,
            text=" Enroll as New Letter (புதிய எழுத்து சேர்க்க) ",
            font=("Nirmala UI", 11, "bold"),
            fg="#38bdf8",
            bg="#1e1e1e",
            padx=10,
            pady=10
        )
        enroll_box.pack(fill=tk.X, pady=15)

        tk.Label(
            enroll_box,
            text="Enter Tamil Letter (e.g. ச, சா, ழ, ணே):",
            fg="#e2e8f0",
            bg="#1e1e1e",
            font=("Nirmala UI", 10)
        ).pack(anchor=tk.W, pady=2)

        self.enroll_entry = tk.Entry(
            enroll_box,
            font=("Nirmala UI", 13),
            bg="#2d3748",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT
        )
        self.enroll_entry.pack(fill=tk.X, pady=5)

        enroll_btn = tk.Button(
            enroll_box,
            text="Enroll Current Drawing (சேர்)",
            font=("Nirmala UI", 10, "bold"),
            bg="#0284c7",
            fg="white",
            relief=tk.FLAT,
            pady=4,
            command=self.enroll_letter
        )
        enroll_btn.pack(fill=tk.X, pady=5)

        info_lbl = tk.Label(
            enroll_box,
            text="Zero retraining required! Prototype updates instantly.",
            fg="#64748b",
            bg="#1e1e1e",
            font=("Nirmala UI", 8)
        )
        info_lbl.pack(anchor=tk.W)

    def _start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def _draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            r = self.brush_slider.get() // 2
            # Draw on Tkinter canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_slider.get(),
                fill="white",
                capstyle=tk.ROUND,
                smooth=True
            )
            # Mirror on Pillow image
            self.pil_draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255,
                width=self.brush_slider.get()
            )
            self.pil_draw.ellipse(
                [event.x - r, event.y - r, event.x + r, event.y + r],
                fill=255
            )
            self.last_x = event.x
            self.last_y = event.y

    def _stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.pred_listbox.delete(0, tk.END)
        self.status_label.config(text="Canvas cleared.", fg="#94a3b8")

    def predict_canvas(self):
        img_np = np.array(self.pil_image, dtype=np.uint8)
        if img_np.max() == 0:
            messagebox.showinfo("Empty Canvas", "Please draw a character first.")
            return

        # Preprocess directly
        preprocessed = preprocess_image(
            img_np,
            image_size=self.cfg.preprocessing.image_size,
            use_binarization=True
        )
        if preprocessed is None:
            messagebox.showwarning("Preprocessing Error", "Could not preprocess character.")
            return

        results = self.registry.predict(preprocessed, top_k=5)
        self.pred_listbox.delete(0, tk.END)

        if not results:
            self.status_label.config(text="No predictions available.", fg="#f87171")
            return

        top = results[0]
        if top["rejected"]:
            self.status_label.config(text="⚠ Character rejected: Unknown / Not enrolled letter", fg="#f87171")
        else:
            self.status_label.config(
                text=f"✓ Top match: '{top['class']}' with {top['confidence']*100:.1f}% confidence",
                fg="#4ade80"
            )

        for i, r in enumerate(results, 1):
            rej_tag = " [UNKNOWN]" if r["rejected"] else ""
            line = f" #{i}  {r['class']}  —  {r['confidence']*100:.1f}% (dist: {r['distance']:.3f}){rej_tag}"
            self.pred_listbox.insert(tk.END, line)

    def enroll_letter(self):
        label = self.enroll_entry.get().strip()
        if not label:
            messagebox.showwarning("Missing Label", "Please type the Tamil character label to enroll.")
            return

        img_np = np.array(self.pil_image, dtype=np.uint8)
        if img_np.max() == 0:
            messagebox.showinfo("Empty Canvas", "Please draw a character to enroll.")
            return

        # Preprocess and enroll
        pre = preprocess_image(img_np, self.cfg.preprocessing.image_size)
        if pre is None:
            messagebox.showwarning("Error", "Could not preprocess the drawn sample.")
            return

        save_arr = (pre * 255.0).astype(np.uint8)
        self.registry.enroll_from_canvas_image(label, save_arr)

        messagebox.showinfo(
            "Success",
            f"Successfully enrolled '{label}'!\nClass prototype updated in real-time."
        )
        self.enroll_entry.delete(0, tk.END)
        self.predict_canvas()


def launch_app():
    root = tk.Tk()
    app = TamilCanvasApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_app()

