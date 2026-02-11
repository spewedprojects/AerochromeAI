import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import time

from core.loader import ImageLoader
from core.segmentation import SegmentationEngine
from core.color_engine import ColorEngine

ctk.set_appearance_mode("Dark")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Aerochrome AI - Final Build")
        self.geometry("1400x900")

        # --- Core Modules ---
        self.loader = ImageLoader()
        self.ai_engine = SegmentationEngine()
        self.color_engine = ColorEngine()

        # --- State ---
        self.current_image = None  # Original RGB (Float32)
        self.current_masks = None  # AI Masks
        self.processed_image = None  # Final Result
        self.last_process_time = 0

        self._setup_ui()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === LEFT SIDEBAR (Controls) ===
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Header
        ctk.CTkLabel(self.sidebar, text="CONTROLS", font=("Arial", 16, "bold")).pack(pady=(20, 10))

        # 1. Load & Analyze
        self.btn_load = ctk.CTkButton(self.sidebar, text="1. Load Image", command=self.load_file)
        self.btn_load.pack(pady=10, padx=20, fill="x")

        self.btn_analyze = ctk.CTkButton(self.sidebar, text="2. Run AI Analysis",
                                         command=self.run_analysis, state="disabled", fg_color="gray")
        self.btn_analyze.pack(pady=10, padx=20, fill="x")

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Waiting for image...", text_color="gray", wraplength=250)
        self.lbl_status.pack(pady=10)

        # Separator
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").pack(fill="x", padx=10, pady=10)

        # 2. Sliders (Disabled until AI runs)
        ctk.CTkLabel(self.sidebar, text="CHLOROPHYLL (IR)", anchor="w").pack(padx=20, fill="x")
        self.slider_gain = self._create_slider("Intensity", 0.5, 3.0, 1.2)

        ctk.CTkLabel(self.sidebar, text="ATMOSPHERE", anchor="w").pack(padx=20, fill="x", pady=(20, 0))
        self.slider_sky = self._create_slider("Sky Protect", 0.0, 1.0, 0.0)
        self.slider_sat = self._create_slider("Saturation", 0.0, 2.0, 1.1)

        # 3. Export
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").pack(fill="x", padx=10, pady=20)
        self.btn_save = ctk.CTkButton(self.sidebar, text="Save 16-bit TIFF", command=self.save_file, state="disabled",
                                      fg_color="green")
        self.btn_save.pack(pady=10, padx=20, fill="x")

        # === RIGHT AREA (Preview) ===
        self.preview_area = ctk.CTkFrame(self, fg_color="transparent")
        self.preview_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.lbl_image = ctk.CTkLabel(self.preview_area, text="Load an image to begin")
        self.lbl_image.pack(expand=True, fill="both")

    def _create_slider(self, label, vmin, vmax, vdefault):
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=5)

        lbl = ctk.CTkLabel(frame, text=label, width=80, anchor="w")
        lbl.pack(side="left")

        slider = ctk.CTkSlider(frame, from_=vmin, to=vmax, command=self.on_slider_change)
        slider.set(vdefault)
        slider.pack(side="right", expand=True, fill="x", padx=5)
        return slider

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF", "*.tif *.tiff")])
        if path:
            self.lbl_status.configure(text="Loading...")
            self.update()
            self.current_image, _ = self.loader.load_image(path)

            if self.current_image is not None:
                self.lbl_status.configure(text="Loaded. Click 'Run AI' to unlock sliders.")
                self.btn_analyze.configure(state="normal", fg_color="#1f538d")  # Reset blue color
                self.display_image(self.current_image)
            else:
                self.lbl_status.configure(text="Error loading file")

    def run_analysis(self):
        if self.current_image is None: return

        self.lbl_status.configure(text="Analyzing... (Please wait)")
        self.btn_analyze.configure(state="disabled")
        self.update()

        try:
            # AI Inference
            self.current_masks = self.ai_engine.segment_image(self.current_image)

            self.lbl_status.configure(text="Analysis Done! Adjust sliders.")
            self.btn_save.configure(state="normal")

            # Trigger initial process
            self.process_image()

        except Exception as e:
            self.lbl_status.configure(text=f"AI Error: {e}")
            self.btn_analyze.configure(state="normal")

    def on_slider_change(self, value):
        # Debounce: Prevent processing if AI hasn't run yet
        if self.current_masks is None: return

        # Optional: Limit FPS if needed, but 16-bit math is fast enough for interaction usually
        self.process_image()

    def process_image(self):
        # Gather params
        params = {
            'ir_gain': self.slider_gain.get(),
            'sky_protect': self.slider_sky.get(),
            'saturation': self.slider_sat.get()
        }

        # Run Color Engine
        self.processed_image = self.color_engine.process(
            self.current_image,
            self.current_masks,
            params
        )

        # Update Display
        self.display_image(self.processed_image)

    def display_image(self, img_float):
        pil_img = self.loader.get_preview(img_float)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        self.lbl_image.configure(image=ctk_img, text="")

    def save_file(self):
        if self.processed_image is None: return

        path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF", "*.tif")])
        if path:
            self.lbl_status.configure(text="Saving 16-bit TIFF...")
            self.update()
            success = self.loader.save_image(path, self.processed_image)
            if success:
                self.lbl_status.configure(text="Saved Successfully!")
            else:
                self.lbl_status.configure(text="Error saving.")


if __name__ == "__main__":
    app = App()
    app.mainloop()