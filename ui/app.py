import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np

from core.loader import ImageLoader
from core.segmentation import SegmentationEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Aerochrome AI - Phase 2: Segmentation")
        self.geometry("1400x900")

        self.loader = ImageLoader()
        self.ai_engine = SegmentationEngine()

        self.current_image = None
        self.masks = None  # Will store our AI results

        self._setup_ui()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.btn_load = ctk.CTkButton(self.sidebar, text="Load Image", command=self.load_file)
        self.btn_load.pack(pady=20, padx=20)

        self.btn_analyze = ctk.CTkButton(self.sidebar, text="Run AI Analysis", command=self.run_analysis,
                                         state="disabled")
        self.btn_analyze.pack(pady=10, padx=20)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Waiting...", text_color="gray", wraplength=200)
        self.status_label.pack(pady=20)

        # --- Main Preview Area ---
        # We'll use tabs to switch between Original and Masks
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.tab_original = self.tab_view.add("Original")
        self.tab_mask_flora = self.tab_view.add("Flora Mask")
        self.tab_mask_sky = self.tab_view.add("Sky Mask")

        # Labels to hold images
        self.lbl_original = ctk.CTkLabel(self.tab_original, text="")
        self.lbl_original.pack(expand=True, fill="both")

        self.lbl_flora = ctk.CTkLabel(self.tab_mask_flora, text="")
        self.lbl_flora.pack(expand=True, fill="both")

        self.lbl_sky = ctk.CTkLabel(self.tab_mask_sky, text="")
        self.lbl_sky.pack(expand=True, fill="both")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF", "*.tif *.tiff")])
        if path:
            self.status_label.configure(text="Loading 16-bit data...")
            self.update()

            self.current_image, _ = self.loader.load_image(path)

            if self.current_image is not None:
                self.status_label.configure(text="Image Loaded.\nReady for AI.")
                self.btn_analyze.configure(state="normal")
                self.display_image(self.current_image, self.lbl_original)
            else:
                self.status_label.configure(text="Error loading file")

    def run_analysis(self):
        if self.current_image is None: return

        self.status_label.configure(text="Running SegFormer B5...\n(This triggers download on first run)")
        self.btn_analyze.configure(state="disabled")
        self.update()

        # Run AI
        try:
            self.masks = self.ai_engine.segment_image(self.current_image)
            self.status_label.configure(text="Analysis Complete!\nCheck tabs to see masks.")

            # Display Masks
            # Multiply by 255 to make them visible (0.0-1.0 -> 0-255)
            # We stack them into RGB so PIL can display them grayscale

            flora_vis = np.stack([self.masks['flora']] * 3, axis=-1)
            sky_vis = np.stack([self.masks['sky']] * 3, axis=-1)

            self.display_image(flora_vis, self.lbl_flora)
            self.display_image(sky_vis, self.lbl_sky)

        except Exception as e:
            self.status_label.configure(text=f"AI Error: {str(e)}")
            print(e)

        self.btn_analyze.configure(state="normal")

    def display_image(self, img_float, label_widget):
        # Reuse loader's preview logic
        pil_img = self.loader.get_preview(img_float)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        label_widget.configure(image=ctk_img)


if __name__ == "__main__":
    app = App()
    app.mainloop()