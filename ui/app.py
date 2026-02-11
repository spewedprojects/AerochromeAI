import customtkinter as ctk
from tkinter import filedialog
from core.loader import ImageLoader

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Aerochrome AI - 16bit Workstation")
        self.geometry("1200x800")

        self.loader = ImageLoader()
        self.current_image = None  # This will hold our heavy float32 array
        self.original_dtype = None

        # --- Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Sidebar (Controls)
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.btn_load = ctk.CTkButton(self.sidebar, text="Load 16-bit TIFF", command=self.load_file)
        self.btn_load.pack(pady=20, padx=20)

        self.label_status = ctk.CTkLabel(self.sidebar, text="No image loaded", text_color="gray")
        self.label_status.pack(pady=10)

        # Right Area (Preview)
        self.preview_area = ctk.CTkFrame(self, fg_color="transparent")
        self.preview_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.label_image = ctk.CTkLabel(self.preview_area, text="")  # Image goes here
        self.label_image.pack(expand=True, fill="both")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF", "*.tif *.tiff")])
        if path:
            self.label_status.configure(text="Loading 16-bit data...")
            self.update()  # Force UI update

            # Load the heavy data
            self.current_image, self.original_dtype = self.loader.load_image(path)

            if self.current_image is not None:
                self.label_status.configure(text=f"Loaded: {self.current_image.shape}\nDepth: {self.original_dtype}")
                self.show_preview()
            else:
                self.label_status.configure(text="Error loading file")

    def show_preview(self):
        # Generate a lightweight 8-bit preview for the UI
        pil_img = self.loader.get_preview(self.current_image)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        self.label_image.configure(image=ctk_img)


if __name__ == "__main__":
    app = App()
    app.mainloop()