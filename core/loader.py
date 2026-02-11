import numpy as np
import tifffile
from PIL import Image


class ImageLoader:
    def load_image(self, path):
        """
        Loads a 16-bit or 8-bit image and converts it to a normalized Float32 numpy array (0.0 - 1.0).
        Returns: (image_float, original_metadata)
        """
        try:
            # tifffile handles 16-bit properly where PIL sometimes struggles
            img = tifffile.imread(path)

            # Handle metadata/profile preservation later; for now capturing basics
            dtype = img.dtype

            # Normalize to 0.0 - 1.0 Float32 for processing
            if dtype == 'uint16':
                img_float = img.astype(np.float32) / 65535.0
            elif dtype == 'uint8':
                img_float = img.astype(np.float32) / 255.0
            elif dtype == 'float32':
                img_float = img  # Already float
            else:
                raise ValueError(f"Unsupported bit depth: {dtype}")

            # Ensure we have 3 channels (RGB). Drop Alpha if present for calculation, or keep it separate.
            if len(img_float.shape) == 3 and img_float.shape[2] > 3:
                img_float = img_float[:, :, :3]  # simple drop alpha for MVP

            return img_float, dtype

        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None

    def save_image(self, path, img_float, original_dtype='uint16'):
        """
        Converts normalized Float32 back to original bit depth and saves.
        """
        try:
            # Clip values to valid range to prevent artifacts
            img_float = np.clip(img_float, 0.0, 1.0)

            if original_dtype == 'uint16':
                img_out = (img_float * 65535).astype(np.uint16)
            else:
                img_out = (img_float * 255).astype(np.uint8)

            tifffile.imwrite(path, img_out)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    def get_preview(self, img_float, max_size=1024):
        """
        Returns a PIL image for display in the UI (downsampled to 8-bit).
        """
        h, w, c = img_float.shape
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Downsample for UI speed
        img_small = (img_float[::int(1 / scale), ::int(1 / scale)] * 255).astype(np.uint8)

        # Convert to PIL for Tkinter
        return Image.fromarray(img_small)