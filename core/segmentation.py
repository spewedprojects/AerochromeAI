import torch
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image


class SegmentationEngine:
    def __init__(self):
        self.model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
        self.processor = None
        self.model = None
        self.device = self._get_device()

        # ADE20K indices for Aerochrome logic
        self.IDX_SKY = [2]
        self.IDX_FLORA = [4, 9, 17, 72]  # Tree, Grass, Plant, Palm

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Mac Silicon
        else:
            return "cpu"

    def load_model(self):
        """Loads model into memory only when needed to save RAM on startup."""
        if self.model is None:
            print(f"Loading AI Model ({self.device})... this may take a moment...")
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            print("Model loaded successfully.")

    def segment_image(self, image_float):
        """
        Input: Float32 RGB image (H, W, 3) normalized 0.0-1.0
        Output: Dictionary of binary masks {'flora': np.array, 'sky': np.array}
        """
        self.load_model()

        h, w, _ = image_float.shape

        # 1. Convert Numpy Float -> PIL Image for the processor
        # We multiply by 255 because the processor expects standard 8-bit range inputs
        image_uint8 = (image_float * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)

        # 2. Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: [1, 150, 512, 512]

        # 4. Resize mask back to original image size
        # We use bilinear interpolation on the logits before argmax for smoother edges
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        # 5. Get Class IDs
        pred_seg = upsampled_logits.argmax(dim=1)[0]  # Shape: [H, W]

        # Move back to CPU/Numpy
        pred_seg = pred_seg.cpu().numpy()

        # 6. Create Binary Masks
        # isin is faster than looping. Checks if pixel class is in our list.
        mask_flora = np.isin(pred_seg, self.IDX_FLORA).astype(np.float32)
        mask_sky = np.isin(pred_seg, self.IDX_SKY).astype(np.float32)

        return {
            "flora": mask_flora,
            "sky": mask_sky
        }