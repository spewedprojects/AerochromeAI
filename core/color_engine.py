import numpy as np


class ColorEngine:
    def process(self, image_float, masks, params):
        """
        image_float: (H, W, 3) Float32 0.0-1.0
        masks: {'flora': (H,W), 'sky': (H,W)}
        params: dict of slider values
        """
        # Unpack params
        ir_gain = params.get('ir_gain', 1.0)  # How red the trees get
        sky_protect = params.get('sky_protect', 0.0)  # Blend original sky back?
        saturation = params.get('saturation', 1.0)  # Global saturation boost

        # 1. Extract Channels
        r_in = image_float[:, :, 0]
        g_in = image_float[:, :, 1]
        b_in = image_float[:, :, 2]

        # 2. GENERATE SYNTHETIC IR CHANNEL
        # Logic: Everything reflects some IR (base), but Plants reflect A LOT (gain).
        # We use the Green channel as a proxy for "Chlorophyll density" inside the mask.

        base_ir = (r_in * 0.1 + g_in * 0.1 + b_in * 0.1)  # Base reflectivity of the world
        flora_ir = g_in * masks['flora'] * ir_gain  # The "Pop" from the plants

        synthetic_ir = base_ir + flora_ir
        synthetic_ir = np.clip(synthetic_ir, 0.0, 1.0)

        # 3. AEROCHROME CHANNEL SWAP (The "Film" Emulation)
        # Real Aerochrome:
        #   IR Light  -> Red Dye
        #   Red Light -> Green Dye
        #   Green Light -> Blue Dye
        #   (Blue Light is usually blocked by a yellow filter)

        r_out = synthetic_ir
        g_out = r_in
        b_out = g_in

        # 4. COMPOSITE
        result = np.stack([r_out, g_out, b_out], axis=-1)

        # 5. SKY PROTECTION (Optional)
        # If the user wants normal blue skies instead of the Aerochrome cyan/black sky
        if sky_protect > 0:
            # Expand mask to 3 channels for multiplication
            sky_mask_3 = np.stack([masks['sky']] * 3, axis=-1)
            # Blend: result * (1-mask) + original * mask
            result = result * (1 - (sky_mask_3 * sky_protect)) + \
                     image_float * (sky_mask_3 * sky_protect)

        # 6. GLOBAL SATURATION (Simple adjustment)
        if saturation != 1.0:
            # Convert to gray to get luminance
            gray = result @ np.array([0.299, 0.587, 0.114])
            gray = np.stack([gray] * 3, axis=-1)
            result = gray + (result - gray) * saturation

        return np.clip(result, 0.0, 1.0)