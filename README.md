# The Project overview

**This is the APRD (Analysis, Plan, Requirements, Design) document for your Python AI Aerochrome Tool.**

---
### A: ANALYSIS
- **Objective:** Create a local desktop application that converts standard 16-bit RGB TIFF images into false-color Aerochrome (Infrared) emulations.
- **Core Problem:** RGB sensors block Infrared light. To simulate it, we must synthetically generate an "IR Channel" based on the likelihood of an object containing chlorophyll.
- **differentiation:** Unlike simple LUTs, this tool uses Semantic Segmentation to isolate vegetation and differentiates "Fresh" vs. "Old" foliage based on spectral heuristics (Luminance/Hue analysis within the mask).
- **Target Output:** 16-bit TIFF (uncompressed), preserving the full dynamic range of the input.

### P: PLAN
We will execute this in 4 distinct phases to isolate complexity.
- **Phase 1: The 16-bit Pipeline & GUI Skeleton:**
  - Build the I/O system using tifffile (to bypass PIL's 8-bit truncation).
  - Create a basic CustomTkinter interface with an image viewer and placeholder sliders.
  - Structure:
    > `AerochromeAI/`\
    `├── main.py            # Entry point`\
    `├── core/`\
    `│   ├── __init__.py`\
    `│   ├── pipeline.py    # Image processing logic`\
    `│   └── loader.py      # 16-bit Tiff loading/saving`\
    `└── ui/`\
    `    ├── __init__.py`\
    `    └── app.py         # GUI Logic`

- **Phase 2: The Vision Engine (Segmentation):**
  - Implement SegFormer (b5-ade20k) to classify the image into masks: Sky, Vegetation, Inorganic.
  - Implement the caching system: Run AI once per image load, not every time a slider moves.

- **Phase 3: The "Chlorophyll" Heuristics:**
  - Develop the logic to determine the IR based on the object type and level of greens, instead of segmenting into old and new foliage.
  - Generate the "Synthetic IR Map."

- **Phase 4: The Color Engine (Matrix Math):**
  - Implement the Aerochrome matrix transformation.
  - Link UI sliders to the matrix weights (e.g., "Spectrum Shift" modifies the channel blending).

### R: REQUIREMENTS
- **Hardware:**
  - GPU: NVIDIA (CUDA) or Mac Silicon (MPS) highly recommended.
  - RAM: 16GB+ (Processing 16-bit full-res arrays requires significant memory).

- **Libraries (The "Stack"):**
  - **UI:** customtkinter (Modern, dark-mode aware).
  - **I/O:** tifffile (Strict requirement for 16-bit support).
  - **AI:** transformers, torch.
  - **Image Proc:** opencv-python (Mask blurring), numpy (Matrix math).

- **Model Weights:**
  - `nvidia/segformer-b5-finetuned-ade20k-512x512` (Best balance of accuracy/speed for foliage).

### D: DESIGN
This is the technical architecture for the codebase.

1. **Data Flow:**\
`Input (16-bit TIFF)` $\rightarrow$ `Normalization (0.0 - 1.0 Float)` $\rightarrow$ `AI Segmentation` $\rightarrow$ `Mask Generation` $\rightarrow$ `Color Grading` $\rightarrow$ `Denormalization (16-bit Int)` $\rightarrow$ `Output`


2. **The "Mask Caching" Strategy (Critical for Performance)**\
Since 16-bit images are heavy and AI is slow, we cannot re-run inference when the user drags a slider. 
   - **On Image Load:** Run SegFormer. Generate 3 static masks: `Mask_Sky`, `Mask_Flora`, `Mask_Other`. Store these in RAM. 
   - **On Slider Drag:** Use the cached masks + the original pixel data to calculate the color shift. This makes the UI responsive.


3. **The "Chlorophyll" Logic Class**\
We will treat the "IR Channel" as a calculated grayscale image map. 
$$\text{Simulated IR} = (\text{Mask}_{\text{Foliage}} \times \text{Green}_{\text{Channel}}) \times \text{Gain}$$
    - Mask_Foliage: The AI simply says "This area is plants" (0 or 1)
    - Green_Channel: The intensity of green in that pixel (0.0 to 1.0).
    - Gain: A slider you control to boost the effect.


4. **UI Layout**
- **Left Panel:** Image Preview (Downsampled for speed).
- **Right Panel:**
  - _Global:_ "Aerochrome Strength"
  - _Foliage:_ "Freshness Sensitivity" (Threshold for what counts as fresh), "Chlorophyll Gain" (How bright the red is).
  - _Sky:_ "Sky Darkening" (Simulating the polarizing effect of IR).
