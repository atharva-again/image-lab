# Image Processing Lab

This repository contains implementations for Digital Image Processing (DIP) experiments as outlined in the course curriculum. All implementations focus on manual pixel-level manipulation to understand the mathematical and algorithmic foundations of image processing.

## Human Readers

### Project Goal
The objective of this lab is to study and implement core image processing algorithms from scratch using Python. We prioritize manual implementation over high-level library functions to gain a deeper understanding of spatial and frequency domain transformations.

### List of Experiments
1. **Mathematical Signals**: Study and plot basic signals (unit step, sine, cosine, exponential, square wave, impulse).
2. **Point Processing**:
    - Image Negation
    - Image Thresholding
3. **Contrast Stretching**: Increasing the dynamic range of gray levels.
4. **Bit Plane Slicing**: Analyzing the relative importance of individual bits in an image.
5. **Histogram Equalization**: Enhancing image contrast by spreading the gray level distribution.
6. **Zooming Techniques**:
    - Zooming by Replication
    - Zooming by Interpolation
7. **Spatial Domain Filtering**:
    - Low Pass Filtering (Smoothing)
    - High Pass Filtering (Sharpening)
    - Median Filtering
8. **Noise Reduction**: Implementing median filtering specifically for Salt and Pepper noise.
9. **Edge Detection**: Using derivative filter masks (Prewitt, Sobel, Laplacian).
10. **Mini Project**: Application-based image processing implementation.

### Setup and Usage
This project uses uv for dependency management.

1. **Install uv**: Follow the instructions at astral.sh/uv.
2. **Run an experiment**:
   ```bash
   uv run python exp6_zooming.py
   ```
3. **Visualization**: Experiments generate comparison plots using matplotlib, typically saved as PNG files in the root directory.

---

## AI Agents

### Technical Requirements
- **Package Manager**: uv (Mandatory). Use `uv add`, `uv run`, `uv pip` commands.
- **Core Dependencies**: numpy, opencv-python (for I/O only), matplotlib (for plotting).
- **Environment**: Python 3.12+ as configured in `pyproject.toml`.

### Implementation Guidelines
- **No High-Level Functions**: Do not use library-based implementation functions (e.g., `cv2.resize`, `cv2.threshold`, `cv2.equalizeHist`) for the core logic of experiments. Write the algorithms manually using loops or vectorized NumPy operations.
- **Data Types**: Be mindful of uint8 overflows during arithmetic operations. Cast to float32 for calculations and handle clipping before casting back to uint8.
- **Structure**: Each experiment should follow the `exp{N}_{name}.py` naming convention.
- **Test Assets**: Use `cameraman.jpg` as the primary test image. Fallback to synthetic patterns if the image is missing.
- **No Emojis**: Do not use emojis in code comments, commit messages, or documentation.
- **Git Protocol**: Do not commit changes to the repository unless explicitly instructed by the user.

### Knowledge Base Reference
The official lab manual is available in the repository as `image_analysis_lab.pdf`. Refer to this document for specific mathematical formulas and expected algorithmic behavior for each experiment.
