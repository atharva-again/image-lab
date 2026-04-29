# Image Processing Lab

This repository contains implementations for Digital Image Processing (DIP) experiments as outlined in the course curriculum. All implementations focus on manual pixel-level manipulation to understand the mathematical and algorithmic foundations of image processing.

## Human Readers

### Project Goal
The objective of this lab is to study and implement core image processing algorithms from scratch using Python. We prioritize manual implementation over high-level library functions to gain a deeper understanding of spatial and frequency domain transformations.

### List of Experiments
1. **Basic Signals**: Study and plot basic signals (unit step, sine, cosine, exponential, square wave, impulse).
2. **Negative Image**: Compute the photographic negative of a grayscale image using point processing.
3. **Thresholding**: Convert a grayscale image to binary using manual thresholding at different levels.
4. **Contrast Stretching**: Increase the dynamic range of gray levels in a low-contrast image.
5. **Bit Plane Slicing**: Decompose an image into its 8 individual bit planes.
6. **Histogram Equalization**: Enhance image contrast by redistributing pixel intensities uniformly.
7. **Magnification**: Zoom images using pixel replication and bilinear interpolation.
8. **Low Pass Filtering**: Smooth images using mean (averaging) filters in the spatial domain.
9. **Median Filtering**: Remove salt and pepper noise using a manual median filter.
10. **Edge Detection & Segmentation**: Detect edges using Sobel operators and segment by thresholding edge magnitude.

### Setup and Usage
This project uses uv for dependency management.

1. **Install uv**: Follow the instructions at astral.sh/uv.
2. **Sync dependencies**:
   ```bash
   uv sync
   ```
3. **Run an experiment**:
   ```bash
   uv run python a7/a7_magnification.py
   ```
4. **Visualization**: Experiments generate comparison plots using matplotlib, typically saved within the respective `aN/` directory alongside the source code.

---

## AI Agents

### Technical Requirements
- **Package Manager**: uv (Mandatory). Use `uv add`, `uv run`, `uv pip` commands.
- **Core Dependencies**: numpy, opencv-python (for I/O only), matplotlib (for plotting).
- **Environment**: Python 3.12+ as configured in `pyproject.toml`.

### Implementation Guidelines
- **No High-Level Functions**: Do not use library-based implementation functions (e.g., `cv2.resize`, `cv2.threshold`, `cv2.equalizeHist`) for the core logic of experiments. Write the algorithms manually using loops or vectorized NumPy operations.
- **Data Types**: Be mindful of uint8 overflows during arithmetic operations. Cast to float32 for calculations and handle clipping before casting back to uint8.
- **Structure**: Each experiment is organized in its own `a{N}/` directory with the file named `a{N}_{description}.py`.
- **Test Assets**: Use `cameraman.jpg` as the primary test image. Fallback to synthetic patterns if the image is missing.
- **No Emojis**: Do not use emojis in code comments, commit messages, or documentation.
- **Git Protocol**: Do not commit changes to the repository unless explicitly instructed by the user.

### Knowledge Base Reference
The official lab manual is available in the repository as `image_analysis_lab.pdf`. Refer to this document for specific mathematical formulas and expected algorithmic behavior for each experiment.
