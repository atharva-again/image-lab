# AGENTS.md

## Project Overview
This repository contains Python implementations for Digital Image Processing (DIP) experiments. The core philosophy is **manual algorithmic implementation** to understand mathematical principles, avoiding high-level library abstractions.

## Development Environment
- **Package Manager**: `uv` (Mandatory).
- **Python Version**: >= 3.12.
- **Core Libraries**: `numpy`, `opencv-python` (for I/O only), `matplotlib` (for visualization).

## Build and Execution Commands
- **Install Dependencies**: `uv sync`
- **Add Package**: `uv add <package_name>`
- **Run Experiment**: `uv run python exp{N}_{name}.py` (e.g., `uv run python exp6_zooming.py`)
- **Linting/Diagnostics**: Use `lsp_diagnostics` tool on the target file.

## Code Style Guidelines

### 1. Manual Implementation Rule (CRITICAL)
- **DO NOT** use high-level library functions for core algorithmic steps.
- **Forbidden**: `cv2.resize()`, `cv2.threshold()`, `cv2.equalizeHist()`, `cv2.GaussianBlur()`, etc.
- **Allowed**: Use `cv2.imread()` and `cv2.imwrite()` for image I/O only.
- **Vectorization**: Prefer NumPy vectorized operations over explicit loops for performance, but loops are acceptable if they match the pedagogical logic of the lab manual.

### 2. Naming Conventions
- **Files**: `exp{N}_{description}.py` (e.g., `exp6_zooming.py`).
- **Functions**: `snake_case` (e.g., `zoom_by_interpolation`).
- **Variables**: `snake_case` (e.g., `image_data`, `row_idx`).
- **Constants**: `UPPER_SNAKE_CASE`.

### 3. Imports
- Standard library imports first, then third-party libraries.
- Use established aliases:
  ```python
  import numpy as np
  import cv2
  import matplotlib.pyplot as plt
  ```

### 4. Error Handling
- Use explicit checks rather than broad try-except blocks where possible.
- **Image Loading**: Always check if `image is None` after `cv2.imread()`.
- **Fallbacks**: Provide fallback logic (e.g., synthetic image generation) if assets are missing.

### 5. Data Types and Overflows
- Images are typically `uint8` [0-255].
- **Calculation**: Cast to `float32` or `int64` before performing arithmetic to prevent overflow/underflow.
- **Storage**: Clip results to [0, 255] and cast back to `uint8` before saving or displaying.

### 6. Documentation and Constraints
- **Docstrings**: Provide a brief summary of the experiment's logic and parameters.
- **No Emojis**: Do not use emojis in comments, docstrings, or documentation.
- **Git**: Do not commit changes unless explicitly requested by the user.

## Reference Material
- Refer to `image_analysis_lab.pdf` for the mathematical formulas and specific requirements for each experiment.
- Refer to `README.md` for the current status of the laboratory curriculum.
