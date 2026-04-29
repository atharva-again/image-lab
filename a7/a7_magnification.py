"""
Experiment No. 7: Magnification by Replication and Interpolation
Digital Image Processing Lab

This script implements two zooming techniques:
Part A: Zooming by Replication (Pixel Duplication)
Part B: Zooming by Interpolation (Bilinear Interpolation)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def zoom_by_replication(image, factor=2):
    """Zoom by replicating each pixel into a factor x factor block."""
    rows, cols = image.shape
    zoomed = np.zeros((rows * factor, cols * factor), dtype=image.dtype)
    for i in range(rows):
        for j in range(cols):
            pixel_value = image[i, j]
            for fi in range(factor):
                for fj in range(factor):
                    zoomed[i * factor + fi, j * factor + fj] = pixel_value
    return zoomed


def zoom_by_interpolation(image, factor=2):
    """Zoom by bilinear interpolation."""
    rows, cols = image.shape
    new_rows = rows * factor
    new_cols = cols * factor
    zoomed = np.zeros((new_rows, new_cols), dtype=np.float32)

    for i in range(new_rows):
        for j in range(new_cols):
            x = i / factor
            y = j / factor

            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, rows - 1)
            y1 = min(y0 + 1, cols - 1)

            dx = x - x0
            dy = y - y0

            top_left = float(image[x0, y0])
            top_right = float(image[x0, y1])
            bottom_left = float(image[x1, y0])
            bottom_right = float(image[x1, y1])

            top = top_left + dy * (top_right - top_left)
            bottom = bottom_left + dy * (bottom_right - bottom_left)
            value = top + dx * (bottom - top)

            zoomed[i, j] = value

    return np.clip(zoomed, 0, 255).astype(np.uint8)


def display_results(original, replication, interpolation):
    """Display original and zoomed images."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title(f"Original\n{original.shape}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(replication, cmap="gray")
    plt.title(f"Replication\n{replication.shape}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(interpolation, cmap="gray")
    plt.title(f"Interpolation\n{interpolation.shape}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("magnification_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved magnification_comparison.png")


def main():
    print("=" * 60)
    print("Experiment No. 7: Magnification")
    print("=" * 60)

    image_path = "qutub_minar.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Creating synthetic test image...")
        image = np.zeros((100, 100), dtype=np.uint8)
        image[25:75, 25:75] = 128
        image[40:60, 40:60] = 255
        cv2.imwrite("synthetic_test.png", image)
        print("Created synthetic_test.png")
    else:
        print(f"Loaded image: {image_path}")

    print("Applying magnification by replication (2x)...")
    replication = zoom_by_replication(image, factor=2)

    print("Applying magnification by interpolation (2x)...")
    interpolation = zoom_by_interpolation(image, factor=2)

    display_results(image, replication, interpolation)

    cv2.imwrite("zoom_replication.png", replication)
    cv2.imwrite("zoom_interpolation.png", interpolation)
    print("Saved zoom_replication.png, zoom_interpolation.png")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
