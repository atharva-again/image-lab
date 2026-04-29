"""
Experiment No. 4: Contrast Stretching
Digital Image Processing Lab

This script implements contrast stretching to increase the dynamic range
of gray levels in an image.
Formula: s = ((r - r_min) / (r_max - r_min)) * 255
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def contrast_stretch(image):
    """Apply manual contrast stretching to a grayscale image."""
    r_min = np.min(image)
    r_max = np.max(image)

    if r_max == r_min:
        return image.copy()

    output = np.zeros_like(image, dtype=np.float32)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            r = float(image[i, j])
            s = ((r - r_min) / (r_max - r_min)) * 255.0
            output[i, j] = s

    return np.clip(output, 0, 255).astype(np.uint8)


def display_results(original, stretched):
    """Display original and contrast-stretched images with histograms."""
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.hist(original.ravel(), bins=256, range=(0, 256), color="gray")
    plt.title("Original Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 3)
    plt.plot([0, 255], [0, 255], "r--", label="Identity")
    plt.title("Transformation Function")
    plt.xlabel("Input r")
    plt.ylabel("Output s")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.imshow(stretched, cmap="gray")
    plt.title("Contrast Stretched")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.hist(stretched.ravel(), bins=256, range=(0, 256), color="gray")
    plt.title("Stretched Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 6)
    r_min = np.min(original)
    r_max = np.max(original)
    r_vals = np.arange(0, 256)
    s_vals = np.clip(((r_vals - r_min) / (r_max - r_min)) * 255, 0, 255)
    plt.plot(r_vals, s_vals, "b-", label="Stretch")
    plt.title("Actual Transformation")
    plt.xlabel("Input r")
    plt.ylabel("Output s")
    plt.legend()

    plt.tight_layout()
    plt.savefig("contrast_stretching.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved contrast_stretching.png")


def main():
    print("=" * 60)
    print("Experiment No. 4: Contrast Stretching")
    print("=" * 60)

    image_path = "naayab.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Creating synthetic low-contrast test image...")
        image = np.random.randint(100, 150, size=(256, 256), dtype=np.uint8)
        cv2.imwrite("synthetic_low_contrast.png", image)
        print("Created synthetic_low_contrast.png")
    else:
        print(f"Loaded image: {image_path}")

    print("Applying contrast stretching...")
    stretched = contrast_stretch(image)

    display_results(image, stretched)

    cv2.imwrite("contrast_stretched_output.png", stretched)
    print("Saved contrast_stretched_output.png")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
