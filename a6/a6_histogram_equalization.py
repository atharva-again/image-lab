"""
Experiment No. 6: Histogram Equalization
Digital Image Processing Lab

This script implements histogram equalization manually to enhance image contrast.
Steps:
1. Compute histogram manually
2. Compute CDF (cumulative distribution function)
3. Normalize CDF and map pixel values
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_histogram(image):
    """Compute histogram of a grayscale image manually."""
    hist = np.zeros(256, dtype=np.int64)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            hist[image[i, j]] += 1
    return hist


def histogram_equalization(image):
    """Apply manual histogram equalization to a grayscale image."""
    rows, cols = image.shape
    total_pixels = rows * cols

    hist = compute_histogram(image)

    cdf = np.zeros(256, dtype=np.int64)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = np.min(cdf[cdf > 0])

    output = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            r = image[i, j]
            s = round(((cdf[r] - cdf_min) / (total_pixels - cdf_min)) * 255)
            output[i, j] = int(s)

    return output


def display_results(original, equalized):
    """Display original and equalized images with histograms."""
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.bar(range(256), compute_histogram(original), color="gray", width=1)
    plt.title("Original Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 3)
    cdf_original = np.cumsum(compute_histogram(original))
    plt.plot(range(256), cdf_original, color="blue")
    plt.title("Original CDF")
    plt.xlabel("Pixel Value")
    plt.ylabel("Cumulative Frequency")

    plt.subplot(2, 3, 4)
    plt.imshow(equalized, cmap="gray")
    plt.title("Equalized Image")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.bar(range(256), compute_histogram(equalized), color="gray", width=1)
    plt.title("Equalized Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 6)
    cdf_equalized = np.cumsum(compute_histogram(equalized))
    plt.plot(range(256), cdf_equalized, color="blue")
    plt.title("Equalized CDF")
    plt.xlabel("Pixel Value")
    plt.ylabel("Cumulative Frequency")

    plt.tight_layout()
    plt.savefig("histogram_equalization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved histogram_equalization.png")


def main():
    print("=" * 60)
    print("Experiment No. 6: Histogram Equalization")
    print("=" * 60)

    image_path = "feyd_irulan.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Creating synthetic test image...")
        image = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            image[:, i] = i
        cv2.imwrite("synthetic_gradient.png", image)
        print("Created synthetic_gradient.png")
    else:
        print(f"Loaded image: {image_path}")

    print("Applying histogram equalization...")
    equalized = histogram_equalization(image)

    display_results(image, equalized)

    cv2.imwrite("equalized_output.png", equalized)
    print("Saved equalized_output.png")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
