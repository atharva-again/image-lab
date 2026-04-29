"""
Experiment No. 3: Thresholding
Digital Image Processing Lab

This script applies manual thresholding to a grayscale image.
Formula: s = 255 if r >= T else 0, where T is the threshold value.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def threshold_image(image, threshold):
    """Apply manual thresholding to a grayscale image."""
    output = np.zeros_like(image)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if image[i, j] >= threshold:
                output[i, j] = 255
            else:
                output[i, j] = 0
    return output


def display_results(original, thresh_128, thresh_64, thresh_192):
    """Display original and thresholded images."""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(thresh_64, cmap="gray")
    plt.title("Threshold T = 64")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(thresh_128, cmap="gray")
    plt.title("Threshold T = 128")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(thresh_192, cmap="gray")
    plt.title("Threshold T = 192")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("threshold_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved threshold_comparison.png")


def main():
    print("=" * 60)
    print("Experiment No. 3: Thresholding")
    print("=" * 60)

    image_path = "cdg.png"
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

    print("Applying thresholding at T = 64, 128, 192...")
    thresh_64 = threshold_image(image, 64)
    thresh_128 = threshold_image(image, 128)
    thresh_192 = threshold_image(image, 192)

    display_results(image, thresh_128, thresh_64, thresh_192)

    cv2.imwrite("threshold_64.png", thresh_64)
    cv2.imwrite("threshold_128.png", thresh_128)
    cv2.imwrite("threshold_192.png", thresh_192)
    print("Saved threshold_64.png, threshold_128.png, threshold_192.png")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
