"""
Experiment No. 2: Negative Image
Digital Image Processing Lab

This script computes the negative of a grayscale image manually.
Formula: s = L - 1 - r, where L = 256 for 8-bit images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def negative_image(image):
    """Compute the negative of a grayscale image manually."""
    return 255 - image


def display_results(original, negative):
    """Display original and negative images side by side."""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(negative, cmap="gray")
    plt.title("Negative Image")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("negative_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved negative_comparison.png")


def main():
    print("=" * 60)
    print("Experiment No. 2: Negative Image")
    print("=" * 60)

    image_path = "creation_of_adam.png"
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

    print("Computing negative image...")
    negative = negative_image(image)

    display_results(image, negative)

    cv2.imwrite("negative_output.png", negative)
    print("Saved negative_output.png")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
