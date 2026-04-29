"""
Experiment No. 8: Low Pass Filtering in Spatial Domain
Digital Image Processing Lab

This script applies a 3x3 averaging (mean) filter manually using convolution.
A low pass filter smooths the image by averaging neighboring pixels.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def apply_filter(image, kernel):
    """Apply a convolution filter manually with zero padding."""
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape
    pad_r = k_rows // 2
    pad_c = k_cols // 2

    padded = np.pad(
        image, ((pad_r, pad_r), (pad_c, pad_c)), mode="constant", constant_values=0
    )
    output = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            region = padded[i : i + k_rows, j : j + k_cols]
            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


def low_pass_filter(image, size=3):
    """Apply a mean (averaging) low pass filter of given size."""
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return apply_filter(image, kernel)


def display_results(original, filtered_3x3, filtered_5x5):
    """Display original and filtered images."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_3x3, cmap="gray")
    plt.title("Low Pass (3x3 Mean)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(filtered_5x5, cmap="gray")
    plt.title("Low Pass (5x5 Mean)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("low_pass_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved low_pass_comparison.png")


def main():
    print("=" * 60)
    print("Experiment No. 8: Low Pass Filtering")
    print("=" * 60)

    image_path = "delhi_metro.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Creating synthetic test image...")
        image = np.zeros((256, 256), dtype=np.uint8)
        image[80:176, 80:176] = 255
        np.random.seed(0)
        noise = np.random.randint(0, 50, size=(256, 256), dtype=np.uint8)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite("synthetic_noisy.png", image)
        print("Created synthetic_noisy.png")
    else:
        print(f"Loaded image: {image_path}")

    print("Applying 3x3 low pass filter...")
    filtered_3x3 = low_pass_filter(image, size=3)

    print("Applying 5x5 low pass filter...")
    filtered_5x5 = low_pass_filter(image, size=5)

    display_results(image, filtered_3x3, filtered_5x5)

    cv2.imwrite("low_pass_3x3.png", filtered_3x3)
    cv2.imwrite("low_pass_5x5.png", filtered_5x5)
    print("Saved low_pass_3x3.png, low_pass_5x5.png")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
