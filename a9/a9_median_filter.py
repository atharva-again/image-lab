"""
Experiment No. 9: Median Filtering after Salt and Pepper Noise
Digital Image Processing Lab

This script:
1. Adds salt and pepper noise to a grayscale image manually
2. Applies a median filter manually using a sliding window
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_salt_and_pepper(image, amount=0.05):
    """Add salt and pepper noise manually to a grayscale image."""
    noisy = image.copy()
    rows, cols = image.shape
    num_salt = int(amount * rows * cols * 0.5)
    num_pepper = int(amount * rows * cols * 0.5)

    for _ in range(num_salt):
        i = np.random.randint(0, rows)
        j = np.random.randint(0, cols)
        noisy[i, j] = 255

    for _ in range(num_pepper):
        i = np.random.randint(0, rows)
        j = np.random.randint(0, cols)
        noisy[i, j] = 0

    return noisy


def median_filter(image, size=3):
    """Apply median filter manually with zero padding."""
    rows, cols = image.shape
    pad = size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    output = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            region = padded[i : i + size, j : j + size]
            flat = region.flatten()
            flat.sort()
            output[i, j] = flat[len(flat) // 2]

    return output


def display_results(original, noisy, filtered):
    """Display original, noisy, and filtered images."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(noisy, cmap="gray")
    plt.title("Salt & Pepper Noise")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(filtered, cmap="gray")
    plt.title("Median Filtered (3x3)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("median_filter_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved median_filter_comparison.png")


def main():
    print("=" * 60)
    print("Experiment No. 9: Median Filtering")
    print("=" * 60)

    image_path = "fashion_is_my_second_favorite_f_word.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Creating synthetic test image...")
        image = np.zeros((256, 256), dtype=np.uint8)
        image[80:176, 80:176] = 255
        cv2.imwrite("synthetic_test.png", image)
        print("Created synthetic_test.png")
    else:
        print(f"Loaded image: {image_path}")

    print("Adding salt and pepper noise...")
    noisy = add_salt_and_pepper(image, amount=0.05)

    print("Applying 3x3 median filter...")
    filtered = median_filter(noisy, size=3)

    display_results(image, noisy, filtered)

    cv2.imwrite("noisy_salt_pepper.png", noisy)
    cv2.imwrite("median_filtered.png", filtered)
    print("Saved noisy_salt_pepper.png, median_filtered.png")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
