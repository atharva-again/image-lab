"""
Experiment No. 5: Bit Plane Slicing
Digital Image Processing Lab

This script decomposes a grayscale image into its 8 bit planes.
Each bit plane shows the contribution of a specific bit to the image.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def bit_plane_slice(image):
    """Extract all 8 bit planes from a grayscale image."""
    bit_planes = []
    for bit in range(8):
        plane = np.zeros_like(image)
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                plane[i, j] = 255 if (image[i, j] >> bit) & 1 else 0
        bit_planes.append(plane)
    return bit_planes


def display_results(original, bit_planes):
    """Display original and all 8 bit planes."""
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    for bit in range(8):
        plt.subplot(3, 3, bit + 2)
        plt.imshow(bit_planes[bit], cmap="gray")
        plt.title(f"Bit Plane {bit}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("bit_planes.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved bit_planes.png")


def main():
    print("=" * 60)
    print("Experiment No. 5: Bit Plane Slicing")
    print("=" * 60)

    image_path = "feyd.png"
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

    print("Extracting bit planes...")
    bit_planes = bit_plane_slice(image)

    display_results(image, bit_planes)

    for bit in range(8):
        cv2.imwrite(f"bit_plane_{bit}.png", bit_planes[bit])
    print("Saved individual bit_plane_*.png files")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
