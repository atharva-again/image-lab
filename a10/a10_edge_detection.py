"""
Experiment No. 10: Image Segmentation using Edge Detection
Digital Image Processing Lab

This script implements edge detection using Sobel operators manually,
then segments the image by thresholding the edge magnitude.
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

    return output


def sobel_edge_detection(image):
    """Apply Sobel operators manually to detect edges."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = apply_filter(image, sobel_x)
    gy = apply_filter(image, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return gx, gy, magnitude


def segment_by_edges(magnitude, threshold=50):
    """Segment image by thresholding edge magnitude."""
    output = np.zeros_like(magnitude)
    rows, cols = magnitude.shape
    for i in range(rows):
        for j in range(cols):
            if magnitude[i, j] >= threshold:
                output[i, j] = 255
            else:
                output[i, j] = 0
    return output


def display_results(original, gx, gy, magnitude, segmented):
    """Display original, gradients, magnitude, and segmented images."""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(np.abs(gx), cmap="gray")
    plt.title("Sobel Gx")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(gy), cmap="gray")
    plt.title("Sobel Gy")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(magnitude, cmap="gray")
    plt.title("Edge Magnitude")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(segmented, cmap="gray")
    plt.title("Segmented (Edges)")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.hist(magnitude.ravel(), bins=256, range=(0, 256), color="gray")
    plt.title("Magnitude Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("edge_detection_segmentation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved edge_detection_segmentation.png")


def main():
    print("=" * 60)
    print("Experiment No. 10: Edge Detection & Segmentation")
    print("=" * 60)

    image_path = "csmvs.png"
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

    print("Applying Sobel edge detection...")
    gx, gy, magnitude = sobel_edge_detection(image)

    print("Segmenting by edge magnitude threshold...")
    segmented = segment_by_edges(magnitude, threshold=50)

    display_results(image, gx, gy, magnitude, segmented)

    cv2.imwrite("edge_magnitude.png", magnitude)
    cv2.imwrite("segmented_edges.png", segmented)
    print("Saved edge_magnitude.png, segmented_edges.png")

    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
