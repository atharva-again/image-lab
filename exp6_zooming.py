"""
Experiment No. 6: Zooming by Interpolation and Replication
Digital Image Processing Lab

This script implements two zooming techniques:
Part A: Zooming by Replication (Pixel Duplication)
Part B: Zooming by Interpolation (Linear Interpolation)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def zoom_by_replication(image):
    """
    Part A: Zoom by Replication

    Logic: Transform an NxN image into a 2Nx2N image by duplicating
    each pixel and then duplicating each row.

    Parameters:
        image: Input grayscale image (2D numpy array)

    Returns:
        zoomed_image: 2x zoomed image using replication
    """
    rows, cols = image.shape
    zoomed = np.zeros((2 * rows, 2 * cols), dtype=image.dtype)

    for i in range(rows):
        for j in range(cols):
            pixel_value = image[i, j]
            zoomed[2 * i, 2 * j] = pixel_value
            zoomed[2 * i, 2 * j + 1] = pixel_value
            zoomed[2 * i + 1, 2 * j] = pixel_value
            zoomed[2 * i + 1, 2 * j + 1] = pixel_value

    return zoomed


def zoom_by_interpolation(image):
    """
    Part B: Zoom by Interpolation

    Logic: For a 2x zoom, take the average of two adjacent pixels
    along the rows and place the result between them. Then, perform
    the same operation along the columns.

    Formula: P_new = (P1 + P2) / 2

    Parameters:
        image: Input grayscale image (2D numpy array)

    Returns:
        zoomed_image: 2x zoomed image using linear interpolation
    """
    rows, cols = image.shape

    row_expanded = np.zeros((rows, 2 * cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            row_expanded[i, 2 * j] = image[i, j]
            if j < cols - 1:
                row_expanded[i, 2 * j + 1] = (
                    float(image[i, j]) + float(image[i, j + 1])
                ) / 2.0
            else:
                row_expanded[i, 2 * j + 1] = image[i, j]

    zoomed = np.zeros((2 * rows, 2 * cols), dtype=np.float32)
    for j in range(2 * cols):
        for i in range(rows):
            zoomed[2 * i, j] = row_expanded[i, j]
            if i < rows - 1:
                zoomed[2 * i + 1, j] = (
                    row_expanded[i, j] + row_expanded[i + 1, j]
                ) / 2.0
            else:
                zoomed[2 * i + 1, j] = row_expanded[i, j]

    return zoomed.astype(image.dtype)


def display_results(original, replication_zoom, interpolation_zoom):
    """
    Display the original image and both zoomed versions using matplotlib subplots.

    Parameters:
        original: Original grayscale image
        replication_zoom: Zoomed image using replication
        interpolation_zoom: Zoomed image using interpolation
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title(f"Original Image\n{original.shape[0]}x{original.shape[1]}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(replication_zoom, cmap="gray")
    plt.title(
        f"Zoom by Replication\n{replication_zoom.shape[0]}x{replication_zoom.shape[1]}"
    )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(interpolation_zoom, cmap="gray")
    plt.title(
        f"Zoom by Interpolation\n{interpolation_zoom.shape[0]}x{interpolation_zoom.shape[1]}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("zoom_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 60)
    print("Zoom Comparison Results")
    print("=" * 60)
    print(
        f"Original Image Size:        {original.shape[1]} x {original.shape[0]} pixels"
    )
    print(
        f"Replication Zoom Size:      {replication_zoom.shape[1]} x {replication_zoom.shape[0]} pixels"
    )
    print(
        f"Interpolation Zoom Size:    {interpolation_zoom.shape[1]} x {interpolation_zoom.shape[0]} pixels"
    )
    print("=" * 60)


def main():
    print("=" * 60)
    print("Experiment No. 6: Zooming by Interpolation and Replication")
    print("=" * 60)

    image_path = "cameraman.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Attempting to load alternative image...")

        alternative_paths = ["cameraman.png", "lenna.jpg", "lena.jpg", "test_image.jpg"]
        for alt_path in alternative_paths:
            image = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                print(f"Successfully loaded: {alt_path}")
                break

        if image is None:
            print("Error: Could not load any image.")
            image = np.zeros((100, 100), dtype=np.uint8)
            image[25:75, 25:75] = 128
            image[40:60, 40:60] = 255
            cv2.imwrite("sample_test_image.png", image)
            print("Created sample_test_image.png for testing.")
    else:
        print(f"Successfully loaded: {image_path}")

    print(f"\nInput image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Input image dtype: {image.dtype}")

    print("\n[Part A] Performing Zoom by Replication...")
    replication_result = zoom_by_replication(image)

    print("\n[Part B] Performing Zoom by Interpolation...")
    interpolation_result = zoom_by_interpolation(image)

    display_results(image, replication_result, interpolation_result)

    cv2.imwrite("zoom_replication.png", replication_result)
    cv2.imwrite("zoom_interpolation.png", interpolation_result)

    print("\nOutput images saved:")
    print("  - zoom_replication.png")
    print("  - zoom_interpolation.png")
    print("  - zoom_comparison.png (combined view)")

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
