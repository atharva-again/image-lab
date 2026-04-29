# **Experiment 1**

## **Aim**: To study and plot basic mathematical signals used in digital image processing, including unit step, sine wave, cosine wave, exponential signal, square wave, and impulse function.

## **Software Used**: Python 3.12, NumPy, Matplotlib

## **Theory**: Basic signals are fundamental building blocks in signal and image processing. The **Unit Step Signal** u(n) is 0 for n < 0 and 1 for n >= 0. The **Sine and Cosine Waves** are periodic signals represented as A * sin(2 * pi * f * n + phi) and A * cos(2 * pi * f * n + phi). The **Exponential Signal** grows or decays exponentially as base^n. The **Square Wave** alternates between two levels periodically. The **Impulse (Delta) Function** is 1 at n = 0 and 0 elsewhere. Understanding these signals is essential before analyzing digital images.

## **Code**:

```python
import numpy as np
import matplotlib.pyplot as plt


def unit_step(n, shift=0):
    """Generate unit step signal u(n - shift)."""
    return np.where(n >= shift, 1, 0)


def sine_wave(n, amplitude=1, frequency=1, phase=0):
    """Generate sine wave: amplitude * sin(2 * pi * frequency * n + phase)."""
    return amplitude * np.sin(2 * np.pi * frequency * n + phase)


def cosine_wave(n, amplitude=1, frequency=1, phase=0):
    """Generate cosine wave: amplitude * cos(2 * pi * frequency * n + phase)."""
    return amplitude * np.cos(2 * np.pi * frequency * n + phase)


def exponential_signal(n, base=2):
    """Generate exponential signal: base ** n."""
    return np.power(base, n)


def square_wave(n, period=10):
    """Generate square wave with given period."""
    return np.where((n % period) < (period / 2), 1, -1)


def impulse(n, shift=0):
    """Generate impulse (delta) signal delta(n - shift)."""
    return np.where(n == shift, 1, 0)


def plot_signals():
    """Generate and plot all basic signals in a 3x2 grid."""
    n = np.arange(-10, 11, 1)
    n_continuous = np.linspace(-10, 10, 400)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Basic Signals", fontsize=16)

    # Unit Step
    ax = axes[0, 0]
    ax.stem(n, unit_step(n), basefmt=" ")
    ax.set_title("Unit Step Signal u(n)")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Sine Wave
    ax = axes[0, 1]
    ax.plot(n_continuous, sine_wave(n_continuous, frequency=0.2), color="blue")
    ax.set_title("Sine Wave")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Cosine Wave
    ax = axes[1, 0]
    ax.plot(n_continuous, cosine_wave(n_continuous, frequency=0.2), color="green")
    ax.set_title("Cosine Wave")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Exponential Signal
    ax = axes[1, 1]
    n_exp = np.arange(0, 11, 1)
    ax.stem(n_exp, exponential_signal(n_exp, base=2), basefmt=" ")
    ax.set_title("Exponential Signal (2^n)")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Square Wave
    ax = axes[2, 0]
    ax.plot(n_continuous, square_wave(n_continuous, period=4), color="red")
    ax.set_title("Square Wave")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Impulse Function
    ax = axes[2, 1]
    ax.stem(n, impulse(n), basefmt=" ")
    ax.set_title("Impulse (Delta) Function")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("basic_signals.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved as basic_signals.png")


def main():
    print("=" * 60)
    print("Experiment No. 1: Basic Signals")
    print("=" * 60)
    plot_signals()
    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## **Result**:

Fig 1: Basic Signals - Unit Step, Sine Wave, Cosine Wave, Exponential Signal, Square Wave, and Impulse Function plotted in a 3x2 grid

## **Conclusion**: This experiment successfully demonstrated the generation and visualization of fundamental mathematical signals used in digital image processing. Each signal was plotted using NumPy and Matplotlib, confirming their characteristic shapes and behaviors. These signals form the theoretical foundation for understanding more complex image transformations in the spatial and frequency domains.

# **Experiment 2**

## **Aim**: To compute the negative of a grayscale image manually using pixel-level operations.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: Image negation is a point processing operation that inverts the intensity values of an image. For an 8-bit grayscale image with intensity levels in the range [0, 255], the negative is computed using the formula: **s = L - 1 - r**, where L = 256 and r is the original pixel value. This transforms dark regions into bright regions and vice versa, producing a photographic negative effect. This operation is useful in medical imaging and enhancing white details in dark regions.

## **Code**:

```python
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
```

## **Result**:

Fig 2: Original grayscale image and its computed negative side by side

## **Conclusion**: The experiment successfully demonstrated image negation using the formula s = 255 - r. The output clearly shows the inversion of intensity values, with dark areas becoming bright and bright areas becoming dark. This point processing operation is computationally efficient and forms the basis for more complex image enhancement techniques.

# **Experiment 3**

## **Aim**: To apply manual thresholding to a grayscale image to convert it into a binary image at different threshold levels.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: Thresholding is a fundamental point processing technique used for image segmentation. It converts a grayscale image into a binary image by comparing each pixel value against a threshold T. The formula is: **s = 255 if r >= T else 0**, where r is the input pixel intensity. Pixels with intensity greater than or equal to the threshold are set to white (255), and others are set to black (0). Different threshold values (64, 128, 192) produce different segmentation results, allowing us to isolate objects of varying brightness from the background.

## **Code**:

```python
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
```

## **Result**:

Fig 3: Original image and thresholded outputs at T = 64, T = 128, and T = 192

## **Conclusion**: Thresholding was successfully applied at three different levels. Lower thresholds (T = 64) retain more bright regions, while higher thresholds (T = 192) retain only the brightest regions. T = 128 provides a balanced binary segmentation. This demonstrates how threshold selection directly impacts image segmentation quality and object isolation.

# **Experiment 4**

## **Aim**: To implement contrast stretching to increase the dynamic range of gray levels in a low-contrast grayscale image.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: Contrast stretching is a point processing technique that expands the range of intensity values in an image to cover the full available dynamic range [0, 255]. The linear transformation formula is: **s = ((r - r_min) / (r_max - r_min)) * 255**, where r_min and r_max are the minimum and maximum intensity values in the original image. This operation improves the visual quality of images where the intensity values are concentrated in a narrow range, making details more distinguishable. Histograms of the original and stretched images clearly show the redistribution of pixel values across the full range.

## **Code**:

```python
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
```

## **Result**:

Fig 4: Original image, its histogram, contrast-stretched image, and the corresponding stretched histogram along with the transformation function

## **Conclusion**: Contrast stretching successfully expanded the dynamic range of the input image. The histograms clearly show that the original image had pixel values concentrated in a narrow range, while the output image utilizes the full [0, 255] range. This improves visual quality and makes subtle details more apparent, demonstrating the effectiveness of linear contrast enhancement.

# **Experiment 5**

## **Aim**: To decompose a grayscale image into its 8 bit planes and analyze the contribution of each bit to the overall image.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: Bit plane slicing is a technique that decomposes an image into its individual binary bit planes. In an 8-bit grayscale image, each pixel value is represented by 8 bits. The **Most Significant Bit (MSB)** plane (bit 7) contains the most coarse intensity information, while the **Least Significant Bit (LSB)** plane (bit 0) contains fine detail and noise. By extracting each bit plane using the formula: **plane(i, j) = 255 if (pixel >> bit) & 1 else 0**, we can visualize the contribution of each bit. Higher bit planes are more visually significant, and reconstructing an image using only the higher planes can reduce storage while preserving important features.

## **Code**:

```python
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
```

## **Result**:

Fig 5: Original image and all 8 bit planes from bit 0 (LSB) to bit 7 (MSB)

## **Conclusion**: Bit plane slicing successfully decomposed the image into its 8 binary components. The higher bit planes (6 and 7) contain the majority of the visually significant information, while the lower bit planes (0 and 1) appear mostly as noise-like patterns. This experiment demonstrates that storing or transmitting only the higher bit planes can significantly reduce data size while preserving the essential structure of the image.

# **Experiment 6**

## **Aim**: To implement histogram equalization manually to enhance the contrast of a grayscale image.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: Histogram equalization is a global contrast enhancement technique that redistributes pixel intensities to utilize the full dynamic range more uniformly. The process involves: (1) Computing the histogram manually by counting occurrences of each intensity value, (2) Computing the Cumulative Distribution Function (CDF), (3) Normalizing the CDF using the formula: **s = round(((CDF(r) - CDF_min) / (Total_Pixels - CDF_min)) * 255)**, and (4) Mapping each original pixel value to its new equalized value. This transformation spreads out frequent intensity values and compresses infrequent ones, resulting in an image with more balanced contrast and a roughly uniform histogram distribution.

## **Code**:

```python
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
```

## **Result**:

Fig 6: Original image with its histogram and CDF, and the equalized image with its histogram and CDF

## **Conclusion**: Histogram equalization successfully redistributed the pixel intensity values across the full dynamic range. The original histogram was concentrated in specific regions, while the equalized histogram shows a more uniform spread. The CDF plots confirm the transformation from a steep curve to a more linear cumulative distribution. This technique effectively enhances image contrast without requiring parameter tuning.

# **Experiment 7**

## **Aim**: To implement image magnification (zooming) using pixel replication and bilinear interpolation techniques.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: Image magnification increases the spatial resolution of an image. **Zooming by Replication** (also known as nearest-neighbor or pixel duplication) replaces each pixel with a factor x factor block of identical pixels. This is simple but produces blocky, pixelated results. **Zooming by Interpolation** (bilinear interpolation) computes new pixel values by taking a weighted average of the four nearest neighboring pixels in the original image. The interpolation formula uses fractional distances (dx, dy) from the surrounding pixels: value = (1-dx)(1-dy)*Q11 + dx(1-dy)*Q21 + (1-dx)dy*Q12 + dx*dy*Q22. Bilinear interpolation produces smoother, more visually appealing results but requires more computation.

## **Code**:

```python
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
```

## **Result**:

Fig 7: Original image, 2x zoom by replication, and 2x zoom by bilinear interpolation

## **Conclusion**: Both magnification techniques successfully increased the image size by a factor of 2. Replication produced a blocky appearance with visible pixel boundaries, while bilinear interpolation created smoother transitions between pixels. This experiment highlights the trade-off between computational simplicity and visual quality in image resizing operations. Bilinear interpolation is generally preferred for its superior visual results.

# **Experiment 8**

## **Aim**: To apply low pass filtering in the spatial domain using mean (averaging) filters of different sizes to smooth an image.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: Low pass filtering is a spatial domain technique used for image smoothing and noise reduction. A **Mean (Averaging) Filter** replaces each pixel with the average of its neighboring pixels within a kernel window. The convolution operation is performed as: **output(i, j) = sum(kernel * region)**, where the kernel is a matrix of ones divided by the kernel size (e.g., 1/9 for 3x3). Zero padding is applied at the image boundaries to maintain dimensions. Larger kernels (e.g., 5x5) produce stronger smoothing effects but may blur important edges and details. This filter is effective against Gaussian noise but less effective against impulse noise.

## **Code**:

```python
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
```

## **Result**:

Fig 8: Original image, low pass filtered with 3x3 mean kernel, and low pass filtered with 5x5 mean kernel

## **Conclusion**: Low pass filtering successfully smoothed the input image. The 3x3 filter provided moderate smoothing while preserving more edge details, whereas the 5x5 filter produced stronger smoothing with more noticeable blurring. This demonstrates the direct relationship between kernel size and the degree of smoothing, highlighting the trade-off between noise reduction and edge preservation in spatial domain filtering.

# **Experiment 9**

## **Aim**: To add salt and pepper noise to a grayscale image and then apply a median filter to remove the noise.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: **Salt and Pepper Noise** is a type of impulse noise where random pixels are set to either maximum intensity (salt = 255, white) or minimum intensity (pepper = 0, black). This noise is commonly caused by sharp disturbances in the image signal. The **Median Filter** is a nonlinear spatial filter that replaces each pixel with the median value of its neighborhood. Unlike mean filters, median filters are excellent at removing impulse noise while preserving edges because the median is robust to extreme values. The filtering process involves: extracting the neighborhood window, sorting the pixel values, and selecting the middle value as the new pixel intensity.

## **Code**:

```python
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
```

## **Result**:

Fig 9: Original image, image corrupted with 5% salt and pepper noise, and the noise-reduced image after applying 3x3 median filter

## **Conclusion**: Salt and pepper noise was successfully added to the image, and the median filter effectively removed most of the noise while preserving edge sharpness. Unlike linear filters such as the mean filter, the median filter's nonlinear nature makes it particularly suitable for impulse noise removal. The results clearly show that the median filter outperforms averaging filters for this type of noise, maintaining image details that would otherwise be blurred.

# **Experiment 10**

## **Aim**: To detect edges in a grayscale image using Sobel operators and perform image segmentation by thresholding the edge magnitude.

## **Software Used**: Python 3.12, NumPy, OpenCV, Matplotlib

## **Theory**: Edge detection identifies boundaries between regions of different intensity in an image. The **Sobel Operator** uses two 3x3 kernels to approximate the image gradient in the horizontal (Gx) and vertical (Gy) directions. The kernels are: Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] and Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]. The **edge magnitude** is computed as: **M = sqrt(Gx^2 + Gy^2)**. After computing the edge magnitude, **image segmentation** is performed by thresholding: pixels with magnitude above the threshold are classified as edges (255), and others as background (0). This separates objects from the background based on their intensity discontinuities.

## **Code**:

```python
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
```

## **Result**:

Fig 10: Original image, Sobel Gx gradient, Sobel Gy gradient, edge magnitude, segmented edges, and edge magnitude histogram

## **Conclusion**: Edge detection using Sobel operators successfully identified intensity discontinuities in the image. The horizontal and vertical gradient components (Gx and Gy) highlighted edges in their respective orientations. The edge magnitude image combined both directions to show all prominent edges. Thresholding the magnitude at T = 50 produced a clean binary segmentation of the edges. This experiment demonstrates the fundamental role of derivative-based filters in image segmentation and feature extraction for higher-level computer vision tasks.
