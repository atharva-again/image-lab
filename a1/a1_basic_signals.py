"""
Experiment No. 1: Basic Signals
Digital Image Processing Lab

This script generates and plots various basic signals:
- Unit Step
- Sine Wave
- Cosine Wave
- Exponential Signal
- Square Wave
- Impulse Function
"""

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
