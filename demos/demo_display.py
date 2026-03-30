# -*- coding: utf-8 -*-
"""
Demo script for display utilities in hugheslab.display.

Runs examples for:
- qshow: quick image display
- cshow: amplitude/phase display for complex images
- qplot: quick line plotting
"""

import os
import sys

import numpy as np

# Allow running this demo directly from the repository without installation.
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hugheslab.display import qshow, cshow, qplot


def make_test_image(size=256):
    """Create a smooth 2D test pattern image."""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    xx, yy = np.meshgrid(x, y)
    img = np.exp(-(xx**2 + yy**2)) * np.cos(4 * xx) * np.sin(4 * yy)
    return img


def make_complex_image(size=256):
    """Create a complex field with varying amplitude and phase."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    amp = np.exp(-4 * (xx**2 + yy**2))
    phase = 8 * np.pi * np.sqrt(xx**2 + yy**2)
    field = amp * np.exp(1j * phase)
    return field


def make_plot_data(n=300):
    """Create simple x/y data for qplot demonstration."""
    x = np.linspace(0, 4 * np.pi, n)
    y = 0.7 * np.sin(x) + 0.3 * np.sin(3 * x)
    return x, y


def main():
    image = make_test_image()
    qshow(
        image,
        title="qshow demo: test image",
        axes=("x (px)", "y (px)"),
        cmap="gray",
        dpi=150,
        figsizem=(200, 200),
    )

    complex_image = make_complex_image()
    cshow(
        complex_image,
        dpi=150,
        figsizem=(200, 100),
        phase_cmap="twilight",
        amp_cmap="gray",
        title="cshow demo: amplitude and phase",
    )

    x, y = make_plot_data()
    qplot(
        x,
        y,
        title="qplot demo: synthetic waveform",
        axes=("x", "y"),
        dpi=150,
        figsizem=(200, 160),
    )


if __name__ == "__main__":
    main()
