# -*- coding: utf-8 -*-
"""
Unit tests for display helper functions.

"""

import context
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hugheslab.display import qshow, cshow, qplot


class TestDisplay(unittest.TestCase):

    def tearDown(self):
        # Keep tests isolated by closing all figures after each test.
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_qshow_returns_figure_and_image(self, mock_show):
        img = np.random.random((32, 48))
        expected_figsize = (120 / 25.4, 80 / 25.4)

        fig, im = qshow(
            img,
            title="test",
            axes=("x", "y"),
            cmap="gray",
            dpi=120,
            figsize=(5, 5),
            figsizem=(120, 80),
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(im)
        self.assertAlmostEqual(fig.get_size_inches()[0], expected_figsize[0], places=3)
        self.assertAlmostEqual(fig.get_size_inches()[1], expected_figsize[1], places=3)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_cshow_returns_figure_and_two_axes(self, mock_show):
        real = np.random.random((40, 40))
        imag = np.random.random((40, 40))
        img = real + 1j * imag

        fig, (ax1, ax2) = cshow(
            img,
            dpi=110,
            figsize=(8, 4),
            phase_cmap="twilight",
            amp_cmap="gray",
            title="complex",
        )

        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(ax1.get_title(), "Amplitude")
        self.assertEqual(ax2.get_title(), "Phase")
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_qplot_returns_figure_and_line(self, mock_show):
        y = np.linspace(0, 1, 25)
        expected_figsize = (100 / 25.4, 60 / 25.4)

        fig, lines = qplot(
            y,
            title="line",
            axes=("x", "y"),
            dpi=100,
            figsize=(5, 5),
            figsizem=(100, 60),
        )

        self.assertIsNotNone(fig)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(fig.get_size_inches()[0], expected_figsize[0], places=3)
        self.assertAlmostEqual(fig.get_size_inches()[1], expected_figsize[1], places=3)
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
