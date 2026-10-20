# HughesLab

A collection of utilities useful for an (optical) imaging research lab.

All functions work with images and image-like data stored as numpy arrays.

Documentation is on [Readthedocs](https://hugheslab.readthedocs.io/en/latest/).

## Installation
Download from github and run:
```
pip install -r requirements.txt
```

## Modules

### Image Display and Plotting
Quick-display helpers for interactive use:
- `qshow` : display a 2D image in a matplotlib figure with sensible defaults
- `cshow` : display a complex image as side-by-side amplitude and phase plots
- `qplot` : display a 1D line plot with sensible defaults

### Figure Generator
Tools to make publication-style multi-panel figure from images:
- `multi_im` : lay out a grid of images with automatic lettering and labels
- `ax_zoom` / `img_zoom` : add a zoomed inset to an existing axes or image
- `scalebar` : draw a scalebar on an image axes

### Image Tools
Image processing and analysis utilities:
- **Cropping:** `crop_image`, `crop_zero`, `crop_zero_box`, `extract_central`
- **Bit-depth conversion:** `to8bit`, `to16bit`
- **Scaling:** `log_scale_image`, `log_scale_min`
- **Channel operations:** `average_channels`, `max_channels`, `mean_channels`
- **Registration:** `get_shifts`, `find_shift`
- **Stack utilities:** `filter_stack`, `resize_stack`, `condense_stack`
- **Misc:** `radial_profile`, `rect_to_pol`, `hamming_window`

### Image Loading and Saving
Loading and saving images and image stacks:
- **Loading:** `load_image`, `load_stack`, `load_folder_images`
- **Saving single images:** `save_image8`, `save_image16`, `save_image_colour`, `save_image8_scaled`, `save_image16_scaled`
- **Saving stacks:** `save_tif_stack`, `stack_to_folder`, `folder_to_tif_stack`, `tif_stack_to_folder`
- **Video:** `save_video`

### Image Analysis
Quantitative image analysis:
- `profile_line` : extract a line profile from an image
- `fwhm` : calculate full-width at half-maximum of a profile
- `profile_gaussian_fit` / `gaussian_fit` : fit a Gaussian to a profile
- `GHT` : generalised histogram transform for contrast enhancement


## Requirements

- matplotlib
- numpy
- opencv-python
- Pillow
