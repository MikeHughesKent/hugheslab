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

### display
Quick-display helpers for interactive use:
- `qshow` — display a 2D image in a matplotlib figure with sensible defaults
- `cshow` — display a complex image as side-by-side amplitude and phase plots
- `qplot` — display a 1D line plot with sensible defaults

### fig_maker
Publication-quality multi-panel figure creation:
- `multi_im` — lay out a grid of images with automatic lettering and labels
- `ax_zoom` / `img_zoom` — add a zoomed inset to an existing axes or image
- `scalebar` — draw a scalebar on an image axes
- `mm2pix` / `get_zoom` — coordinate conversion and zoom helpers
gi
### im_tools
Image processing and analysis utilities:
- **Cropping:** `crop_image`, `crop_zero`, `crop_zero_box`, `extract_central`
- **Bit-depth conversion:** `to8bit`, `to16bit`
- **Scaling:** `log_scale_image`, `log_scale_min`
- **Channel operations:** `average_channels`, `max_channels`, `mean_channels`
- **Registration:** `get_shifts`, `find_shift`
- **Stack utilities:** `filter_stack`, `resize_stack`, `condense_stack`
- **Misc:** `rect_to_pol`, `hamming_window`

### im_file_tools
Loading and saving images and image stacks:
- **Loading:** `load_image`, `load_stack`, `load_folder_images`
- **Saving single images:** `save_image8`, `save_image16`, `save_image_colour`, `save_image8_scaled`, `save_image16_scaled`
- **Saving stacks:** `save_tif_stack`, `stack_to_folder`, `folder_to_tif_stack`, `tif_stack_to_folder`
- **Video:** `save_video`

### im_analysis
Quantitative image analysis:
- `profile_line` — extract a line profile from an image
- `fwhm` — calculate full-width at half-maximum of a profile
- `profile_gaussian_fit` / `gaussian_fit` — fit a Gaussian to a profile
- `GHT` — generalised histogram transform for contrast enhancement


### file_tools
- `file_sort` — sort a list of filenames into natural numeric order

### spot_utilities
Utilities for working with images and stacks of focused spots (e.g. laser).
- `find_spot` — find the location of the brightest pixel in an image
- `crop_spot_image` — crop an image to be centred on the brightest spot
- `plot_spots` — display a set of spot images in a grid
- `spot_profile` — extract the vertical, horizontal, and radial profiles of a spot
- `analyse_spot_images` — analyse a set of spot images and return their FWHM values
- `fwhm` — calculate the full-width at half-maximum of a profile
- `fit_gaussian2d` — fit a 2D Gaussian to a spot image


## Requirements

- matplotlib
- numpy
- opencv-python
- Pillow
