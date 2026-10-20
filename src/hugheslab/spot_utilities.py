# -*- coding: utf-8 -*-
"""
Utilities for analysing focused spots. 

"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, curve_fit
from scipy.ndimage import rotate

import cv2 as cv


def spot_profile(img, plot_size, background_removal = True, normalise = True, rotate_spot = False):
    """
    Produces radial average, vertical and horizontal profiles through the centre of
    a beam spot somewhere in an image that is stored as a 2D numpy array.
    
    Arguments:
        img       : numpy.ndarray
                    2D numpy array representing the image.
        plot_size : int
                    The number of pixels to include on either side of the centre.
              
    Optional Keyword Arguments:       
        background_removal : bool
                             If True (default), the minimum value of the profile is subtracted
                             from the profile.
        normalise          : bool
                             If True (default), the profile is normalised so that max value is 1.

    Returns:
        tuple: A tuple containing three 1D numpy arrays, the radial average, 
               vertical and horizontal profiles through the center of the spot.
    """
   
    if rotate_spot is True:
        angle = spot_orientation(img, background=None, threshold=0)[0]
        img = rotate(img, 180 + angle, reshape=True)  
    
    # Find the centre of the spot in the image
    centre_x, centre_y = find_spot(img, filter_size = 6)
    
    # Define the start and end indices for the profiles
    y_min = max(centre_y - plot_size, 1)
    y_max = min(centre_y + plot_size + 1, img.shape[0])
    x_min = max(centre_x - plot_size, 1)
    x_max = min(centre_x + plot_size + 1, img.shape[1])

    # Extract the vertical and horizontal profiles
    vertical_profile = img[y_min:y_max, centre_x]  # Vertical profile centered at center_x
    horizontal_profile = img[centre_y, x_min:x_max]  # Horizontal profile centered at center_y
    
    # Extract a radial average profile
    img_crop = img[y_min-1:y_max+1, x_min-1:x_max+1]
    average_profile = radial_profile(img_crop, (plot_size+1, plot_size+1))

    # Append a reversed version of the average profile to itself to make it symmetric
    average_profile = np.append(average_profile[:0:-1],average_profile)
    
    if background_removal:
        average_profile = average_profile - np.min(average_profile)
        vertical_profile = vertical_profile - np.min(vertical_profile)
        horizontal_profile = horizontal_profile - np.min(horizontal_profile)

    if normalise:
        average_profile = average_profile / np.max(average_profile)
        vertical_profile = vertical_profile / np.max(vertical_profile)
        horizontal_profile = horizontal_profile / np.max(horizontal_profile)

    return average_profile, vertical_profile, horizontal_profile


def find_spot(img, filter_size = 3):
    """ Find the centre of a laser spot in an image by looking for
    brightest pixels after smoothing. An optional Guassian smoothing
    filter is applied to reduce the impact of spurious bright pixels.

    Arguments:
        img         : numpy.ndarray
                      2D numpy array representing the image.
    
    Optional Keyword Arguments:
        filter_size : int
                      size (sigma) of the Gaussian smoothing filter to 
                      apply to the image as part of the detection process. 
                      Default is 3. Set to None for no filter.
    
    Returns:
        tuple: A tuple containing the x and y coordinates of the spot's centre.
    """
    
     # Apply smoothing filter to image
    if filter_size is not None:
        img = gaussian_filter(img, filter_size)

    # Find the center of the spot in the image
    centre_y, centre_x = np.unravel_index(np.argmax(img), img.shape)

    return centre_x, centre_y


def gaussian_filter(img, filter_size, kernel_size = None):
    """ Applies 2D Gaussian filter to image. By default
    the kernel size is 4 times the filter_size (sigma).
    
    Returns filtered image as numpy array.
    
    Arguments:        
        img          : np.ndarray
                       input image as 2D/3D numpy array
        filter_size  : float
                       sigma of Gaussian filter
   
    Optional Keyword Arguments:   
        kernel_size   : int
                        size of convolution kernal. Default is None, in which
                        case it is 4 times filter_size. Will be rounded up to nearest
                        odd integer.

    """
    
    # Kernal size needs to be larger than sigma
    if kernel_size is None:
        kernel_size = round(filter_size * 4)             
    
    # Force kernel size to be odd
    kernel_size = kernel_size + 1 - kernel_size % 2       
    
    img_filt = cv.GaussianBlur(img,(kernel_size, kernel_size), filter_size)
    
    return img_filt


def radial_profile(img, centre):
    """ Produces angular-averaged radial profile through image img centred on
    centre, a tuple of (x_centre, y_centre).
    
    Returns radial profile as 1D numpy array

    Arguments:
        img    : ndarray
                 input image as 2D numpy array
        centre : tuple of (int, int)
                 centre point for radial profile, tuple of (x,y)  
    Returns:
        ndarray: 1D radial average profile             
    """
    
    y, x = np.indices((img.shape))
    
    # Calculate radial distance of each pixel from the centre
    r = np.round(np.sqrt((x - centre[1])**2 + (y - centre[0])**2))
    r = r.astype(int)

    # Make hisotgram of pixel values at each radius
    pixel_sums = np.bincount(r.ravel(), weights = img.ravel())
    pixel_counts = np.bincount(r.ravel())
   
    # Remove bins with no entries
    pixel_sums= pixel_sums[pixel_counts > 0]
    pixel_counts = pixel_counts[pixel_counts > 0]

    # The radial profile is the sum of pixel values at each radius divided 
    # by the number of pixels at that radius
    radialprofile = pixel_sums  / pixel_counts
    
    radialprofile = radialprofile[:int(np.min(img.shape) / 2)]
    
    return radialprofile 


def analyse_spot_images(image_set, plot_size):
    """
    Analyses a set of images, calculates the FWHM of the vertical, horizontal and 
    radial average profiles for each, and returns the FWHM values as an array.

    Arguments:
        image_set : numpy.ndarray or list of numpy.ndarray
                    3D numpy array or list of 2D numpy arrays representing the images.
        plot_size : int
                    The number of pixels to include on either side of the centre.
    
    Returns:
        numpy.ndarray: Array containing the FWHM of the vertical profile for each image.
    
    """
    fwhm_values_radial = []
    fwhm_values_h = []
    fwhm_values_v = []
    

    for img in image_set:
        radial, h, v = spot_profile(img, plot_size)  
   
        fwhm_value_radial = fwhm(radial) 
        fwhm_value_h = fwhm(h) 
        fwhm_value_v = fwhm(v) 

        
        # Store the FWHM in the lists        
        fwhm_values_radial.append(fwhm_value_radial)
        fwhm_values_h.append(fwhm_value_h)
        fwhm_values_v.append(fwhm_value_v)


    # Return FWHM values as a numpy array
    return np.array(fwhm_values_radial), np.array(fwhm_values_h), np.array(fwhm_values_v)



def fwhm(profile, upsample = 100):
    """
    Calculates the Full Width at Half Maximum (FWHM) for a 1D profile by measuring
    where the profile first rises above the half maximum and when it subsequntly first
    fall back below the half maximum.

    Arguments:
        profile : numpy.ndarray
                  1D array representing the profile.
            
    Optional Keyword Arguments:
        upsample : int
                   factor to upsample by prior to calculating fwhm

    Returns:
        float: The FWHM of the profile.
    """
    
    # Remove background level
    profile = profile - np.min(profile)

    half_max = np.max(profile) / 2
    
    # Upsample the profile to get a more accurate FWHM
    indices_up = np.linspace(0, len(profile) - 1, len(profile) * upsample)
    profile_up = interp1d(np.arange(len(profile)), profile, kind='cubic')
    profile = profile_up(indices_up)

    # Starting from the maxmum, find the first point where the profile goes below half max on either side
    left_point = np.where(profile > half_max)[0][0]
    right_point = np.where(profile > half_max)[0][-1]

    # Calculate the FWHM as the difference between the two crossing points
    fwhm = indices_up[right_point] - indices_up[left_point]
   
    return fwhm


def plot_spots(spots, rows = 1, dpi = 300, fig_width = 4, labels = None):
    """ Produces a simple figure showing all the images of focus spots
    in a stack.

    Arguments:
        spots : np.ndarray or list of np.ndarray
                images to display
        rows :  number of rows of images in plot (default = 1)
        dpi  :  figure resolution
        fig_width : figure size in in inches

    """
        
    cols = math.ceil(len(spots) /rows)
    
    fig_height = fig_width * rows / cols
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, dpi = dpi, figsize = (fig_width, fig_height))
    axes = axes.flatten()
    for idx, spot in enumerate(spots):
        axes[idx].imshow(spot, cmap = 'jet')
        # Remove axis ticks and labels
        axes[idx].axis('off')
    
    # Remove unused axes if the number of spots is less than rows*cols
    for idx in range(len(spots), len(axes)):
        axes[idx].axis('off')

    if labels is not None:
        for ax, label in zip(axes, labels):
            ax.set_title(label)

    plt.tight_layout()
    
    


def crop_spot_image(img, plot_size):
    """ Crops an image to be centred on the brightest spot.
    
    Arguments:
        img       : numpy.ndarray
                    image to be cropped
        plot_size : int
                    cropped image covers this distance either side of centre

    Returns:
        numpy.ndarray: cropped image
    """   

    # Find centre pixel of spot
    centre_x, centre_y = find_spot(img)
       
    y_min = max(centre_y - plot_size, 0)
    y_max = min(centre_y + plot_size + 1, img.shape[0])
    x_min = max(centre_x - plot_size, 0)
    x_max = min(centre_x + plot_size + 1, img.shape[1])
    
    return img[y_min:y_max, x_min:x_max]




def fit_circle(x, y):
    """ Fits a circle to set of points.

    Arguments:
        x, y  : lists or arrays of x and y co-ordinates.

    Returns:
        tuple of (xx, yc, r, rms_error)
        xc, yc    : x and y centre of circle
        r         : radius of circle
        rms_error : of fit
    """

    x = np.asarray(x)
    y = np.asarray(y)

    x_m = np.mean(x)
    y_m = np.mean(y)
    r_m = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))

    def residuals(params):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    result = least_squares(residuals, x0=[x_m, y_m, r_m])

    xc, yc, r = result.x

    # Radial residuals
    res = residuals(result.x)

    # RMS error (good for error bars)
    rms_error = np.sqrt(np.mean(res**2))

    return xc, yc, r, rms_error


def spot_orientation(image, background=None, threshold=0):
    """
    Compute the major-axis orientation of an elongated spot using
    intensity-weighted second moments.

    Arguments:
        image       : np.ndarray
                      Spot image.

    Optional Keyword Arguments:
        background  : float or None
                      Background level to subtract. If None, no subtraction.
        threshold   : float
                      Pixels below this value after background subtraction are ignored.

    Returns:
        angle_deg : float
                    Major-axis angle in degrees, measured counterclockwise from +x.
        centroid :  tuple
                    (x_c, y_c)
        eigenvalues : np.ndarray
                      Principal variances [major, minor].
        eigenvectors : np.ndarray
                       Corresponding eigenvectors as columns.
    """

    I = np.asarray(image, dtype=float)

    if background is not None:
        I = I - background

    I[I < threshold] = 0

    total = I.sum()
    if total <= 0:
        raise ValueError("No positive intensity remains after preprocessing.")

    # Pixel coordinates
    y, x = np.indices(I.shape)

    # Intensity-weighted centroid
    x_c = (I * x).sum() / total
    y_c = (I * y).sum() / total

    # Coordinates relative to centroid
    dx = x - x_c
    dy = y - y_c

    # Covariance matrix
    Cxx = (I * dx * dx).sum() / total
    Cyy = (I * dy * dy).sum() / total
    Cxy = (I * dx * dy).sum() / total

    C = np.array([[Cxx, Cxy],
                  [Cxy, Cyy]])

    # Principal axes
    evals, evecs = np.linalg.eigh(C)

    # Sort so largest eigenvalue is first
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Major axis direction
    vx, vy = evecs[:, 0]

    angle_deg = np.degrees(np.arctan2(vy, vx))

    return angle_deg, (x_c, y_c), evals, evecs


def gaussian2d(coords, A, x0, y0, sigma_x, sigma_y, theta, B):
    """
    Generates a rotated 2D Gaussian.
    """
    x, y = coords

    ct = np.cos(theta)
    st = np.sin(theta)

    xp = (x - x0) * ct + (y - y0) * st
    yp = -(x - x0) * st + (y - y0) * ct

    g = A * np.exp(
        -0.5 * (
            (xp / sigma_x) ** 2 +
            (yp / sigma_y) ** 2
        )
    ) + B

    return g.ravel()


def fit_gaussian2d(image):
    """
    Fit a rotated 2D Gaussian to an image.

    Arguments:
        image : np.ndarray

    Returns: 
    
        params : dict
                  A, x0, y0, sigma_x, sigma_y, theta_deg, B
    """

    img = np.asarray(image, dtype=float)

    ny, nx = img.shape
    y, x = np.indices(img.shape)

    # Initial guesses
    B0 = np.percentile(img, 10)
    A0 = img.max() - B0

    total = np.sum(img - B0)
    if total <= 0:
        total = np.sum(img)

    x0 = np.sum(x * (img - B0)) / total
    y0 = np.sum(y * (img - B0)) / total

    sigma_x0 = nx / 10
    sigma_y0 = ny / 10
    theta0 = 0.0

    p0 = (
        A0,
        x0,
        y0,
        sigma_x0,
        sigma_y0,
        theta0,
        B0,
    )

    bounds = (
        [
            0,          # A
            0,          # x0
            0,          # y0
            0.1,        # sigma_x
            0.1,        # sigma_y
            -np.pi/2,   # theta
            -np.inf     # B
        ],
        [
            np.inf,
            nx,
            ny,
            nx,
            ny,
            np.pi/2,
            np.inf
        ]
    )

    popt, pcov = curve_fit(
        gaussian2d,
        (x, y),
        img.ravel(),
        p0=p0,
        bounds=bounds
    )

    A, x0, y0, sigma_x, sigma_y, theta, B = popt

    # Ensure sigma_x corresponds to major axis
    if sigma_y > sigma_x:
        sigma_x, sigma_y = sigma_y, sigma_x
        theta += np.pi / 2

    theta_deg = np.degrees(theta)

    # Wrap angle into [-90,90)
    theta_deg = ((theta_deg + 90) % 180) - 90

    return {
        "A": A,
        "x0": x0,
        "y0": y0,
        "sigma_major": sigma_x,
        "sigma_minor": sigma_y,
        "theta_deg": theta_deg,
        "background": B,
        "covariance": pcov,
    }