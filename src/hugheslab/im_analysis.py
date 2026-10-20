# -*- coding: utf-8 -*-
"""
HughesLab: im_analysis

"""

import os
import math

import re

import numpy as np
import numpy.ma as ma

from PIL import Image

from scipy.optimize import curve_fit
from scipy.special import erf

import cv2 


VERTICAL = 0
HORIZONTAL = 1

def profile_line(image, position, direction = 0, length = 100, width=1):
    """
    Generates a profile across a line in an image.
    
    Arguments:
    image :     numpy.ndarray
                The image to profile.
    position :  int
                The pixel position of the profile.
    direction : int
                The direction of the profile. VERTICAL (0, default) or HORIZONTAL (1).
    length :    int
                The length of the profile. 

    Returns:
        numpy.ndarray: The profile of the image.


    """
    im_height, im_width = image.shape[:2]
    
    # Extract a profile across the whole image, averaged over the specified width
    if direction == VERTICAL:
        profile = np.mean(image[:, position-width//2:position+width//2 + 1], axis=1)
    elif direction == HORIZONTAL:
        profile = np.mean(image[position-width//2:position+width//2 + 1, :], axis=0)
    
  
    # Find the peak of the profile
    peak = np.argmax(profile)

    # Crop profile between -length/2 and +length/2 around the peak. Check that the crop is within the image.
    if direction == VERTICAL:
        profile = profile[max(peak-length//2,0):min(peak+length//2, im_height - 1)]
    elif direction == HORIZONTAL:
        profile = profile[max(peak-length//2,0):min(peak+length//2, im_width - 1)]
   

    return profile



def fwhm(profile):
    """
    Calculates the Full Width at Half Maximum (FWHM) for a 1D profile.

    Arguments:
        profile : numpy.ndarray
                  1D array representing the profile.

    Returns:
        float: The FWHM of the profile.
    """
    
    # Normalize the profile
    half_max = np.max(profile) / 2
    
    indices = np.arange(len(profile))

    # Find where the profile crosses the half max on both sides of the peak
    pts = np.where(profile > half_max)[0]

    if len(pts) < 2:
        return np.nan  # If no proper crossing points are found
    
    left_point = pts[0]
    right_point = pts[-1]   

    # Calculate the FWHM as the difference between the two crossing points
    return right_point - left_point


def gaussian(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b



def profile_gaussian_fit(profile, upsample = 1, crop = False, crop_factor = 10, match_height = False):
    """ Fits a Gaussian to a profile with optional upsampling and returns the 
    FWHM, paramters of the fit and upsampled plots of original data and fit.
    
 
    Arguments:
        profile  : numpy.ndarray
                   1D array containing profile to fit
      
    Optional Keyword Arguments:
        upsample     : int
                       factor to upsample by (default = 1, no upsampling)     
        crop         : boolean
                       if True, crops around estimated peak position before
                       fitting. Returned profiles will be cropped to same area
        crop_factor :  int
                       If crop == True, when cropping the cropped area will this
                       number multiplied by estimated FWHM.         
 
    Returns:
        Tuple of:
            fwhm: int: FWHM of Gaussian fit
            fit: tuple: containing parameters of fit (height, x offset, sigma)
            x_points: ndarray: 1D vector containing pixel numbers, adjusted for upsampling
            profile: ndarray: profile, upsampled (if upsampling used)
            profile_fit: ndarray: fitted profile, upsampled
    """
        
    from scipy.optimize import curve_fit
    
    if crop:
        peak_loc = np.argmax(profile)
        peak_val = np.max(profile)

        # Find approx width of peak
        start = np.where(profile > peak_val / 2)[0][0] - 1
        end = np.where(profile[peak_loc:] < peak_val / 2)[0][0] + peak_loc + 1

        fwhm_estimate = end - start
        region_start = start - int((fwhm_estimate // 2 ) * crop_factor)
        region_end = end + int((fwhm_estimate // 2 ) * crop_factor)

        region_start = max(region_start, 0)
        region_end = min(region_end, len(profile))

            
        # Extract ROI around peak
        profile = profile[region_start: region_end]

    
    # Upsample profile
    if upsample != 1:
        x_points = np.arange(len(profile))
        xnew = np.linspace(0, len(profile), num=len(profile) * upsample)
        profile = np.interp(xnew, x_points, profile)
    

    indices = np.arange(len(profile))

    fit = gaussian_fit(profile, match_height = match_height)
    
    predicted = gaussian(indices, *fit)

       
    sigma = fit[2]
    fwhm = sigma * 2.355 / upsample
    

    x_points = (indices - fit[1]) / upsample

    return fwhm, fit, x_points, profile, predicted



def gaussian_fit(profile, match_height = False):
    """ Fits a Gaussian to a profile and returns the parameters of the fit.
    """
    from scipy.optimize import curve_fit
    
    indices = np.arange(len(profile))

    # Initial guess for the parameters
    a = np.max(profile)
    x0 = np.argmax(profile)
    sigma = fwhm(profile) / 2.355
    if np.isnan(sigma):
        sigma = 1
    b = 0

    if match_height:
        popt, pcov = curve_fit(
            lambda x, x0, sigma, b: gaussian(x, a, x0, sigma, b),
            indices,
            profile,
            p0=[x0, sigma, b]
            )
        return [a, *popt]

    else:
        popt, pcov = curve_fit(
            gaussian, 
            indices, 
            profile, 
            p0=[a, x0, sigma, b])
        return popt






    

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)


def esf(x, offset, amplitude, x0, sigma):
        return offset + amplitude * 0.5 * (
            1 + erf((x - x0)/(np.sqrt(2)*sigma))
   )

def edge_fit_erf(edge_profile):
    """ Fits an error function to an edge profile and returns the parameters 
    of the fit including the Gaussian sigma and the FWHM of the edge.
    
    Arguments:
        edge_profile : numpy.ndarray
                       1D array containing edge profile to fit
   

    """
   

    y = edge_profile
    x = np.arange(len(y))
    
    # Initial guesses
    p0 = [
        np.min(y),                    # offset
        np.max(y) - np.min(y),        # amplitude
        x[np.argmin(np.abs(y - np.mean(y)))],  # edge location
        1.0                           # sigma
    ]

    popt, pcov = curve_fit(esf, x, y, p0=p0)

    offset, amplitude, x0, sigma = popt

    return offset, amplitude, x0, sigma, 



def GHT(n, x=None, nu=1, tau=1, kappa=1, omega=0.5):
    """
    Generalized Histogram Thresholding (GHT) method for image thresholding.  
    This function computes the optimal threshold for a histogram of pixel intensities
    based on the Generalized Histogram Thresholding method proposed by Barron in 2020.

    Taken from:
    @article{BarronECCV2020,
    Author = {Jonathan T. Barron},
    Title = {A Generalization of Otsu's Method and Minimum Error Thresholding},
    Journal = {ECCV},
    Year = {2020}

    Arguments:
    n : numpy.ndarray
        Histogram of pixel intensities, where n[i] is the count of pixels with intensity i.
    x : numpy.ndarray, optional 
        The pixel intensity values corresponding to the histogram, n. If None, it defaults to
        the indices of the histogram.
    nu : float, optional        
        A parameter that controls the influence of the variance term. Default is 1.
    tau : float, optional
        A parameter that controls the influence of the mean term. Default is 1.
    kappa : float, optional
        A parameter that controls the influence of the prior term. Default is 1.
    omega : float, optional
        A parameter that controls the balance between the two classes. Default is 0.5.
    Returns:
    tuple :
        A tuple containing the optimal threshold and the value of the objective function at that threshold.
    """
    
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1

    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[:-1])

    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    
    return argmax(x, f0 + f1), f0 + f1




def fwhm_from_sigma(sigma):
    """ Calculates the Full Width at Half Maximum (FWHM) from a Gaussian sigma.
    
    Arguments:
        sigma : float
                The standard deviation of the Gaussian.
    """
    return sigma * 2.355


def smooth_profile(profile, window_size=5):
    """ Smooths a 1D profile using a simple moving average.
    
    Arguments:
        profile : numpy.ndarray
                  1D array representing the profile.
        window_size : int
                      The size of the moving average window (default is 5).
    """
    return np.convolve(profile, np.ones(window_size)/window_size, mode='same')




from scipy.optimize import curve_fit


def gaussian2d(coords, A, x0, y0, sigma_x, sigma_y, theta, B):
    """
    Rotated 2D Gaussian.
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

    Parameters
    ----------
    image : 2D ndarray

    Returns
    -------
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

