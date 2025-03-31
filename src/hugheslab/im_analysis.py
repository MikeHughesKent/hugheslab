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

from tqdm import tqdm

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


def profile_gaussian_fit(profile):
    """ Fits a Gaussian to a profile and returns the parameters of the fit.
    """
    from scipy.optimize import curve_fit

    indices = np.arange(len(profile))

    # Initial guess for the parameters
    a = np.max(profile)
    x0 = np.argmax(profile)
    sigma = fwhm(profile) / 2.355
    b = 0

    popt, pcov = curve_fit(gaussian, indices, profile, p0=[a, x0, sigma, b])

    return popt

