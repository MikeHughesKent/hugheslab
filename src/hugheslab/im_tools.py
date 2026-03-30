# -*- coding: utf-8 -*-
"""
HughesLab: im_tools

Utilities for image processing, including cropping, scaling, radial profiles.

"""

import os
import math

import re

import numpy as np
import numpy.ma as ma

from PIL import Image

import cv2 

def crop_image(img, centre, dims):
    """Crops an image or stack of images stored as a numpy array, with cropping performed
    around a specified central point.
    
    Arguments:
        img:     ndarray
                 2D or 3D image as numpy array. If 2D must be (y,x), if 3D, 
                 must be (frame_number, y, x)
        centre:  tuple of (int, int)
                 pixel to be at centre of cropped image (x, y)
        dims:    tuple of (int, int)
                 size of cropped image (width, height)
    
    Returns:
        ndarray: cropped image or stack of cropped images
    """
    # Check if the input is 2D or 3D
    if img.ndim == 2:
        # If it's 2D, just crop the single image
        return __crop_single_image(img, centre, dims)
    elif img.ndim == 3:
        # If it's 3D, loop through the first dimension (each image) and crop
        cropped_images = []
        for i in range(img.shape[0]):
            cropped_image = __crop_single_image(img[i], centre, dims)
            cropped_images.append(cropped_image)
        return np.stack(cropped_images)
    else:
        raise ValueError("Input image must be a 2D or 3D numpy array")
        

def __crop_single_image(img, centre, dims):
    """Crops a single 2D image around the specified centre."""
    # Extract dimensions of the image and the target crop
    img_height, img_width = img.shape[:2]
    crop_width, crop_height = dims
    
    # Calculate cropping boundaries
    x_centre, y_centre = centre
    x_start = max(0, x_centre - crop_width // 2)
    y_start = max(0, y_centre - crop_height // 2)
    x_end = min(img_width, x_start + crop_width)
    y_end = min(img_height, y_start + crop_height)
    
    # Adjust the starting coordinates in case the end coordinates are too close to the image boundary
    x_start = max(0, x_end - crop_width)
    y_start = max(0, y_end - crop_height)
    
    # Return the cropped portion of the image
    return img[y_start:y_end, x_start:x_end]

    
def crop_zero(img):
    """ 
    Crops an image to the smallest rectangle that contains all non-zero
    pixels.
    
    Returns cropped image as 2D numpy array.
    
    Arguments:
        img      : ndarray
                   image as numpy array
                   
    Returns
        ndarray, cropped image                
    """
    
    hMax = np.amax(img, axis = 0)

    hCrop = (np.argwhere(hMax > 0)[0].item(), np.argwhere(hMax > 0)[-1].item() )
    
    vMax = np.amax(img, axis = 1)
    vCrop = (np.argwhere(vMax > 0)[0].item(), np.argwhere(vMax > 0)[-1].item() )
    
    img = img[vCrop[0]:vCrop[1],hCrop[0]:hCrop[1]]
    
    return img


def crop_zero_box(img):
    """ 
    Returns co-ordinates of box to crop and image so that there are no
    rows or columns totally black.
    
    
    Arguments:
        img      : ndarray
                   image as numpy array
                   
    Returns
        tuple    : (x1, x2, y1, y2)                
    """
    
    hMax = np.amax(img, axis = 0)

    hCrop = (np.argwhere(hMax > 0)[0].item(), np.argwhere(hMax > 0)[-1].item() )
    
    vMax = np.amax(img, axis = 1)
    vCrop = (np.argwhere(vMax > 0)[0].item(), np.argwhere(vMax > 0)[-1].item() )
    
    return vCrop[0], vCrop[1], hCrop[0], hCrop[1]

    
    

def extract_central(img, boxSize = None):
    """ 
    Extract a central square from an image. The extracted square is centred
    on the input image, with size 2 * boxSize if possible, otherwise the largest
    square that can be extracted.
    
    Returns cropped image as 2D numpy array.
    
    Arguments:
        img     : input image as 2D numpy array
        
    Keyword Arguments:    
        boxSize : size of cropping square, default is largest possible
        
    Returns:
        ndarray, cropped image
    """

    w = np.shape(img)[0]
    h = np.shape(img)[1]

    if boxSize is None:
        boxSize = min(w,h)
    cx = w/2
    cy = h/2
    boxSemiSize = min(cx,cy,boxSize)
    
    imgOut = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
    
    return imgOut


def to8bit(img, minVal = None, maxVal = None):
    """ Returns an 8 bit representation of image. If min and max are specified,
    these pixel values in the original image are mapped to 0 and 255 
    respectively, otherwise the smallest and largest values in the 
    whole image are mapped to 0 and 255, respectively.
    
    Arguments:
            img    : ndarray
                     input image as 2D numpy array
        
    Keyword Arguments:    
            minVal : float
                     optional, pixel value to scale to 0
            maxVal : float
                     optional, pixel value to scale to 255
    """
 
    img = img.astype('float64')
       
    if minVal is None:
        minVal = np.min(img)
                    
    img = img - minVal
        
    if maxVal is None:
        maxVal = np.max(img)
    else:
        maxVal = maxVal - minVal
        
    img = img / maxVal * 255
    img = img.astype('uint8')
    
    return img


def to16bit(img, minVal = None, maxVal = None):
    """ Returns an 16 bit representation of image. If min and max are specified,
    these pixel values in the original image are mapped to 0 and 2^16 
    respectively, otherwise the smallest and largest values in the 
    whole image are mapped to 0 and 2^16 - 1, respectively.
    
    Arguments:
        img    : ndarray
                 input image as 2D numpy array
        
    Keyword Arguments:    
        minVal : float
                 optional, pixel value to scale to 0
        maxVal : float
                 optional, pixel value to scale to 2^16 - 1
                 
    Returns:
        ndarray, 16 bit image             
    """   
        
    img = img.astype('float64')
       
    if minVal is None:
        minVal = np.min(img)
                    
    img = img - minVal
        
    if maxVal is None:
        maxVal = np.max(img)
    else:
        maxVal = maxVal - minVal
        
    img = img / maxVal * (2**16 - 1)
    img = img.astype('uint16')
    
    return img


def radial_profile(img, centre):
    """ Produce angular averaged radial profile through image img centred on
    centre, a tuple of (x_centre, y_centre)
    
    Returns radial profile as 1D numpy array

    Arguments:
        img    : ndarray
                 input image as 2D numpy array
        centre : (int, int)
                 centre point for radial profile, tuple of (x,y)  
    Returns:
        ndarray, 1D profile             
    """
    
    y, x = np.indices((img.shape))
    r = np.sqrt((x - centre[1])**2 + (y - centre[0])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), weights = img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin  / nr
    
    return radialprofile 
 

def average_channels(img):
    """ Returns an image which is the the average pixel value across all channels of a colour image.
    It is safe to pass a 2D array which will be returned unchanged.
    
    
    Arguments:
        img:    ndarray
                 image as 2D/3D numpy array
            
    Returns:
        ndarray, averaged image        
    """     

    if img.ndim == 3:
        return np.mean(img, 2)
    else:
        return img
    
    
def max_channels(img):
    """ Returns an image which is the the maximum pixel value across all channels of a colour image.
    It is safe to pass a 2D array which will be returned unchanged.
    
    
    Arguments:
        img:   ndarray
               image as 2D/3D numpy array
               
    Returns:
        ndarray, max value image           
    """     

    if img.ndim == 3:
        return np.max(img, 2)
    else:
        return img
        

def mean_channels(img):
    """ Returns an image which is the the mean pixel value across all channels of a colour image.
    It is safe to pass a 2D array which will be returned unchanged.
        
    Arguments:
        img:   ndarray
               image as 2D/3D numpy array
               
    Returns:
        ndarray, mean value image           
    """     

    if img.ndim == 3:
        return np.mean(img, 2)
    else:
        return img
        
    
def log_scale_image(img, min_val = None):
    """ Generates a log-scaled image, adjusted for good visual appearance,
    particularly for OCT images.
    
    Arguments:
        img     : ndarray 
                  image as 2D numpy array
     
    Optional Keyword Arguments:
        min_val : float
                  if specified, 0 in image will correspond to this value 
                  in log scaled image before windowing
                  
    Returns:
        ndarray, log-scaled image (8 bit)             
                   .
    """

    if min_val is None:
        dispMin = log_scale_min(img)
        
    else:
        dispMin = min_val
    
    imgOut = np.log(img + 0.00001)
    imgOut = imgOut - dispMin
    maxVal = np.max(imgOut) * 0.9
    imgOut = imgOut / maxVal
    imgOut[imgOut < 0] = 0
    imgOut[imgOut > 1] = 1
    
    return imgOut


def log_scale_min(img):
    """Calculates minimum display value for a log-scaled image so that 
    it does no appear overly noisy.
    
    Arguments:
        img     : ndarray 
                  image as 2D numpy array
         
    Returns:
        float, minimum value           
                   .
    """    
    
    mask = img == 0
    im = np.log(img + 0.00001)
    imgOut = ma.masked_array(im, mask)
    valRange = np.max(imgOut) - ma.min(imgOut)

    lineAv = ma.mean(imgOut,0)
    sVals = np.sort(lineAv)

    minVal = sVals[min(10, len(sVals) - 1)]

    dispMin = minVal - valRange / 25
    if dispMin is not ma.masked:
        return dispMin
    else:
        return 0    
   
    
   
def rect_to_pol(image, fov_angle=360, depth = None, offset = 0, num_points = None):
    """ Converts a rectangular image to a polar plot. Assumes that
    the y direction in the image is radial, and the x direction
    is angle.
    
    Arguments:
        image     : ndarray 
                    image as 2D numpy array
    Keyword Arguments:                  
        fov_angle : float
                    angular range corresponding to width of image, degrees
        depth     : float
                    max vertical range in inout immge to use, default is None
                    in which case all of image is used.
        offset    : float
                    min vertical position in input image to use, default is 0
        num_points: int
                    x and y size of output. If not specified, will be same
                    as y size of input image
                   
    Returns:
        ndarray, 2D numpy array containing output image
    """
    

    height, width = image.shape[:2]

    if depth is None:
        depth = height
        
    if num_points is None:
        num_points = depth
       
    origin = height // 2
    
    # Create coordinate maps
    #y, x = np.indices((depth, depth))
    x, y = np.meshgrid(np.linspace(0, depth, num_points), np.linspace(0,depth,num_points))
    
    # Calculate polar coordinates
    r = np.sqrt((x - origin)**2 + (y - origin)**2)
    theta = np.arctan2(x - origin, y - origin)
    
    # Map polar to Cartesian coordinates
    source_x = ((theta - np.min(theta)) / np.radians(fov_angle)) * (width - 1) 
    source_y = r * 2 + offset  
    
      
    # Interpolate values from the original image
    im_out = cv2.remap(image, source_x.astype(np.float32), 
                                 source_y.astype(np.float32), 
                                 cv2.INTER_LINEAR)
    
    return im_out 

    
def get_shifts(imgs, templateSize = None, refSize = None, upsample = 2, **kwargs):
    """ Determines the shift of each image in a stack w.r.t. first image
    
    Return shifts as 2D numpy array.
    
    Arguments:
        
        imgs         : stack of images as 3D numpy array
        
    Keyword Arguments:
        
        templateSize : int, a square of this size is extracted from imgs 
                       as the template, default is 1/4 image size
        refSize      : int, a square of this size is extracted from first 
                       image as the reference image, default is 1/2 image
                       size. Must be bigger than  
                       templateSize and the maximum shift detectable is 
                       (refSize - templateSize)/2   
        upSample     : upsampling factor for images before shift detection  
                       for sub-pixel accuracy, default is 2.
    """

    imgSize = np.min(np.shape(imgs)[1:3])

    if templateSize is None:
        templateSize = imgSize / 4

    if refSize is None:
        refSize = imgSize / 2

    nImages = np.shape(imgs)[0]
    refImg = imgs[0]
    shifts = np.zeros((nImages, 2))
    for iImage in range(1, nImages):
        img = imgs[iImage]
        thisShift = find_shift(
            refImg, img, templateSize, refSize, upsample)

        shifts[iImage, 0] = thisShift[0]
        shifts[iImage, 1] = thisShift[1]

    return shifts


def find_shift(img1, img2, templateSize, refSize, upsample, returnMax = False):
    """ Determines shift between two images by Normalised Cross 
    Correlation (NCC). A square template extracted from the centre of img2 
    is compared with a square region extracted from the reference image 
    img1. The size of the template (templateSize) must be less than the 
    size of the reference (refSize). The maximum detectable shift is 
    (refSize - templateSize) / 2.
    
    If returnMax is False, returns shift as a tuple of (x_shift, y_shift).
    If returnMax is True, returns tuple of (shift, cc. peak value).
    
    Arguments:
        img1         : image as 2D numpy array
        img2         : image as 2D numpy array
        templateSize : int, size of square region of img2 to use as template. 
        refSize      : int, size of square region of img1 to template match with
        upsample     : int, factor to scale images by prior to template matching to
                       allow for sub-pixel registration.  
                       
    Keyword Arguments:
        returnMax    : boolean, if true returns cc.peak value as well
                       as shift, default is False. 
               
    """
    

    if refSize < templateSize or min(np.shape(img1)) < refSize or min(np.shape(img2)) < refSize:
        return -1
    else:

        template = extract_central(img2, templateSize).astype('float32')
        refIm = extract_central(img1, refSize).astype('float32')
  

        if upsample != 1:

            template = cv2.resize(template, (np.shape(template)[
                                 0] * upsample, np.shape(template)[1] * upsample))
            refIm = cv2.resize(
                refIm, (np.shape(refIm)[0] * upsample, np.shape(refIm)[0] * upsample))

        res = cv2.matchTemplate(template, refIm, cv2.TM_CCORR_NORMED)
       
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        shift = [(max_loc[0] - (refSize - templateSize)   * upsample)/upsample,
                 (max_loc[1] - (refSize - templateSize)   * upsample)/upsample]
        
        if returnMax:
            return shift, max_val
        else:
            return shift

def filter_stack(stack, size):
    """Applies Guassian smoothing to each image in a stack.
    """

    kernel_size = math.ceil(size * 6)
    if kernel_size % 2 == 0:
        kernel_size += 1
    for i, im in enumerate(stack):
        stack[i] = cv2.GaussianBlur(im, (kernel_size, kernel_size), size)
    return stack
       

def resize_stack(stack, new_size, factor = None):
    """ Resizes each image in a stack. 
    
    Arguments:
        stack    : numpy.ndarray
                   stack of images, image number along axis 0
        new_size : tuple of (int,int)
                   New y and x size of image (same convention as numpy).
                   This is ignored if the factor is specified.
    Keyword Arguments:
        factor   : float or None
                   If not None (default) this will be used to determine the new
                   image sizes instead of new_size. Dimensions of stack are
                   multiplied by the factor, e.g. to reduce by a factor of 2
                   set factor = 0.5.
        
        
        """

    if factor is not None:
        new_size = ( int(np.shape(stack)[1] * factor), int(np.shape(stack)[2] * factor) )

    out_stack = np.zeros((np.shape(stack)[0], new_size[0], new_size[1]), dtype = stack.dtype)
    for idx, im in enumerate(stack):
        out_stack[idx,:,:] = cv2.resize(im, (new_size[0], new_size[1]))

    return out_stack


def hamming_window(num_points):
    """ Returns a Hamming window.

    Arguments:
      num_points     : int
                       length of window
    
    Returns:
      numpy.nddrray  : 1D array, values of window

    """
    a0 = 25/46
    window = a0 - (1 - a0) * np.cos(2 * math.pi *
                                    np.arange(1, num_points + 1) / num_points)
    return window


def condense_stack(stack):
    """ Assuming a 3D OCT stack of the form (frames, depth, x), produces a 2D stack that
    is of size (depth, x * frames), running fast through x and slow through frames.
    
    Arguments:
        stack : numpy.ndarray
                3D stack of images, shape (frames, x, y)

    Returns:
        numpy.ndarray : 2D stack of images, shape (frames, x * y)   
    """
    if stack.ndim != 3:
        raise ValueError("Input stack must be a 3D numpy array.")   
    frames, depth, x = stack.shape
    stack = np.swapaxes(stack, 0, 1)  # Change to (depth, frames, x)
    stack = np.reshape(stack, (depth, frames * x))
    return stack




