# -*- coding: utf-8 -*-
"""
HughesLab: im_file_tools.py

Utilities for loading and saving images and stacks of images.

"""

import os
import re

import numpy as np

from PIL import Image

import cv2 

def load_image(filename):
    """
    Loads an image or stack of images from a file. 
    
    Arguments:
        filename    : str or Path
                      path to file
    
    Returns:
        ndarray     : 2D, 3D or 4D numpy array representing image or stack  

    """
    with Image.open(filename) as im:
    
        if hasattr(im, 'n_frames') and im.n_frames > 1:
            h,w = np.shape(np.array(im))
            dt = np.array(im).dtype
            stack = np.zeros((im.n_frames, h,w), dtype = dt)
            for i in range(im.n_frames):
                im.seek(i)
                stack[i,:,:] = np.array(im)
            return stack
        else:    
            return np.array(Image.open(filename))




def save_image_colour(img, filename):
    """ Saves image as a colour image without scaling
    Arguments:
        img      : ndarray
                   input image as 3D numpy array
        filename : str
                   path to save to, folder must exist   
    """
        
    with Image.fromarray(img.astype('uint8')) as im:
        im.save(filename)


def save_image8(img, filename):
    """ Saves image as 8 bit image without scaling. File type is determined by filename extension. See
    PIL documentation for supported file types.

    Arguments:
         img      : ndarray, 
                    input image as 2D numpy array
                   
         filename : str
                    path to save to, folder must exist
    """
    
    with Image.fromarray(img.astype('uint8')) as im:
        im.save(filename)



def save_image16(img, filename):
    """ Saves image as 16 bit image without scaling. File type is determined by filename extension. See
    PIL documentation for supported file types. Tested with tif files, which support 16 bit images, 
    but may not work with file types that do not support 16 bit images.
        
    Arguments:
         img      : ndarray, 
                    input image as 2D numpy array
                   
         filename : str
                    path to save to, folder must exist
    """
    with Image.fromarray(img.astype('uint16')) as im:
        im.save(filename)


     
def save_image8_scaled(img, filename):
    """ Saves image as 8 bit tif with scaling to use full dynamic range.
            
    Arguments:
         img      : ndarray, 
                    input image as 2D numpy array
                   
         filename : str
                    path to save to, folder must exist
    """
    
    with Image.fromarray(to8bit(img)) as im:
        im.save(filename)
    
    
def save_image16_scaled(img, filename):
    """ Saves image as 16 bit tif with scaling to use full dynamic range.
            
    Arguments:
         img      : ndarray, 
                    input image as 2D numpy array
                   
         filename : str
                    path to save to, folder must exist
    """
        
    with Image.fromarray(to16bit(img)) as im:
        im.save(filename) 



def load_stack(folder, num = None, status = True):
    """ Loads a stack of images from a folder into a 3D numpy array
    
    Arguments:
        folder      : str
                      path to folder
    Keyword Arguments:
        num         : int or None
                      if None (default), or images will be loaded, otherwise
                      the first num image will be loaded
        status      : boolean
                      if True, updates status on console (default = True)              
                      
    Returns:
        ndarray, 3D numpy array (im, y, x)                     
    """    

    image_files = [f for f in os.listdir(folder)]
    
    if num is None:
        nImages = len(image_files)  
    else:
        nImages = num
        
    image_files = image_files[:nImages]
    
    testIm = np.array(Image.open(os.path.join(folder,image_files[0])))
    
    h, w = np.shape(testIm)    
    
    data = np.zeros((nImages, h, w), dtype = testIm.dtype)
   
   
    if status: print(f"Loading files from '{folder}'")
    
    if status:
        for idx, image_file in enumerate(tqdm(image_files)):            
            data[idx,:,:] = np.array(Image.open(os.path.join(folder,image_file)))
    else:
        for idx, image_file in enumerate(image_files):            
            data[idx,:,:] = np.array(Image.open(os.path.join(folder,image_file)))        
        
    return data   
  

def save_tif_stack(stack, filename, bit_depth = 16, auto_contrast = None, fixed_min = None):
    """ Writes stack of images from 3D numpy array to file. The array must 
    be ordered (frame, y, x).
    
    Arguments:
        stack         : ndarray
                        3D numpy array (frame, y, x)
        filename      : str or Path
                        path to file name. Folder must exist.            
                       
    Keyword Arguments:
        bit_depth     : int
                        8 or 16 (default)
        auto_contrast : str or None
                        Whether or not to scale images to use full bit depth
                        'image' to autoscale each image individually
                        'stack' to autoscale entire stack
                        None or 'none' for no autoscaling
        fixed_min     : int or None
                        if auto_contrast is 'image' or 'stack', setting this
                        value fixed the lower range of the saved image pixel
                        values rather than taking the minimum from the images.
                                         
    """
    if stack.ndim != 3:
       raise Exception("Stack must be 3D array.")

    if auto_contrast == 'stack':
       maxVal = np.max(stack)
       if fixed_min is None:
           minVal = np.min(stack)
       else:
           minVal = fixed_min
    elif auto_contrast == 'image' or auto_contrast == 'none' or auto_contrast is None:
        pass
    else:
        raise Exception("Keyword auto_contrast only accepts 'stack', 'image' or None.")

       
    if bit_depth == 16:
        dt = 'uint16'
    elif bit_depth == 8:
        dt = 'uint8'
    else:
        raise Exception("Bit depth can only be 8 or 16.")
    
    imlist = []
    for im in stack:
        
        if auto_contrast == 'image':
            maxVal = np.max(np.abs(im))
            if fixed_min is None:
                minVal = np.min(np.abs(im))
            else:
                minVal = fixed_min
                
        if auto_contrast is not None:
            im = im.astype('float64') - minVal
            intrange = maxVal - minVal
            if intrange > 0:
                im = im / intrange * (2**bit_depth - 1)
            else:
                im[:] = 2**bit_depth - 1
        imlist.append(Image.fromarray(im.astype(dt)))


    imlist[0].save(filename, compression="tiff_lzw", save_all=True,
           append_images=imlist[1:])
    


def load_folder_images(folder, numeric_sort = True):
    """Loads all images in a folder and stores them as a 3D or 4D numpy array, 
    depending on whether they are monochrome or colour images. All images
    must be the same size.

    Arguments:
        folder : str
                 path to folder
    """
    # get list of files in folder
    files = os.listdir(folder)

    if numeric_sort:
        # Function to extract numbers from a filename
        def extract_number(filename):
            # Find all numbers in the filename
            match = re.findall(r'\d+', filename)
            # Convert to an integer and return the first number, or return 0 if none found
            return int(match[0]) if match else 0

        # Sort filenames using the custom key
        files = sorted(files, key=extract_number)    
    
    # load first image to determine size
    with Image.open(os.path.join(folder, files[0])) as im_first:
        w, h = im_first.size
        mode = im_first.mode
    
    
    n = len(files)

    # check if colour or monochrome
    if im_first.mode == 'RGB':
        stack = np.zeros((n, h, w, 3), dtype = np.uint16)
    else:
        stack = np.zeros((n, h, w), dtype = np.uint16)

    # load images
    for i, file in enumerate(files):
        with Image.open(os.path.join(folder, file)) as im:

            # check image is same size as first
            if im.size != (w, h):
                raise ValueError(f"Image {file} is not the same size as the first image.")
            
            # check if colour or monochrome same as first
            if im_first.mode != im.mode:
                raise ValueError(f"Image {file} is not same format as the first image.")
                
               
            stack[i] = np.array(im) 

    return stack    
        

def folder_to_tif_stack(folder, tif_file):
    """Loads all images in a folder and saves them as a 16 bit tif stack.
    
    Arguments:
        folder   : str
                   path to folder
        tif_file : str
                   path to tif file to save to
    """
    stack = load_folder_images(folder)
    save_tif_stack(stack, tif_file)


def tif_stack_to_folder(tif_file, folder):
    """Loads a 16 bit tif stack and saves the images to a folder as 16 bit tifs.
    """
    stack = load_image(tif_file)
    for i, im in enumerate(stack):
        save_image16(im, os.path.join(folder, f"image_{i}.tif"))


def stack_to_folder(stack, folder, prefix = 'image_', bit_depth = 16):
    """Saves a tif stack to a folder of tifs.

    Arguments:
        stack   : ndarray
                  3D numpy array (frame, y, x)
        folder  : str  

    Keyword Arguments:
        prefix  : str
                  prefix for image files, default is 'image_'
        bit_depth : int
                    8 or 16, default is 16     
    """

    # Check if folder exists, otherwise create it
    if not os.path.exists(folder):
        os.makedirs(folder)
   
    for i, im in enumerate(stack):
        if bit_depth == 8:
            save_image8(im, os.path.join(folder, f"{prefix}_{i}.tif"))
        elif bit_depth == 16:   
            save_image16(im, os.path.join(folder, f"{prefix}_{i}.tif"))        



def save_video(stack, filename, fps = 30, auto_contrast = None, fixed_min = None):
    """ Saves a stack of images as a video file.
    
    Arguments:
        stack    : ndarray
                   3D numpy array (frame, y, x)
        filename : str
                   path to save to, folder must exist
    
    Keyword Arguments:
        fps      : int
                   frames per second, default is 30
        auto_contrast : str or None
                        Whether or not to scale images to use full bit depth
                        'image' to autoscale each image individually
                        'stack' to autoscale entire stack
                        None or 'none' for no autoscaling 
        fixed_min     : int or None
                        if auto_contrast is 'image' or 'stack', setting this
                        value fixes the lower range of the saved image pixel
                        values rather than taking the minimum from the images.
                        Default is None.

    """ 

    bit_depth = 8

    if auto_contrast == 'stack':
       maxVal = np.max(stack)
       if fixed_min is None:
           minVal = np.min(stack)
       else:
           minVal = fixed_min
    elif auto_contrast == 'image' or auto_contrast == 'none' or auto_contrast is None:
        pass
    else:
        raise Exception("Keyword auto_contrast only accepts 'stack', 'image' or None.")

    num_images, h, w = np.shape(stack)[0:3]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    
    for im in stack:

        if auto_contrast == 'image':
            maxVal = np.max(np.abs(im))
            if fixed_min is None:
                minVal = np.min(np.abs(im))
            else:
                minVal = fixed_min
                
        if auto_contrast is not None:
            im = im.astype('float64') - minVal
            intrange = maxVal - minVal
            if intrange > 0:
                im = im / intrange * (2**bit_depth - 1)
            else:
                im[:] = 2**bit_depth - 1

        # Image is        
        # Images is monochrome, so covert to colour with all channels having the same value
        im_to_save = np.stack((im, im, im), axis = -1)
        im_to_save = im_to_save.astype('uint8')

        out.write(im_to_save)
        
    out.release()


