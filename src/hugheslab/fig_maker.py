# -*- coding: utf-8 -*-
"""
Functions to help build nice figures.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
import string
import numpy as np

from PIL import Image

def qshow(img, title = None, axes = None, cmap = None):
    """ Utility to quickly display an image.
    """
    
    if cmap is None:
        cmap = 'gray'
    
    ax = plt.figure(dpi=150)
    plt.imshow(img, cmap = 'gray')
    if title is not None: plt.title(title)
    if axes is not None:
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])
    plt.show()    
    
    return 



def multi_im(imgs, nCols = 4, labelLetters = True, labels = None, rowLabels = None, columnLabels = None, fontSize = 8, fontFamily = "Times New Roman", width = 140, dpi = 150, autoScale = True):
    """ Produces a multi-panel figure made up of square images.
    
    Arguments:
        img          : list of images, each image as 2D/3D np array
        
    Keyword Arguments:        
        nCols        : number of columns in figure (default is 4)
        labelLetters : if true, each image gets a letter label (default is True)
        labels       : list of labels to use for images (default None).
        rowLabels    : list of labels for rows (default None)
        columnLabels : list of labels for columns (default None)
        fontSize     : font size for all labels
        fontFamily   : font family as a string
        width        : width of figure in mm (default 140)
        dpi          : dpi of figure (defaut 150)
        
    Returns :
        Matplotlib figure
        
    """
    
    plt.rcParams["font.family"] = fontFamily
    plt.rcParams["font.size"] = fontSize

    nIms = len(imgs)

    nRows = math.ceil(nIms / nCols)
    
    width = width / 25.4                # convert mm to inches
    
    height = (width / nCols * nRows)
    
    fig, axes = plt.subplots(nRows, nCols, dpi=dpi, figsize = (width, height), sharex=False, sharey=False)

    for idx, img in enumerate(imgs):
        txt = ''
        if nRows > 1:
            
            this = axes[idx // nCols, idx % nCols ]
        else:
            this = axes[idx]
        if autoScale:
            this.imshow(img, cmap='gray')
        else:
            this.imshow(img, cmap='gray', vmin = 0, vmax = 1)
        if labelLetters:
            txt = txt + f"({string.ascii_lowercase[idx]})"
        if labels is not None:
            txt = txt + labels[idx]
        this.set_xlabel(txt) 
   

    for x, axi in enumerate(axes):
        if nRows > 1:
            for y, ax in enumerate(axi):
                 if x * nCols + y > nIms - 1:
                     ax.set_axis_off()
                 ax.set_yticks([])
                 ax.set_xticks([])
        else:
            if x > nIms - 1:
                axi.set_axis_off()
            axi.set_yticks([])
            axi.set_xticks([])
            
            
    if rowLabels is not None:
        for idx in range(len(rowLabels)):
            fig.get_axes()[idx * nCols].set_ylabel(rowLabels[idx])   
            
    if columnLabels is not None:
        for idx in range(len(columnLabels)):
            fig.get_axes()[idx].set_title(columnLabels[idx], fontsize = fontSize)           
       
 
             
    return fig    


def img_zoom(img, loc, zoom = 4, place = 'se', displayLoc = None):
    """ Adds an zoomed inset to an image.
    
    Arguments:
        img    : ndarray
                 the original image, 2D numpy array
        loc    : tuple
                 location of inset, tuple of (x,y,w,h)
        zoom   : float
                 zoom factor (default 4)
        place  : str
                 compass location: 'se', 'ne', 'nw' or 'sw'
    
    """
    
    zoomImg = get_zoom(img, loc, zoom)
    
    zy, zw = np.shape(zoomImg)

    if place == 'se':
        img[-zy:,-zw:] = zoomImg
    elif place == 'ne':
        img[:zy,-zw:] = zoomImg
    elif place == 'nw':
        img[:zy, :zw] = zoomImg    
    elif place == 'sw':
        img[-zy:,:zw] = zoomImg
    return img


def img_ax(ax):
    """ Returns image data from axis as numpy array
    """
    
    return ax.get_images()[0].get_array().toflex()['_data'] 
    


def ax_zoom(ax, loc, zoom = 4, place = 'se', 
            displayLoc = None, offset = 0,
            originBox = True, zoomBox = True,
            lines = True, mainBox = True, autoScale = True, scaleFactor = 1):
    """ Adds a zoom inset to an axis
    
    Arguments:
        ax       : reference to axis
        loc      : zoomed area as tuple of (x,y,w,h)
        
    Keyword Arguments:
        zoom       : zoom factor
        place      : reserved
        displayLoc : reserved
        offset     : overhand of inset over edge of original image, either int or tuple of (x,y)
        originBox  : True to draw box around zoom in original image (default)
        zoomBox    : True to draw box around zoom in zoomed image (default)
        lines      : True to draw lines between original and new areas (default)
        mainBox    : True to frame image (default)
    
    """
   
    img = img_ax(ax)
    
    h,w = np.shape(img)[:2]
    if isinstance(offset,tuple):
        xoffset, yoffset = offset
    else:
        xoffset = offset
        yoffset = offset

    xoffset = int(xoffset * w)
    yoffset = int(yoffset * w)

    ox,oy,ow,oh = loc

    zoomImg = get_zoom(img, loc, zoom) * scaleFactor

    if img.ndim == 3:
        imgOut = max(np.max(img), np.max(zoomImg)) * np.ones((h + yoffset, w + xoffset, np.shape(img)[2]), dtype = img.dtype)
        imgOut[:h, :w, :] = img
    else:
        imgOut = max(np.max(img), np.max(zoomImg)) * np.ones((h + yoffset, w + xoffset), dtype = img.dtype)
        imgOut[:h, :w] = img
        

    zh, zw = np.shape(zoomImg)[:2]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
   
    y = h + xoffset - zh
    x = w + yoffset- zw
    if img.ndim == 3:
        imgOut[-zh:,-zw:,:] = zoomImg
    else:
        imgOut[-zh:,-zw:] = zoomImg
    if autoScale:
        ax.imshow(imgOut, cmap = ax.get_images()[0].get_cmap())
    else:
        ax.imshow(imgOut, cmap = ax.get_images()[0].get_cmap(), vmin = 0, vmax = 1)

        

    if mainBox:
        ax.add_patch(patches.Rectangle((0, 0), 0, h, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((0, 0), w, 0, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((0, h), min(w + xoffset - zw, w), 0, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((w, 0), 0, min(h + yoffset - zh,h), linewidth=1, edgecolor='k', facecolor='none'))
        
    
    # inset
    if zoomBox:
        ax.add_patch(patches.Rectangle((y, x), zh, zw, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((y, x), zh, zw, linewidth=1, edgecolor='w', ls = '--', facecolor='none'))

    # origin
    if originBox:
        ax.add_patch(patches.Rectangle((ox, oy), ow, oh, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((ox, oy), ow, oh, linewidth=1, edgecolor='w', ls = '--', facecolor='none'))

    # connecting lines
    if lines and originBox and zoomBox:
        fh,fw = np.shape(imgOut)[:2]
    
        ax.add_patch(patches.Polygon(np.array(((ox, oy + oh), (fw - zw, fh))), closed = False, linewidth=1, edgecolor='k',  facecolor='none'))
        ax.add_patch(patches.Polygon(np.array(((ox + ow, oy), (fw , fh - zh))), closed = False, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Polygon(np.array(((ox, oy + oh), (fw - zw, fh))), closed = False, linewidth=1, edgecolor='w',  ls = '--', facecolor='none'))
        ax.add_patch(patches.Polygon(np.array(((ox + ow, oy), (fw , fh - zh))), closed = False, linewidth=1, edgecolor='w', ls = '--', facecolor='none'))


def scalebar(ax, length = 100, text = None, hoffset = 1, voffset = 2, w = None, h = 1, place = 'nw', padding = 1):
    """ Adds a scalebar to an image
    
    Arguments:
        ax      : axis object
        length  : int
                  length of scalebar in image pixels
        text    : str
                  label to sit next to scale bar
        hoffset : float
                  distance from horizontal edge
        voffset : float
                  distance from vertiacl end
        w       : float
                  if specified sets widths of bar to this in mm
        h       : float
                  height of bar in mm
        place   : str
                  compass point location, 'nw' or 'se' 
    """
    
   
    hoffset = mm2pix(ax,hoffset)
    voffset = mm2pix(ax,voffset)
    y = voffset + min(ax.get_ylim())
    x = hoffset + min(ax.get_xlim())
    
    imH, imW = np.shape(img_ax(ax))[:2]
    
    h = mm2pix(ax,h)

    if w is not None:
        w = mm2pix(ax,w)
    else:
        w = length
    
    if place == 'nw':
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='w', facecolor='w'))
    elif place == 'se':
        ax.add_patch(patches.Rectangle((imW - x - w, imH - y - h), w, h, linewidth=1, edgecolor='w', facecolor='w'))

        
    if text is not None:
        r = ax.get_figure().canvas.get_renderer()

        if place == 'nw':

            tx = x + w + hoffset
            ty = y
            t = ax.text(tx, ty , text, va = 'center_baseline', fontsize = 'small', color = 'w') #, bbox={"fill": True, "lw": 0.2, "color": (0,0,0,.5),"boxstyle": "square,pad=0.1"})
    
            bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
            tw = bb.width
            th = bb.height
            padding = mm2pix(ax,padding)
            ax.add_patch(patches.Rectangle((x - padding, y + th/2 - padding), tx - x + tw + padding *2, -th + padding *2, linewidth=1, edgecolor=None, ls = '--', facecolor=(0,0,0,0.5)))
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='w', facecolor='w'))
        
        elif place == 'se':
            
            tx = imW - x - w - hoffset
            ty = imH - y - h
            t = ax.text(tx, ty , text, va = 'center_baseline', ha = 'right', fontsize = 'small', color = 'w') #, bbox={"fill": True, "lw": 0.2, "color": (0,0,0,.5),"boxstyle": "square,pad=0.1"})
    
            bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
            tw = bb.width
            th = bb.height
            padding = mm2pix(ax,padding)
            ax.add_patch(patches.Rectangle((imW - x - w - tw - hoffset - padding, imH - y + th ), tx + tw + padding *2, -th + padding *2, linewidth=1, edgecolor=None, ls = '--', facecolor=(0,0,0,0.5)))
            ax.add_patch(patches.Rectangle((imW - x - w, imH - y - h), w, h, linewidth=1, edgecolor='w', facecolor='w'))
         
            

def mm2pix(ax, mm):
    """ Converts a dimensions in mm to pixels for a specified axis"""
    
    dpi = ax.get_figure().get_dpi()
    imWidth = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    imPhysicalWidth = ax.get_window_extent().width / dpi * 25.4
    
    frac = mm / imPhysicalWidth 
    
    return frac * imWidth

    

def get_zoom(img, loc, zoom = 4):
    """ Gets a zoomed image from part of another image.
    """
    x,y,w,h = loc
    
    roi = img[y:y + h, x:x+w]
    im = Image.fromarray(roi)
    zoom = np.array(im.resize((int(w * zoom), int(h * zoom))))
    
    return zoom
    