"""
Functions to conveniently display images and complex images using matplotlib.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
import string
import numpy as np

from PIL import Image

def qshow(img, title = None, axes = None, cmap = 'gray', dpi = 150, figsize = (5,5), figsizem = None):
    """ Utility to quickly display an image in a matplotlib figure wtih useful defaults.
    
    Arguments:
        img      : ndarray
                   image to display
    
    Keyword Arguments:
        title    : str
                   plot title, default is not title
        axes     : (str, str)
                   x and y axis labels, default is no labels
        cmap     : str
                   matplotlib colormap, default is 'gray'
        dpi      : int
                   display dpi, default is 150 
        figsize  : (float, float)
                   x and y size of figure in inches, default (5,5)
        figsizem : (float, float)
                   x and y size of figure in mm, default None (overrides figsize if specified)   

        Returns:
        tuple of (Figure, AxesImage) : figure and image objects from matplotlib

    """
    
    if figsizem is not None:
        figsize = (figsizem[0] / 25.4, figsizem[1] / 25.4)

    ax = plt.figure(dpi=dpi, figsize = figsize)
    im = plt.imshow(img, cmap = cmap, interpolation = 'None', aspect = 'equal')
    if title is not None: plt.title(title)
    if axes is not None:
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])
    plt.show()    
    
    return ax, im

def cshow(img, dpi=150, figsize=(10, 5), figsizem=None, phase_cmap="twilight", amp_cmap="gray", title=None):
    """ Utility to quickly display a complex image as two subplots, one for amplitude and one for phase.

    Arguments:
        img        : numpy.ndarray
                     complex image to display
    Keyword Arguments:
        dpi        : int
                     resolution of figure in dots per inch (default = 100)
        figsize    : tuple of (float, float)
                     size of figure in inches (default = (10, 5))
        figsizem   : tuple of (float, float)
                     size of figure in millimeters (overrides figsize if specified, default = None)
        phase_cmap : str
                     name of matplotlib colormap to use for phase (default = 'twilight')
        amp_cmap   : str
                     name of matplotlib colormap to use for amplitude (default = 'gray')
        title      : str or None
                     title to set for whole figure (default = None)
    
    Returns:
        tuple of (Figure, (Axes, Axes)) : figure and axes objects from matplotlib
     """
        
    if figsizem is not None:
        figsize = (figsizem[0] / 25.4, figsizem[1] / 25.4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi = dpi)
    
    # Set title of whole figure
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    amp = ax1.imshow(np.abs(img), cmap=amp_cmap)
    ax1.set_title("Amplitude")

    phase = np.angle(img) % (2 * np.pi)  # Wrap phase to [0, 2π]

    ph = ax2.imshow(phase, cmap=phase_cmap, vmin=0, vmax=2 * np.pi)
    ax2.set_title("Phase")

    # Create a colourbar for the phase plot, scaled to show 0 to 2pi radians
    cbar_phase = plt.colorbar(ph, ax=ax2, fraction=0.046, pad=0.04)
    cbar_phase.set_label('Phase (radians)', rotation=270, labelpad=15)
    cbar_phase.set_ticks([0, np.pi, 2 * np.pi]) 
    cbar_phase.set_ticklabels(['0', 'π', '2π']) 
    
    # Create a colourbar for the amplitude plot
    cbar_amp = plt.colorbar(amp, ax=ax1, fraction=0.046, pad=0.04)
    cbar_amp.set_label('Amplitude', rotation=270, labelpad=15)

    # Set tight layout to prevent overlap of colourbars and titles
    plt.tight_layout()  # No suptitle, use full layout

    plt.show()

    return fig, (ax1, ax2)

def qplot(data1, data2 = None, title = None, axes = None, dpi = 150, figsize = (5,5), figsizem = None):
    """ Utility to quickly display a plot in a matplotlib figure wtih useful defaults.
    
    Arguments:
        x        : ndarray
                   x data to plot
        y        : ndarray
                   y data to plot

    """

    if data2 is None:
        x = np.arange(len(data1))
        y = data1
    else:
        x = data1
        y = data2

    if figsizem is not None:
        figsize = (figsizem[0] / 25.4, figsizem[1] / 25.4)  

    ax = plt.figure(dpi=dpi, figsize = figsize)
    p = plt.plot(x, y)  

    if title is not None: plt.title(title)
    if axes is not None:
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])

    plt.grid(True)  
    plt.show()

    return ax, p