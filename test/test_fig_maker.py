# -*- coding: utf-8 -*-
"""
Simple test the fig_maker.

@author: Mike Hughes
"""

import sys
sys.path.append('../src')

import numpy as np

from hugheslab.fig_maker import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches


imgs = []
for idx in range(8):
    im = np.random.random((400,400))
    imgs.append(im)
 
    
plot = multi_im(imgs, nCols = 4, width = 150, dpi=300)


ax = plot.get_axes()[0]

ax_zoom(ax, loc = (100,100,50,50), zoom = 4, 
        place = 'se', offset = .2, 
        displayLoc = None, originBox = False)

ax = plot.get_axes()[1]

scalebar(ax, length = 100, text = "10 mm", h = 1, place = 'se')

