# -*- coding: utf-8 -*-
"""
Simple test of the image loading and saving.

@author: Mike Hughes
"""

import sys
sys.path.append('../src')

import numpy as np
from numpy.testing import *

from hugheslab.im_tools import *

stack = (np.random.random((20,100,100)) * 255 )

save_tif_stack(stack, 'test_stack.tif', auto_contrast = None, bit_depth = 8)
a = load_image('test_stack.tif')

assert_array_equal(stack.astype(int), a)


stack = (np.random.random((20,100,100)) * 63335 )

save_tif_stack(stack, 'test_stack.tif', auto_contrast = None, bit_depth = 16)
a = load_image('test_stack.tif')

assert_array_equal(stack.astype(int), a)