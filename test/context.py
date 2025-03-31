# -*- coding: utf-8 -*-
"""
Adds folder containing source of Hugheslab to path.

"""

import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../src'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))