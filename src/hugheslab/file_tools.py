# -*- coding: utf-8 -*-
"""
#  HughesLab: file_tools

"""

import re


def file_sort(files):
    """ Sorts a list of filename so that they are in numeric order even if 
    trailing zeros are not used. For example 'file10' will be after 'file2'
    """
    def extract_number(filename):
            # Find all numbers in the filename
            match = re.findall(r'-?\d+', filename)
            # Convert to an integer and return the first number, or return 0 if none found
            return int(match[0]) if match else 0

    # Sort filenames using the custom key
    files = sorted(files, key=extract_number)
     
    return files