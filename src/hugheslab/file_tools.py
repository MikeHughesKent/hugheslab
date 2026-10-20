# -*- coding: utf-8 -*-
"""
#  HughesLab: file_tools

"""

import re
import os


def file_sort(files, return_num = False):
    """ Sorts a list of filename so that they are in numeric order even if 
    trailing zeros are not used. For example 'file10' will be after 'file2'
    """
    def extract_number(filename):
            # Find all numbers in the filename
            file = os.path.split(filename)[1]
            match = re.findall(r'-?\d+', file)
            # Convert to an integer and return the first number, or return 0 if none found
            return int(match[0]) if match else 0

    # Sort filenames using the custom key
    files = sorted(files, key=extract_number)

    if return_num:
        return files, [extract_number(f) for f in files]
    else:
        return files        


def leaf_folders(root_folder):
    """Return all leaf directories under root_folder. Returns relative paths.

    A leaf directory is a directory that contains no subdirectories.
    """
    lowest_level_folders = []

    for current_path, subdirs, _ in os.walk(root_folder):
        if not subdirs:
        # Get path relative to root_folder
            relative_path = os.path.relpath(current_path, root_folder)   
            lowest_level_folders.append(relative_path)

    return lowest_level_folders