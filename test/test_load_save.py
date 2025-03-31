# -*- coding: utf-8 -*-
"""
Tests load and save functionality of HughesLab

"""
import context
import unittest
import os
import shutil


from hugheslab import *

import numpy as np
import matplotlib.pyplot as plt


class TestLoadSave(unittest.TestCase):

    
    def test_save_and_load_image_8bit(self):

        # Generate a random images as a numpy array
        image = np.random.random((100,100)) * 255
        image = image.astype(np.uint8)

        # Save the image to a file
        file_name = 'test_image8.tif'
        save_image8(image, file_name)

        # Load the image from the file
        loaded_image = load_image(file_name)

        # Delete the file
        os.remove(file_name)

        # Check that the loaded image is the same as the original image
        self.assertTrue(np.array_equal(image, loaded_image))


    def test_save_and_load_image_16bit(self):

        # Generate a random images as a numpy array
        image = np.random.random((100,100)) * 65535
        image = image.astype(np.uint16)

        # Save the image to a file
        file_name = 'test_image16.tif'
        save_image16(image, file_name)

        # Load the image \from the file
        loaded_image = load_image(file_name)

        # Delete the file
        os.remove(file_name)

        # Check that the loaded image is the same as the original image
        self.assertTrue(np.array_equal(image, loaded_image)) 


    def test_save_and_load_colour_image(self):    

        # Generate a random images as a numpy array with 3 channels
        image = np.random.random((100,100,3)) * 255

        image = image.astype(np.uint8)

        # Save the image to a file
        file_name = 'test_image_col.tif'

        save_image_colour(image, file_name)

        # Load the image from the file
        loaded_image = load_image(file_name)

        # Delete the file
        os.remove(file_name)
        
        # Check that the loaded image is the same as the original image
        self.assertTrue(np.array_equal(image, loaded_image))



    def test_tif_stack_folder(self):

        # Generate a stack of random images as a 3D numpy array   
        stack = np.random.random((10,100,100)) * 65535
        stack = stack.astype(np.uint16)

        # Save the stack to a folder
        folder_name = 'test_stack_folder'

        stack_to_folder(stack, folder_name)
        loaded_stack = load_folder_images(folder_name)

        # Check that the loaded stack is the same as the original stack
        self.assertTrue(np.array_equal(stack, loaded_stack))

        # Delete the folder
        try:
            shutil.rmtree(folder_name)
        except:
            # warn that folder wasn't deleted 
            print(f"Warning, unable to delete folder: {folder_name}")


    def test_tif_stack_from_folder(self):

        # Generate a stack of random images as a 3D numpy array   
        stack = np.random.random((10,100,100)) * 65535
        stack = stack.astype(np.uint16)

        # Save the stack to a folder
        folder_name = 'test_stack_folder'     
        stack_to_folder(stack, folder_name)

        # Convert folder to tif stack
        folder_to_tif_stack(folder_name, 'test_stack.tif')
        loaded_stack = load_image('test_stack.tif')

        # Check that the loaded stack is the same as the original stack
        self.assertTrue(np.array_equal(stack, loaded_stack))

        # Delete the folder
        try:
            shutil.rmtree(folder_name)
        except:
            # warn that folder wasn't deleted 
            print(f"Warning, unable to delete folder: {folder_name}")

    


                




if __name__ == '__main__':
    unittest.main()  
        
