"""
This module contains network configuration settings and hyper-parameters
"""

# Hyper-Parameters
batch_size = 90
learning_rate = 0.00008
number_of_epochs = 10

# Image and patch properties
# patch_startX and patch_startY control the location of the patch to be masked in the training images
channels = 3
patch_startX = 32
patch_startY = 94
patch_width = 16

# Network input and output sizes. Changing these requires altering the network itself so that the
# input and output dimensions remain correct
G_patch_margin_size = 16
D_patch_margin_size = 8
G_input_size = G_patch_margin_size * 2 + patch_width
D_input_size = 32
G_output_size = 32

# Root location for datasets and output images, and the path containing the learnt model
root = "./"
model_path = 'model/'
faces_model_path = 'faces_model/'

