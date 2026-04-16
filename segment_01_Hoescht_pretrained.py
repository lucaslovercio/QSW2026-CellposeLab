#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Google Colab ---
# In one cell, do: !pip install "cellpose<4.0"
# In another cell do: %matplotlib inline

import os

############################################################################
###################################   PARAMETERS   #########################
############################################################################

#File path to .tif or .tiff file
folder_root = 'images16bit_uncompressed'
filename_hoechst = 'slice_203.tiff'

diameter_nucleus = 500
diameter_chromatin = 20

#Parameter for running the segmentation in CPU or GPU
flag_gpu = False

############################################################################
############################################################################
############################################################################

import os
import sys
import tifffile as tiff
import matplotlib.pyplot as plt
from cellpose import models
import numpy as np


def read_tiff_force(path):
    with tiff.TiffFile(path) as tif:

        # Check multipage
        if len(tif.pages) > 1:
            arr = tif.asarray()
        else:
            page = tif.pages[0]
            arr = page.asarray()

    return arr.astype(np.uint16)

arr16_hoechst = read_tiff_force(os.path.join(folder_root, filename_hoechst))

if not os.path.isdir(folder_root):
    print(folder_root,"does not exist")
    sys.exit(1)


# Using pretrained nuclei model
model_nuclei_pretrained = models.Cellpose(model_type='nuclei')

chromatin_segmentation_pretrained, _, _, _ = model_nuclei_pretrained.eval(
    arr16_hoechst,
    diameter=diameter_chromatin,          # fixed chromatin diameter in pixels
    channels=[0, 0],      # single-channel grayscale image
    do_3D=False           # set True if Z-stack
)

nuclei_segmentation_pretrained, _, _, _ = model_nuclei_pretrained.eval(
    arr16_hoechst,
    diameter=diameter_nucleus,          # fixed nucleus diameter in pixels
    channels=[0, 0],      # single-channel grayscale image
    do_3D=False           # set True if Z-stack
)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout = True)

plt.subplot(2, 2, 1)
plt.title('Hoechst')
plt.imshow(arr16_hoechst,cmap='gray')
plt.axis('off')  # Hide axis

plt.subplot(2, 2, 2)
plt.title('Nuclei seg pre-trained')
plt.imshow(nuclei_segmentation_pretrained,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.subplot(2, 2, 3)
plt.title('Chromatin seg pre-trained')
plt.imshow(chromatin_segmentation_pretrained,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.show()    
