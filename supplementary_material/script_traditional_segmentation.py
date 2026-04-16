#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Google Colab ---
# In one cell, do: !pip install "cellpose<4.0"
# In another cell do: %matplotlib inline
# And below, in the same cell: %run script_traditional_segmentation.py

############################################################################
###################################   PARAMETERS   #########################
############################################################################

#File path to .tif or .tiff file
folder_root = 'images16bit_uncompressed'
filename_hoechst = 'slice_203.tiff'
filename_FaraEdU = 'slice_202.tiff'

############################################################################
############################################################################
############################################################################

import os
from scipy import ndimage as ndi
from skimage.measure import label
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)

def read_tiff_force(path):
    with tiff.TiffFile(path) as tif:

        # multipage?
        if len(tif.pages) > 1:
            arr = tif.asarray()
        else:
            page = tif.pages[0]
            arr = page.asarray()

    return arr.astype(np.uint16)


arr16_hoechst = read_tiff_force(os.path.join('..',folder_root, filename_hoechst))
arr16_FaraEdU = read_tiff_force(os.path.join('..',folder_root, filename_FaraEdU))

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------
# arr16_hoechst: 2D uint16 NumPy array
# shape = (H, W)

# Sanity check (optional)
assert arr16_hoechst.ndim == 2
assert arr16_hoechst.dtype == np.uint16

# ------------------------------------------------------------
# Pipeline 1: Gaussian filter (~20x20) -> threshold -> label
# ------------------------------------------------------------

# Approximate sigma for ~20x20 window (kernel ~ 6*sigma)
sigma_gaussian = 3.5

gaussian_filtered = ndi.gaussian_filter(
    arr16_hoechst,
    sigma=sigma_gaussian
)

binary_gaussian = gaussian_filtered > 300

labels_gaussian, _ = label(
    binary_gaussian,
    return_num=True,
    connectivity=1  # 4-connectivity
)


# ------------------------------------------------------------
# Pipeline 2: Median filter (5x5) -> threshold -> label
# ------------------------------------------------------------

median_filtered = ndi.median_filter(
    arr16_hoechst,
    size=(5, 5)
)

binary_median = median_filtered > 1700

labels_median, _ = label(
    binary_median,
    return_num=True,
    connectivity=1
)

# ------------------------------------------------------------
# Pipeline 3: Median filter (5x5) -> threshold -> label
# ------------------------------------------------------------

arr16_FaraEdU_filtered = ndi.median_filter(
    arr16_FaraEdU,
    size=(5, 5)
)

arr16_FaraEdU_median = arr16_FaraEdU_filtered > 500

arr16_FaraEdU_labels, _ = label(
    arr16_FaraEdU_median,
    return_num=True,
    connectivity=1
)


fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, constrained_layout = True)

plt.subplot(2, 3, 1)
plt.title('Hoechst')
plt.imshow(arr16_hoechst,cmap='gray')
plt.axis('off')  # Hide axis

plt.subplot(2, 3, 2)
plt.title('Nuclei seg filt+th')
plt.imshow(labels_gaussian,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.subplot(2, 3, 3)
plt.title('Chromatin seg filt+th')
plt.imshow(labels_median,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.subplot(2, 3, 4)
plt.title('F-ara-EdU')
plt.imshow(arr16_FaraEdU,cmap='gray')
plt.axis('off')  # Hide axis

plt.subplot(2, 3, 5)
plt.title('F-ara-EdU seg filt+th')
plt.imshow(arr16_FaraEdU_labels,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.show()    
