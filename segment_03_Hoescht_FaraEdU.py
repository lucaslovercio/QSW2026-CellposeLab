#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Google Colab ---
# In one cell, do: !pip install "cellpose<4.0"
# In another cell do: %matplotlib inline
# And below, in the same cell: %run segment_03_Hoescht_FaraEdU.py

import os

############################################################################
###################################   PARAMETERS   #########################
############################################################################

#File path to .tif or .tiff file
folder_root = 'images16bit_uncompressed'
filename_hoechst = 'slice_203.tiff'
filename_FaraEdU = 'slice_202.tiff'

diameter_nucleus = 500
diameter_chromatin = 20
diameter_FaraEdU = 20

#Path to trained architectures
folder_models = 'trained_models_16bit'
path_model_nuclei  = os.path.join(folder_models,'Hoechst_nuclei_diam_400_model_cyto2_ji_0.935.354306')
path_model_chromatin  = os.path.join(folder_models,'Hoechst_cromatin_diam_60_model_cyto_ji_0.6205.5470')
path_model_FaraEdU  = os.path.join(folder_models,'F-ara-EdU_model_cyto2_diam_50_ji_0.5317.386153')

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

        # si es multipágina
        if len(tif.pages) > 1:
            arr = tif.asarray()
        else:
            page = tif.pages[0]
            arr = page.asarray()

    return arr.astype(np.uint16)


arr16_hoechst = read_tiff_force(os.path.join(folder_root, filename_hoechst))
arr16_FaraEdU = read_tiff_force(os.path.join(folder_root, filename_FaraEdU))


if not os.path.isdir(folder_root):
    print(folder_root,"does not exist")
    sys.exit(1)
if not os.path.isdir(folder_models):
    print(folder_models,"does not exist")
    sys.exit(1)
if not os.path.exists(path_model_nuclei):
    print("Pretrained model not found. Stop execution.") 
    sys.exit(1)


# Using re-trained models
model_trained_nuclei = models.CellposeModel(pretrained_model=path_model_nuclei, gpu=flag_gpu)
nuclei_segmentation, _, _ = model_trained_nuclei.eval(arr16_hoechst, diameter=None, channels= [[0,0]])
del model_trained_nuclei

model_trained_chromatin = models.CellposeModel(pretrained_model=path_model_chromatin, gpu=flag_gpu)
chromatin_segmentation, _, _ = model_trained_chromatin.eval(arr16_hoechst, diameter=None, channels= [[0,0]])
del model_trained_chromatin

model_trained_faraedu = models.CellposeModel(pretrained_model=path_model_FaraEdU, gpu=flag_gpu)
faraedu_segmentation, _, _ = model_trained_faraedu.eval(arr16_FaraEdU, diameter=None, channels= [[0,0]])
del model_trained_faraedu



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

faraedu_segmentation_pretrained, _, _, _ = model_nuclei_pretrained.eval(
    arr16_FaraEdU,
    diameter=diameter_FaraEdU,          # fixed EdU diameter in pixels
    channels=[0, 0],      # single-channel grayscale image
    do_3D=False           # set True if Z-stack
)

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, constrained_layout = True)

plt.subplot(3, 3, 1)
plt.title('Hoechst')
plt.imshow(arr16_hoechst,cmap='gray')
plt.axis('off')  # Hide axis

plt.subplot(3, 3, 2)
plt.title('Nuclei seg pre-trained')
plt.imshow(nuclei_segmentation_pretrained,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.subplot(3, 3, 3)
plt.title('Nuclei seg re-trained')
plt.imshow(nuclei_segmentation,cmap='tab20b')
plt.axis('off')  # Hide axis


plt.subplot(3, 3, 4)
plt.title('Chromatin seg pre-trained')
plt.imshow(chromatin_segmentation_pretrained,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.subplot(3, 3, 5)
plt.title('Chromatin seg re-trained')
plt.imshow(chromatin_segmentation,cmap='tab20b')
plt.axis('off')  # Hide axis



plt.subplot(3, 3, 7)
plt.title('F-ara-EdU')
plt.imshow(arr16_FaraEdU,cmap='gray')
plt.axis('off')  # Hide axis

plt.subplot(3, 3, 8)
plt.title('F-ara-EdU seg pre-trained')
plt.imshow(faraedu_segmentation_pretrained,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.subplot(3, 3, 9)
plt.title('F-ara-EdU seg re-trained')
plt.imshow(faraedu_segmentation,cmap='tab20b')
plt.axis('off')  # Hide axis

plt.show()    
