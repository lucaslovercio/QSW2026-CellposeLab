# QSW 2026 - Cellpose Lab

To run the scripts in a Google Colab cell, and view the results, you must include the line "matplotlib inline", for example:

%matplotlib inline

%run segment_01_Hoescht_pretrained.py

## Main scripts

### segment_01_Hoescht_pretrained.py

Example of the usage of Cellpose to segment nucleus and chromatin using its "nuclei" pretrained model, just changing the diameter parameter.

### segment_02_Hoescht_retrained.py

To show the improvement in the segmentation if a pretrained model is trained for a specific task.

### segment_03_Hoescht_FaraEdU.py

Full example of the usage of Cellpose to segment different objects in these images, using their pre-trained models and re-trained models.

## Materials

Please find the trained architectures and sample images in https://drive.google.com/drive/folders/1YNc_E83yyK51SeKf75nHs1kLTsTLHEYb?usp=sharing

The sample images were extracted from the publicly available dataset from:

Batty, P., Langer, C.C., Takács, Z. et al. Cohesin‐mediated DNA loop extrusion resolves sister chromatids in G2 phase. EMBO J 42, EMBJ2023113475 (2023).
https://link.springer.com/article/10.15252/embj.2023113475

