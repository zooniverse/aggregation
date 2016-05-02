#!/usr/bin/env python
from __future__ import print_function
import glob
from sklearn.decomposition import PCA
from active_weather import __extract_region__,__create_mask__,__otsu_bin__,__pca__,__mask_lines__,__ocr_image__
import matplotlib.pyplot as plt
import numpy as np
import cv2

table = __extract_region__("/home/ggdhines/eastwind-wag-279-1946_0523-0.JPG",(127,275,1467,1151))
pca_image, threshold, inverted = __pca__(table, __otsu_bin__)

plt.imshow(pca_image)
plt.show()