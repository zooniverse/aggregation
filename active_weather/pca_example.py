from __future__ import print_function
import glob
from sklearn.decomposition import PCA
from active_weather import __extract_region__,__create_mask__,__cell_boundaries__,__otsu_bin__,__pca__,__mask_lines__,__ocr_image__
import matplotlib.pyplot as plt
import numpy as np
import cv2

def __run__(table,mask):
    masked_image = __mask_lines__(table,mask)

    transcriptions = __ocr_image__(masked_image)

    return transcriptions

example_characters = []

for fname in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:4]:
    # fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0329.JPG"
    table = __extract_region__(fname)
    pca_image, threshold, inverted = __pca__(table, __otsu_bin__)
    id_ = fname.split("/")[-1][:-4]
    print(id_)

    # set a baseline for performance with otsu's binarization
    mask = __create_mask__(table)

    transcriptions = __run__(pca_image, mask)

    for character,conf,a,b,c,d in transcriptions:
        if (conf > 80) and (character == "6"):
            char_bitmap = pca_image[a:d,b:c]
            char_bitmap = cv2.resize(char_bitmap,(28,28))
            example_characters.append(char_bitmap.flatten)

    break

print(len(example_characters))
example_characters = np.asarray(example_characters)
print(example_characters.shape)