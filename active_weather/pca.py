#!/usr/bin/env python
from __future__ import print_function
from sklearn.decomposition import PCA
import cv2
import numpy as np
import matplotlib.pyplot as plt
import paper_quad

__author__ = 'ggdhines'

table = cv2.imread("/home/ggdhines/region.jpg")
pca = PCA(n_components=1)
s = table.shape
flatten_table = np.reshape(table,(s[0]*s[1],3))

X_r = pca.fit_transform(flatten_table)
background = max(X_r)[0]
foreground = 0

pca_table = np.reshape(X_r,s[:2])
plt.imshow(pca_table)
plt.show()
print(pca_table)
# ink_pixels = np.where(pca_table > 0)
# plt.plot(ink_pixels[1],-ink_pixels[0],".")
# plt.show()

gray = cv2.cvtColor(table,cv2.COLOR_BGR2GRAY)

horizontal_lines = paper_quad.__extract_grids__(gray,True)
vertical_lines = paper_quad.__extract_grids__(gray,False)
for l in horizontal_lines:
    corrected_l = paper_quad.__correct__(gray,l,True,background,foreground)
    plt.imshow(corrected_l)
    plt.show()
    pca_table = np.max([pca_table,corrected_l],axis=0)

plt.imshow(pca_table)
plt.show()