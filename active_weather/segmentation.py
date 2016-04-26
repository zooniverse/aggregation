#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
from active_weather import __extract_region__,__cell_boundaries__,__pca__,__mask_lines__,__create_mask__,__otsu_bin__,__binary_threshold_curry__
import glob
import matplotlib.pyplot as plt
import numpy as np

for fname in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:40]:
    fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0005.JPG"
    img = __extract_region__(fname)
    id_ = fname.split("/")[-1][:-4]
    print(id_)

    # set a baseline for performance with otsu's binarization
    horizontal_grid,vertical_grid = __cell_boundaries__(img)

    thres_alg = __binary_threshold_curry__(60)

    mask = __create_mask__(img)
    pca_image,threshold,inverted = __pca__(img,thres_alg)
    masked_image = __mask_lines__(pca_image,mask)

    plt.imshow(masked_image,cmap="gray")
    plt.show()

    for h_index in range(len(horizontal_grid)-1):
        row = masked_image[horizontal_grid[h_index]:horizontal_grid[h_index+1],:]
        for v_index in range(len(vertical_grid)-1):
            cell = row[:,vertical_grid[v_index]:vertical_grid[v_index+1]]

            if len(np.where(cell<255)[0]) < 30:
                continue

            x_pts = np.where(cell<255)[1]

            # plt.imshow(masked_image[:,vertical_grid[v_index]:vertical_grid[v_index+1]],cmap="gray")
            plt.imshow(cell,cmap="gray")
            plt.show()

            # print(x_pts)
            bucket_count,x_values = plt.hist(x_pts,range(0,int(vertical_grid[v_index+1])-int(vertical_grid[v_index])))[:2]
            # plt.xlim((int(vertical_grid[v_index]),int(vertical_grid[v_index+1])))
            # plt.show()

            forward_difference = [bucket_count[i+1]-bucket_count[i] for i in range(len(bucket_count)-1)]
            backward_difference = [bucket_count[i]-bucket_count[i-1] for i in range(1,len(bucket_count))]

            forward_difference.append(0)
            backward_difference.insert(0,0)

            central_difference = [forward_difference[i]-backward_difference[i] for i in range(1,len(bucket_count))]

            plt.plot(x_values[1:-1],central_difference)
            plt.show()