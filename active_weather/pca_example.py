from __future__ import print_function
import glob
from sklearn.decomposition import PCA
from active_weather import __extract_region__,__create_mask__,__otsu_bin__,__pca__,__mask_lines__,__ocr_image__
import matplotlib.pyplot as plt
import numpy as np
import cv2

def __run__(table,mask):
    masked_image = __mask_lines__(table,mask)

    transcriptions = __ocr_image__(masked_image)

    return transcriptions

example_characters = []

bad_examples = []
tr = []

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
        if (conf > 80):
            char_bitmap = pca_image[a:d,b:c]
            char_bitmap = cv2.resize(char_bitmap,(28,28))
            if character == "6":
                example_characters.append(char_bitmap.flatten())
            else:
                bad_examples.append(char_bitmap.flatten())
                tr.append(character)

    break

example_characters = np.asarray(example_characters)
print(example_characters.shape)

pca = PCA(n_components=1)
X_r = pca.fit(example_characters).transform(example_characters)
print(X_r)
print(pca.explained_variance_ratio_)
#
# index = 155
# print(tr.index('8'))
# print(tr[index])
#
# index2 = -2
# r = np.random.rand(784)
# r += 0.5
# r = r.astype(int)
# print(r)
#
# # first_step = pca.transform(r)
# # t = example_characters[index2].reshape((28,28))
# # plt.imshow(r.reshape(28,28),cmap="gray")
# # plt.show()
# # print(first_step)
# # inverse = pca.inverse_transform(first_step)
# # inverse = inverse.reshape((28,28))
# # plt.imshow(inverse,cmap="gray")
# # plt.show()





for fname in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[5:]:
    table = __extract_region__(fname)
    pca_image, threshold, inverted = __pca__(table, __otsu_bin__)
    mask = __create_mask__(table)

    masked_image = __mask_lines__(pca_image, mask)

    widths = []
    heights = []
    _, contour, hier = cv2.findContours(masked_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt, hi in zip(contour, hier[0]):
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        widths.append(w)
        heights.append(h)

    desired_width = int(np.percentile(widths,90))
    desired_height = int(np.percentile(heights,90))

    print(desired_width,desired_height)

    table = np.zeros((table.shape[0],table.shape[1],3),np.uint8)
    table[:,:,0] = masked_image
    table[:,:,1] = masked_image
    table[:,:,2] = masked_image

    # plt.imshow(pca_image)
    # plt.show()

    for i in range(0,table.shape[0]-desired_height,2):
        for j in range(0,table.shape[1]-desired_width,2):
            subimage = pca_image[i:i+desired_height,j:j+desired_width]
            foreground_pixels = np.where(subimage==0)
            if foreground_pixels[0].shape == (0,):
                continue
            # print(foreground_pixels)
            #
            # print(np.min(foreground_pixels,axis=1))
            # print(np.max(foreground_pixels,axis=1))
            bottom,left = np.min(foreground_pixels,axis=1)
            top,right = np.max(foreground_pixels,axis=1)

            # plt.imshow(subimage,cmap="gray")
            # plt.show()
            subimage = subimage[bottom:top+1,left:right+1]
            # plt.imshow(subimage, cmap="gray")
            # plt.show()

            subimage = cv2.resize(subimage,(28,28))



            v = pca.transform(subimage.flatten())
            assert isinstance(v[0][0],float)
            if v[0][0] > 0:
                print(v[0][0])
                # print((i,j),(i+desired_height,j+desired_width))
                table = cv2.rectangle(table,(j,i),(j+desired_width,i+desired_height),color=(255,255,0),thickness=2)
    plt.imshow(table,cmap="gray")
    plt.show()


