import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
import numpy as np

matplotlib.rcParams['font.size'] = 9


# img = img_as_ubyte(data.page())
img = cv2.imread("/home/ggdhines/1.jpg",0)

radius = 100
selem = disk(radius)

local_otsu = rank.otsu(img, selem)
threshold_global_otsu = threshold_otsu(img)
global_otsu = img >= threshold_global_otsu


# fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
# ax1, ax2, ax3, ax4 = ax.ravel()
#
# fig.colorbar(ax1.imshow(img, cmap=plt.cm.gray),
#              ax=ax1, orientation='horizontal')
# ax1.set_title('Original')
# ax1.axis('off')
#
# fig.colorbar(ax2.imshow(local_otsu, cmap=plt.cm.gray),
#              ax=ax2, orientation='horizontal')
# ax2.set_title('Local Otsu (radius=%d)' % radius)
# ax2.axis('off')
#
# ax3.imshow(img >= local_otsu, cmap=plt.cm.gray)
# ax3.set_title('Original >= Local Otsu' % threshold_global_otsu)
# ax3.axis('off')
#
# ax4.imshow(global_otsu, cmap=plt.cm.gray)
# ax4.set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
# ax4.axis('off')
#
# plt.show()

x,y = np.where(img>=local_otsu)
template = np.zeros(img.shape,np.uint8)
template.fill(255)
template[x,y] = 0

cv2.imwrite("/home/ggdhines/local.jpg",template)
# print global_otsu

x,y = np.where(global_otsu)
template = np.zeros(img.shape,np.uint8)
template.fill(255)
template[x,y] = 0

cv2.imwrite("/home/ggdhines/global.jpg",template)
# plt.imshow(img>=local_otsu,cmap="gray")
# plt.imshow(img>=local_otsu,cmap="gray")
# plt.show()
#
# plt.imshow(global_otsu)
# plt.show()