from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
from active_weather import ActiveWeather
from tesseract_font import ActiveTess
import matplotlib.pyplot as plt
import cv2
import numpy as np
directory = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"

# min_x,max_x,min_y,max_y
region_bounds = (559,3282,1276,2097)
project = ActiveWeather()

project.cass_db.__get_subjects__()

horizontal_grid,vertical_grid = project.__get_grid_for_table__(directory,region_bounds)

horizontal_lines = []
for h in horizontal_grid:
    lb = np.min(h,axis=0)[1]-region_bounds[2]
    ub = np.max(h,axis=0)[1]-region_bounds[2]

    horizontal_lines.append((lb,ub))


transcriber = ActiveTess()
fname = directory+"Bear-AG-29-1940-0720.JPG"
print(fname)
image = cv2.imread(fname)

plt.imshow(image)
plt.show()

sub_image = image[region_bounds[2]:region_bounds[3],region_bounds[0]:region_bounds[1]]

plt.imshow(sub_image)


# boxes,text = transcriber.__process_image__(image)
#
# plt.imshow(image)
#
# for (lb,ub) in horizontal_lines:
#     plt.plot([0,2725],[lb,lb])
#     plt.plot([0,2725],[ub,ub])
# plt.show()


for h in horizontal_grid:
    # print(h)
    h = h-(region_bounds[0],region_bounds[2])
    cv2.polylines(sub_image,[h],True,(0,255,255))
    plt.plot(h[:,0]-region_bounds[0],h[:,1]-region_bounds[2])
cv2.imwrite("/home/ggdhines/test.jpg",sub_image)

# for bb,t in zip(boxes,text):
#     x = np.asarray([bb.left,bb.left,bb.right,bb.right,bb.left])
#     y = np.asarray([bb.top,bb.bottom,bb.bottom,bb.top,bb.top])
#     plt.plot(x,y)

# plt.show()