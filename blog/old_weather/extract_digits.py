import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import spatial


def line_removal(pts,num_col,num_row):
    global_tree = spatial.KDTree(pts)
    plt.plot(num_row,num_col,"o",color="red")


def extract(image):
    assert isinstance(image,np.ndarray)
    num_col,num_row,_ = image.shape
    print num_col,num_row

    colours = {}

    for c in range(num_col):
        for r in range(num_row):
            pixel_colour = tuple(image[c,r])
            if pixel_colour not in colours:
                colours[pixel_colour] = 1
            else:
                colours[pixel_colour] += 1

    most_common_colour,_ = sorted(colours.items(),key = lambda x:x[1],reverse=True)[0]
    pts = []
    for c in range(num_col):
        for r in range(num_row):
            pixel_colour = tuple(image[c,r])

            dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(pixel_colour,most_common_colour)]))

            if dist > 100:
                plt.plot(r,c,"o",color="black")
                pts.append((r,c))

    plt.xlim((0,num_row))
    plt.ylim((num_col,0))
    line_removal(pts,num_col,num_row)
    plt.savefig("/home/ggdhines/tmp.png")
    plt.close()