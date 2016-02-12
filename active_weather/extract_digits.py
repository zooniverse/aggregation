import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.cluster import DBSCAN
import Image
import cv2


def line_removal(pts,num_col,num_row):
    global_tree = spatial.KDTree(pts)
    plt.plot(num_row,num_col,"o",color="red")


def extract(image):
    digits = []
    confidence = []
    x_location = []
    assert isinstance(image,np.ndarray)
    num_col,num_row = image.shape

    # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # blur = cv2.GaussianBlur(image,(5,5),0)
    _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pts = np.where(image>0)

    # hopefully corresponds to an empty cell
    if pts == []:
        return [],[]

    rows,columns = pts
    min_r =min(rows)
    max_r =max(rows)

    min_c =min(columns)
    max_c =max(columns)

    # pts = np.asarray([(r,c) for (r,c) in pts if (r>(min_r))and(r<(max_r-2))and(c>(min_c+2))and(c<(max_c-2))])
    pts = np.asarray([(r,c) for (r,c) in zip(rows,columns) if (r>=min_r)and(r<=max_r)and(c>=min_c)and(c<=max_c)])

    db = DBSCAN(eps=3, min_samples=5).fit(pts)
    labels = db.labels_
    unique_labels = set(labels)

    digits = []

    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            col = 'k'
            continue
        else:
            col = "blue"
        class_member_mask = (labels == k)

        xy = pts[class_member_mask]
        digits.append(xy)

    # sort
    digits.sort(key = lambda c:np.mean(c[:,0]))
    return digits