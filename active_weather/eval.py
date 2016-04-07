import cv2
import csv
import matplotlib.pyplot as plt
__author__ = 'ggdhines'

img = cv2.imread("/home/ggdhines/tmp.jpg")
plt.imshow(img)

with open("/home/ggdhines/temp.box","r") as f:
    reader = csv.reader(f, delimiter=' ')

    for _,a,b,c,d,_ in reader:
        plt.plot([a,a,c,c,a],[b,d,d,b,b],color="blue")

plt.show()