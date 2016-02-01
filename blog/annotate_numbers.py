__author__ = 'ggdhines'
import csv
import matplotlib.pyplot as plt

X = []
Y = []
with open('/home/ggdhines/numbers', 'rb') as csvfile:
    reader = csv.reader(csvfile,delimiter=" ")
    for l in reader:
        if len(l) > 1:
            x,y = l
            x = float(x)
            y = float(y)
            X.append(x)
            Y.append(y)

plt.plot(X,Y,".")
plt.show()