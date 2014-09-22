#!/usr/bin/env python
__author__ = 'greghines'
import matplotlib.pyplot as plt

plt.plot(range(1,11),[100 - 0.096, 100 - 1.73336112269, 100 - 3.01705224615, 100 - 4.00546764974,100 - 5.11950255846,100 - 5.89339988633,100 - 6.59206947559,100 - 7.26615443629,100 - 7.71638432277, 100 - 8.13761313171],'-o',color="black")
plt.xlabel("Actual Number of Animals in Photo")
plt.ylabel("Percentage of Classifications with Correct Number of Animals")
plt.ylim(90,100)
plt.show()


