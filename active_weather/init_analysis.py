from __future__ import print_function
import sqlite3 as lite
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter("error", RuntimeWarning)

con = lite.connect('/home/ggdhines/to_upload3/active.db')
cur = con.cursor()

con2 = lite.connect('/home/ggdhines/to_upload4/active.db')
cur2 = con2.cursor()

cur.execute("select * from characters")

confidences = []

init_confidence = {}
updated_confidences = {}

same = 0

for count,r in enumerate(cur.fetchall()):
    original_chr = r[4]

    # lb_x int,ub_x int, lb_y int,ub_y int
    lb_x,ub_x,lb_y,ub_y = r[6:]
    cur2.execute("select * from characters where lb_x = " + str(lb_x) + " and lb_y = " + str(lb_y))
    r2 = cur2.fetchone()
    if r2 == None:
        continue
    updated_chr = r2[4]

    if original_chr == updated_chr:
        same += 1

        c1 = r[5]
        c2 = r2[5]
        print(c1,c2)

        try:
            init_confidence[original_chr].append(c1)
            updated_confidences[original_chr].append(c2)
        except KeyError:
            init_confidence[original_chr] = [c1]
            updated_confidences[original_chr] = [c2]

print(same,count)

x = []
y = []
for c in init_confidence:
    x.append(np.mean(init_confidence[c]))
    y.append(np.mean(updated_confidences[c]))

plt.plot(x,y,"o")
plt.plot([0,100],[0,100])

plt.show()

