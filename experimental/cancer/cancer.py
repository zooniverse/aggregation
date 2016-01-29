#!/usr/bin/env python
__author__ = 'greghines'
import matplotlib.pyplot as plt
import csv

cc = 0

with open("/home/greg/Databases/classifications.csv","rb") as f:
    reader = csv.reader(f,delimiter="\t")
    for row in reader:
        run_id = row[2]

        if run_id != "MB-7017Chrom13.txt":
            #print run_id
            continue

        cc += 1
        print cc

        if cc == 20:
            break

        lines = row[3]
        end_pts = []
        for pts in lines.split("x:")[1:]:
            x,y = pts.split(",")[:2]
            x = float(x)
            y = float(y.split(":")[1].split("}")[0])
            end_pts.append((x,y))


        if len(end_pts) <= 2:
            continue

        for i in range(len(end_pts)/2):
            x1,y1 = end_pts[2*i]
            x2,y2 = end_pts[2*i+1]

            if y1 == 0.3:
                continue

            plt.plot((x1,x2),(y1,y2),color = "blue")



plt.show()