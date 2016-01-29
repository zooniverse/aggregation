#!/usr/bin/env python
__author__ = 'greg'
import os
import matplotlib.pyplot as plt
import urllib
import matplotlib.cbook as cbook
eggs = open("/Users/greg/Databases/MAIVb2013_egg_RAW.csv","rb")
chicks = open("/Users/greg/Databases/MAIVb2013_chick_RAW.csv","rb")

# for Greg - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
    code_directory = base_directory + "/github"
elif os.path.exists("/Users/greg"):
    base_directory = "/Users/greg"
    code_directory = base_directory + "/Code"
else:
    base_directory = "/home/greg"
    code_directory = base_directory + "/github"

while True:
    l1 = eggs.readline()
    l2 = chicks.readline()

    if not(l1) or not(l2):
        break

    egg_image = l1.split(",")[0]
    chick_image = l2.split(",")[0]
    assert egg_image == chick_image
    #print image_fname
    chick_pts = []
    egg_pts = []
    if len(l1.split("\"")) > 1:
        gold_string = l1.split("\"")[1]
        gold_markings = gold_string[:-2].split(";")
        egg_pts = [m.split(",")[:2] for m in gold_markings]
    if len(l2.split("\"")) > 1:
        gold_string = l2.split("\"")[1]
        gold_markings = gold_string[:-2].split(";")
        chick_pts = [m.split(",")[:2] for m in gold_markings]

    if (chick_pts != []) or (egg_pts != []):
        print egg_image
        egg_pts = [(int(x),int(y)) for (x,y) in egg_pts]
        chick_pts = [(int(x),int(y)) for (x,y) in chick_pts]
        slash_index = egg_image.rfind("/")
        fname = egg_image[slash_index+1:]
        if not(os.path.isfile(base_directory+"/Databases/penguin/images/"+fname)):
            urllib.urlretrieve(egg_image, base_directory+"/Databases/penguin/images/"+fname)

        image_file = cbook.get_sample_data(base_directory+"/Databases/penguin/images/"+fname)
        image = plt.imread(image_file)

        fig, ax = plt.subplots()
        im = ax.imshow(image)

        for (x,y) in egg_pts:
            plt.plot([x,],[y,],"o",color="red")
        for (x,y) in chick_pts:
            plt.plot([x,],[y,],"o",color="blue")

        plt.show()