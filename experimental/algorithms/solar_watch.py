#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import matplotlib.pyplot as plt
import datetime
import os
import urllib
import urllib2
from PIL import Image


if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

# connect to the mongo server
client = pymongo.MongoClient()
db = client['sunspot_2015-08-29']
classification_collection = db["sunspot_classifications"]

correct = 0
total = 100
X1 = []
X2 = []

for ii,classification in enumerate(classification_collection.find({"tutorial":{"$ne":True},"created_at":{"$gte":datetime.datetime(2015,1,1)}}).limit(total)):
    print ii
    ids = [str(classification["subjects"][i]["id"]) for i in range(2)]
    selected = classification["annotations"][0]["selected_id"]

    i1 = [index for index in range(2) if ids[index] == selected][0]
    i2 = [index for index in range(2) if ids[index] != selected][0]

    url = classification["subjects"][i1]["location"]["standard"]

    slash_index = url.rfind("/")
    fname1 = url[slash_index+1:]

    image1 = base_directory+"/Databases/images/"+fname1


    if not(os.path.isfile(image1)):
        # urllib.urlretrieve(url, image_path)
        image = urllib2.urlopen(url)
        output = open(image1,'wb')
        output.write(image.read())
        output.close()

    im1 = Image.open(image1)
    im1.save(base_directory+"/Databases/images/_"+fname1,"JPEG", quality=50)

    # and repeat

    url = classification["subjects"][i2]["location"]["standard"]

    slash_index = url.rfind("/")
    fname2 = url[slash_index+1:]

    image2 = base_directory+"/Databases/images/"+fname2



    if not(os.path.isfile(image2)):
        urllib.urlretrieve(url, image2)

    im2 = Image.open(image2)
    im2.save(base_directory+"/Databases/images/_"+fname2,"JPEG", quality=50)

    # if os.path.getsize(base_directory+"/Databases/images/_"+fname1) > os.path.getsize(base_directory+"/Databases/images/_"+fname2):
    if i1 == 0:
        correct += 1
        X1.append(os.path.getsize(base_directory+"/Databases/images/_"+fname1)/float(os.path.getsize(base_directory+"/Databases/images/_"+fname2)))
    else:
        X2.append(os.path.getsize(base_directory+"/Databases/images/_"+fname1)/float(os.path.getsize(base_directory+"/Databases/images/_"+fname2)))

plt.hist(X1, bins=10)
plt.hist(X2, bins=10)

print correct/float(total)
plt.show()