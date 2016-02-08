"""
Align all images in a given directory using a given reference image (i.e. we will align to the template in the
reference image). The template is what a subject looks like without any enteries in it
Use subprocess because opencv seems to have recurring seg faults which occur at random
"""
import os
import subprocess
import numpy

image_directory = "/home/ggdhines/Databases/old_weather/images/"
reference_image = "Bear-AG-29-1940-0005.JPG"




f_index = 0
ship = "Bear"
year = "1940"

image_directory += ship + "/" + year + "/"
log_pages = list(os.listdir(image_directory))

no_alignment = []
errors = []
with open("/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year+"/no_alignment.txt","r") as f:
    for l in f.readlines():
        no_alignment.append(l[:-1])

with open("/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year+"/algebra_problem.txt","r") as f:
    for l in f.readlines():
        errors.append(l[:-1])

# go through at most 100 images in the given directory
for f_name in log_pages:
    # if this particular file is an image file - avoids the hidden files, i.e. starting with .
    if f_name.endswith(".JPG"):
        # already know that there is no alignment
        if (f_name in no_alignment) or (f_name in errors):
            continue

        # if we have already aligned this image - skip it
        if os.path.isfile("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"+f_name):
            f_index += 1
            print "skipping " + str(f_name)
            continue

        print image_directory+f_name

        # call on the template_align scripts
        success = False
        for i in range(3):
            try:
                t = subprocess.check_output(["python","template_align.py",image_directory,f_name,reference_image,"Bear","1940"])
                success = True
                break
            except subprocess.CalledProcessError:
                print "segfault"


        if not success:
            with open("/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year+"/error.txt","a") as f:
                f.write(image_directory+f_name+"\n")