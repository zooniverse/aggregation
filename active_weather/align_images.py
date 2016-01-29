"""
Align all images in a given directory using a given reference image (i.e. we will align to the template in the
reference image). The template is what a subject looks like without any enteries in it
Use subprocess because opencv seems to have recurring seg faults which occur at random
"""
import os
import subprocess

image_directory = "/home/ggdhines/Databases/old_weather/test_cases/"
reference_image = "Bear-AG-29-1939-0191.JPG"

log_pages = list(os.listdir(image_directory))


f_index = 0

# go through at most 100 images in the given directory
while f_index < min(100,len(log_pages)):
    f_name = log_pages[f_index]

    # if this particular file is an image file - avoids the hidden files, i.e. starting with .
    if f_name.endswith(".JPG"):
        # if we have already aligned this image - skip it
        if os.path.isfile("/home/ggdhines/Databases/old_weather/aligned_images/"+f_name):
            f_index += 1
            print "skipping " + str(f_name)
            continue

        print image_directory+f_name

        # call on the template_align scripts
        try:
            t = subprocess.check_output(["python","template_align.py",image_directory+f_name,image_directory+reference_image])
            # only move on to the next image if there is not a seg fault
            f_index += 1
        except subprocess.CalledProcessError:
            print "segfault"