#!/usr/bin/env python
__author__ = 'greg'
import os
import cPickle as pickle

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

results = pickle.load(open(base_directory+"/Databases/serengeti/results.pickle","rb"))

skipped = 0
non_empty = 0

for zooniverse_id in results:
    status = results[zooniverse_id]["status"]
    if status == "skipped":
        skipped += 1
        continue

    classification = results[zooniverse_id]["name"]



    if "deer" in classification:
        non_empty += 1
    elif "hyena" in classification:
        non_empty += 1
    elif ("wildebeest" in classification) or ("wildebeast" in classification):
        non_empty += 1
    elif "zebra" in classification:
        non_empty += 1
    elif "impala" in classification:
        non_empty += 1
    elif "gazelle" in classification:
        non_empty += 1
    elif "buffalo" in classification:
        non_empty += 1
    elif "elephant" in classification:
        non_empty += 1
    elif "animal" in classification:
        non_empty += 1
    elif "lion" in classification:
        non_empty += 1
    elif ("aardvark" in classification) or ("aadvark" in classification):
        non_empty += 1
    elif "horse" in classification:
        non_empty += 1
    elif ("birds" in classification) or ("bird" in classification):
        non_empty += 1
    elif "oryx" in classification:
        non_empty += 1
    elif "antelope" in classification:
        non_empty += 1
    elif "emu" in classification:
        non_empty += 1
    elif "fowl" in classification:
        non_empty += 1
    elif ("suv" in classification) or ("rover" in classification) or ("vehicle" in classification):
        non_empty += 1
    elif "peacock" in classification:
        non_empty += 1
    elif "dikdik" in classification:
        non_empty += 1
    elif "giraffe" in classification:
        non_empty += 1
    elif "bull" in classification:
        non_empty += 1
    elif ("jeans" in classification) or ("board" in classification) or ("pants" in classification) or ("jacket" in classification):
        non_empty += 1
    elif "cow" in classification:
        non_empty += 1
    elif "warthog" in classification:
        non_empty += 1
    elif "baboon" in classification:
        non_empty += 1
    elif "mammal" in classification:
        non_empty += 1
    elif "boar" in classification:
        non_empty += 1
    else:
        print classification


print skipped,non_empty