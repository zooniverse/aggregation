__author__ = 'greghines'
import csv
import os

class independence:
    def __init__(self,userTemplate,photoTemplate):
        self.userNodes = []
        self.photoNodes = []

        self.user_id_list = []
        self.photo_id_list = []

        self.limit = 10

        self.userTemplate = userTemplate
        self.photoTemplate = photoTemplate

        if os.path.isdir("/Users/greghines/Databases/serengeti"):
            self.baseDir = "/Users/greghines/Databases/serengeti/"
        else:
            self.baseDir = "/home/ggdhines/Databases/serengeti/"

    def __readin_gold__(self):
        print("Reading in expert classification")
        reader = csv.reader(open(self.baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
        next(reader, None)

        for row in reader:
            photoStr = row[2]
            species = row[12]

            try:
                photoIndex = self.photo_id_list.index(photoStr)
            except ValueError:
                continue
            photoNode = self.photoNodes[photoIndex]

            photoNode.__updateGoldStandard__(species)