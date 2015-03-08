"""
Python code for accessing the classifications/markings/annotations for different Zooniverse projects hosted on
Ouroboros. The raw data for Ouroboros is stored in MongoDB so that is what this code uses. Technically we could also
use the csv files that are output (and that may be something to do for future work) but working directly with the
MongoDB database seems to be the best way. For Panoptes, the hope is that this one file will be replaced and
everything else can work just fine.

"""

import abc
import csv

class MarkingAnnotations():
    """
    For projects where the user marks regions or points of interest - the most common kind of marking is a 2
    dimensional one (just x and y markings) so that will be the default. Other kinds - such as ellipse markings which
    are 5 dimensional or blobs (Floating Forests) - will have to override the classification_to_markings code
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,scale=1):
        """
        :param scale: Images are often scaled between the original image given by the scientists and the
         images shown to the volunteers. If the scientists enter the gold standard data via. Ouroboros we can
         pretty much ignore the scaling but if we get the gold standard from somewhere else - say a csv file
         that the scientists have personally created, we may need to scale things so that the gold standard and
         the user data match up.
        """
        self.scale = scale

        # if we have subject dependent rois
        self.roi_dict = {}


    def __load_roi__(self,classification):
        """
        roi is the region of interest - if a point lies outside of this region, we will ignore it
        the default region of interest is the whole image - since we don't know in advance what the image
        dimensions are - and since they can change even for one project, just create an infinite ROI
        note that all of the images in multiple projects seem to be flipped along the y axis - so we are actually
        looking for points above the ROI line

        :return a list of piecewise lines giving the ROI
        """
        return [(-float("inf"),-float("inf")),(float("inf"),-float("inf"))]

    @abc.abstractmethod
    def __classification_to_markings__(self,classification):
        """
        This is the main function projects will have to override - given a classification, we need to return the list
        of all markings in that classification
        """
        return []

    def __list_markings__(self, classification):
        """
        :param classification is the classification we are parsing
        :return yield the list of all markings for this classification, scaled as necessary and having removed
        any and all points outside the ROI. Only works for two dimensions named "x" and "y".
        each marking will contain the x and y coordinates and what animal_type it is. For more detailed information
        such as tag number (e.g. condor watch) - you will have to override this function
        """
        marks_list = self.__classification_to_markings__(classification)
        roi = self.__load_roi__(classification)

        for mark in marks_list:
            assert isinstance(mark,dict)
            assert "x" in mark
            assert "y" in mark

            x = float(mark["x"])*self.scale
            y = float(mark["y"])*self.scale

            animal_type = None
            # animal_type is usually provided by sometimes by error can be missing - seems better to not just ignore
            # these markings, since otherwise they are perfectly reasonably. If "animal" is given using a different
            # key, this will need to be updated.
            if "animal" in mark:
                animal_type = mark["animal"]

            #find which line segment on the roi the point lies on (x-axis wise)
            for segment_index in range(len(roi)-1):
                if (roi[segment_index][0] <= x) and (roi[segment_index+1][0] >= x):
                    rX1,rY1 = roi[segment_index]
                    rX2,rY2 = roi[segment_index+1]

                    m = (rY2-rY1)/float(rX2-rX1)
                    rY = m*(x-rX1)+rY1

                    if y >= rY:
                        # we have found a valid marking
                        # create a special type of animal None that is used when the animal type is missing
                        # thus, the marking will count towards not being noise
                        # but will not be used when determining the type

                        yield (x,y),animal_type
                        break
                    else:
                        break

class PenguinAnntations(MarkingAnnotations):
    def __init__(self,subject_collection, roi_file):
        """
        :param subject_collection: need to read in all of the subjects first because we need to load the ROIs in advance
         not ideal but having tried a bunch of different approaches, this seems the best of some not great options
         definitely something to look at in the future
        :param roi_file: the file containing all of the regions of interest for penguins. To open it requires a path
        name that is dependent on which computer you are using (the file itself is from the Penguin repo). Seems
        best to minimize the spots where you have to refer to the directory structure of the local computer so you can
        do this somewhere else
        :return:
        """
        MarkingAnnotations.__init__(self)

        self.subject_collection = subject_collection

        # load the roi for the different sites
        with open(aggregation.github_directory+"/Penguins/public/roi.tsv","rb") as roiFile:
            roiFile.readline()
            reader = csv.reader(roiFile,delimiter="\t")
            for l in reader:
                path = l[0]
                t = [r.split(",") for r in l[1:] if r != ""]
                self.roi_dict[path] = [(int(x)/1.92,int(y)/1.92) for (x,y) in t]

    def __classification_to_markings__(self,classification):
        annotations = classification["annotations"]

        for ann in annotations:
            if ("key" in ann) and (ann["key"] == "marking"):
                try:
                    return ann["value"].values()
                except (AttributeError,KeyError) as e:
                    #print classification
                    return []

        return []


        for v in values_list:
            print v
            if "marking" in v:
                try:
                    return v[1].values()
                except AttributeError:
                    # something about this set of markings is wrong
                    return []

        return []

    def __list_markings__(self, classification):
        marks_list = self.__classification_to_markings__(classification)

        try:
            for mark in marks_list:
                x = float(mark["x"])*self.scale
                y = float(mark["y"])*self.scale

                if not("animal" in mark):
                    animal_type = None
                else:
                    animal_type = mark["animal"]

                yield (x,y),animal_type
        except ValueError:
            print classification
            raise

    def __load_roi__(self,classification):
        zooniverse_id = classification["subjects"][0]["zooniverse_id"]
        # takes the zooniverse id and converts it into a the site code
        subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})
        path = subject["metadata"]["path"]
        slashIndex = path.find("/")
        underscoreIndex = path.find("_")
        site = path[slashIndex+1:underscoreIndex]


        assert (site in self.roi_dict) and (self.roi_dict[site] != [])
        return self.roi_dict[site]






