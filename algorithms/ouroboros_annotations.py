"""
Python code for accessing the classifications/markings/annotations for different Zooniverse projects hosted on
Ouroboros. The raw data for Ouroboros is stored in MongoDB so that is what this code uses. Technically we could also
use the csv files that are output (and that may be something to do for future work) but working directly with the
MongoDB database seems to be the best way. For Panoptes, the hope is that this one file will be replaced and
everything else can work just fine.

"""


class MarkingAnnotations():
    """
    For projects where the user marks regions or points of interest -
    """
    def __init__(self,scale=1):
        self.scale = scale

    def __classification_to_markings__(self,classification):
        assert False

class ROIClassificationTools(ClassificationTools):
    """
    For projects where users are asked to mark images - restricted to regions of interest (ROIs)
    The best example is Penguin Watch. Other projects can be viewed as having ROIs by setting the ROIs to be the
    whole image.
    While in theory no points should even exist in the first place that are outside of the ROI, for Penguin Watch
    some such points have been found. So this class just ignores any such points.
    """
    def __init__(self,scale=1):
        # roi is the region of interest - if a point lies outside of this region, we will ignore it
        # such cases should probably all be errors - a marking lies outside of the image
        # or, somehow, just outside of the ROI - see penguin watch for an example
        # because Penguin Watch started it, and the images for penguin watch are flipped (grrrr)
        # the roi is the region ABOVE the line segments
        ClassificationTools.__init__(self,scale)

        self.roi_dict = {}

    def __load_roi__(self,classification):
        assert False

    def __list_markings__(self,classification):
        marks_list = self.__classification_to_markings__(classification)
        roi = self.__load_roi__(classification)

        for mark in marks_list:
            x = float(mark["x"])*self.scale
            y = float(mark["y"])*self.scale

            if not("animal" in mark):
                animal_type = None
            else:
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
                        # thus, the marking will count towards not being noise but will not be used when determining the type

                        yield (x,y),animal_type
                        break
                    else:
                        break