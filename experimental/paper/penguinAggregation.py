__author__ = 'greg'
import aggregation
import csv

class PenguinTools(aggregation.ROIClassificationTools):
    def __init__(self,subject_collection):
        aggregation.ROIClassificationTools.__init__(self,scale=1) #1.92

        self.subject_collection = subject_collection

        # load the roi for the different sites
        with open(aggregation.code_directory+"/Penguins/public/roi.tsv","rb") as roiFile:
            roiFile.readline()
            reader = csv.reader(roiFile,delimiter="\t")
            for l in reader:
                path = l[0]
                t = [r.split(",") for r in l[1:] if r != ""]
                self.roi_dict[path] = [(int(x)/1.92,int(y)/1.92) for (x,y) in t]

    def __classification_to_markings__(self,classification):
        annotations = classification["annotations"]
        values_list = [ann.values() for ann in annotations]
        for v in values_list:
            if "marking" in v:
                print "====---"
                try:
                    return v[1].values()
                except AttributeError:
                    # something about this set of markings is wrong
                    return []

        return []

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

class PenguinAggregation(aggregation.Aggregation):
    def __init__(self, to_skip=[]):
        aggregation.Aggregation.__init__(self, "penguin", "2015-01-18", to_skip=to_skip)
        self.tools = PenguinTools(self.subject_collection)

    def __get_subjects_per_site__(self,zooniverse_id):
        subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})

        # the path name will contain a unique ID for the site location
        path = subject["metadata"]["path"]
        slash_index = path.index("/")
        underscore_index = path.index("_")
        # for right now, I assuming that the year must be the same as well - not sure if the cameras would have
        # moved
        site = path[slash_index+1:underscore_index]

        id_list = []

        for subject in self.subject_collection.find({"metadata.path":{"$regex":site}}):
            zooniverse_id = subject["zooniverse_id"]
            id_list.append(zooniverse_id)
            self.dimensions[zooniverse_id] = subject["metadata"]["original_size"]

        return id_list

    def __load_dimensions__(self,zooniverse_id,subject=None):
        if subject is None:
            subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})

        self.dimensions[zooniverse_id] = subject["metadata"]["original_size"]
        return self.dimensions[zooniverse_id]






