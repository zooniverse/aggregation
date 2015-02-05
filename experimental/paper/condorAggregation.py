__author__ = 'greg'
import aggregation
import cPickle as pickle
import os.path


class CondorTools(aggregation.ClassificationTools):
    def __init__(self):
        aggregation.ClassificationTools.__init__(self,scale=1.875)

    def __classification_to_markings__(self,classification):
        annotations = classification["annotations"]

        # in which case, the user should not have made any markings
        try:
            marking_location = [ann.keys() for ann in annotations].index(["marks"])
        except ValueError:
            return []

        marks_list = annotations[marking_location].values()[0].values()
        return marks_list


class CondorAggregation(aggregation.Aggregation):
    def __init__(self, to_skip=[]):
        #["carcassOrScale", "carcass", "other", ""]
        aggregation.Aggregation.__init__(self, "condor", "2015-01-22",tools=CondorTools(), to_skip=to_skip)


    def __get_gold_subjects__(self):
        subjects = []
        for classification in self.classification_collection.find({"user_name":"wreness"}):
            zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            state = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})["state"]
            if state == "complete":
                subjects.append(zooniverse_id)

        return subjects

    def __load_gold_standard__(self,zooniverse_id):
        # have we already encountered this subject?
        if os.path.isfile("/Users/greg/Databases/condor/"+zooniverse_id+"_gold.pickle"):
            self.gold_data[zooniverse_id] = pickle.load(open("/Users/greg/Databases/condor/"+zooniverse_id+"_gold.pickle","rb"))
        else:
            annotations = self.classification_collection.find_one({"subjects.zooniverse_id":zooniverse_id,"user_name":"wreness"})["annotations"]

            self.gold_data[zooniverse_id] = []

            #were there any markings?
            for ann in annotations:
                if "marks" in ann:
                    for marks in ann["marks"].values():
                        marks["x"] = 1.875*float(marks["x"])
                        marks["y"] = 1.875*float(marks["y"])
                        self.gold_data[zooniverse_id].append(marks)

            pickle.dump(self.gold_data[zooniverse_id],open("/Users/greg/Databases/condor/"+zooniverse_id+"_gold.pickle","wb"))

    def __readin_subject__(self,zooniverse_id):
        aggregation.Aggregation.__readin_subject__(self,zooniverse_id,users_to_skip=["wreness"])

    def __load_roi__(self,zooniverse_id):
        # the actual markings might be scaled down but since, for condor watch, we want to include every point
        # this should be fine

        return (1920,1080)

