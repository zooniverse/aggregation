__author__ = 'greg'
import aggregation


class PenguinIteration(aggregation.AnnotationIteration):
    def __init__(self, classification,x_max,y_max):
        aggregation.AnnotationIteration.__init__(self, x_max, y_max)

        key_mappings = [t["key"] if "key" in t else None for t in classification["annotations"]]

        if "marking" in key_mappings:
            mark_index = key_mappings.index("marking")

            try:
                self.markings = classification["annotations"][mark_index]["value"].values()
                self.numMarkings = len(self.markings)
            except AttributeError:
                pass


class PenguinAggregation(aggregation.Aggregation):
    def __init__(self, to_skip=[]):
        aggregation.Aggregation.__init__(self, "penguin", "2014-10-22", ann_iterate=PenguinIteration, to_skip=to_skip)

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
