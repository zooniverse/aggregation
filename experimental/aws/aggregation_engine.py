#!/usr/bin/env python
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import tarfile
import cStringIO
import StringIO
from sklearn.cluster import DBSCAN
import numpy as np
import json


class AggregationEngine():
    def __init__(self):
        #connect to the zooniverse-data bucket on S3
        conn = S3Connection()
        self.mybucket = conn.get_bucket("zooniverse-data")

    def __aggregate__(self,param_dict):
        """
        for a given json param set, aggregate - may do more than one aggregation - depending on what is stored in
        the json file
        :param json_param:
        :return:
        """
        #load a csv data file from S3
        date = param_dict["date"]
        project = param_dict["project"]
        t = self.mybucket.get_key("/ouroboros/"+date+"_"+project+"_classifications.csv.tar.gz")

        #convert the bucket into a csv reader
        st = t.get_contents_as_string()
        tar = tarfile.open(mode="r:gz",fileobj = cStringIO.StringIO(st))
        csv_annotations = tar.extractfile("2014-09-28_penguin_classifications.csv")

        csv_header = csv_annotations.readline()

        for run_number in range(len(param_dict["algorithms"])):
            zooniverse_id_index, to_extract, param_type = self.__get_extraction__(csv_header,param_dict,run_number)
            annotations = self.__read_annotations__(csv_annotations,zooniverse_id_index,to_extract,param_type)

            #now that we have the annotation data relevant to this specific aggregation algorithm - whatever it is
            #do the aggregation - right now, only support DBSCAN
            print annotations

        assert False

        #header for the end CSV file
        result_string = "\"zooniverse_id\",\"aggregation_run\",\"type\",\"x\",\"y\"\n"


        annotations = {}

        i = 0




        result_bucket = conn.get_bucket("zooniverse-aggregation")
        k = Key(result_bucket)
        k.key = "penguins_test.tar.gz"
        #k.set_contents_from_string(result_string)
        st = cStringIO.StringIO()
        result_tar = tarfile.open(mode="w:gz",fileobj = st)

        data = StringIO.StringIO(result_string)
        info = result_tar.tarinfo()
        info.name = 'penguins.csv'
        info.size = data.len

        #data.seek(0)
        result_tar.addfile(info, data)
        result_tar.close()

        k.set_contents_from_string(st.getvalue())
        st.close()

    def __get_extraction__(self,csv_header,param_dict,run_number):
        """
        return a list of all of the columns to extract from the user csv file - and the associated type of those
        columns
        :param param_dict: the dictionary containing all of the global/local parameters
        :param run_number: which aggregation run we are on
        :return zooniverse_id_index
        :return to_extract
        :return param_type
        """
        #first find out where the zooniverse_ids are stored - these MUST be here
        headers = csv_header[:-1].split(",")
        zooniverse_id_index = headers.index("\"subject_zooniverse_id\"")

        to_extract = []
        param_type = []

        local_params = param_dict["algorithms"][str(run_number)]
        #is this a clustering algorithm - if so, find out which axes we are clustering on
        if "cluster_on" in local_params:
            #find out each of the axes we will be clustering on
            for axis in local_params["cluster_on"]:
                to_extract.append(headers.index("\""+axis+"\""))
                param_type.append(float)


    def __get_type_filter__(self,csv_header,param_dict,run_number):
        """

        :param csv_header:
        :param param_dict:
        :param run_number:
        :return:
        """
        annotation_filters = [lambda x:True,]
        local_params = param_dict["algorithms"][str(run_number)]

        #do we care about types (e.g. do we want to diff. between adult penguins and chicks?
        if "type_groupings" in local_params:
            assert("type_given_by" in param_dict)
            to_extract.append(headers.index("\""+param_dict["type_given_by"]+"\""))
            param_type.append(str)

        return zooniverse_id_index, to_extract, param_type

    def __attribute__(self,csv_header,param_dict,run_number):
        """
        :param csv_header:
        :param param_dict:
        :param run_number:
        :return:
        """
        headers = csv_header[:-1].split(",")

        local_params = param_dict["algorithms"][str(run_number)]
        type_filter = {}
        type_index = None

        if "type_groupings" in local_params:
            type_index = headers.index("\""+param_dict["type_given_by"]+"\"")
            for group_index in range(len(local_params["type_groupings"])):
                for t in local_params["type_groupings"][group_index]:
                    type_filter[t] = group_index

        return type_index,type_filter

    def __read_annotations__(self,csv_annotations,zooniverse_id_index,to_extract,param_type):
        """
        Read through each row of the csv file and extract the required columns - convert each param to the given
        type - if we are unable to do so, the entry is skipped
        :param csv_annotations: - a file containing the user annotations in csv format, assume that the header
         has already been read
        :param zooniverse_id_index - the index of the zooniverse_id column
        :param to_extract: - a list of the columns which we will extract
        :param param_type - the type of each column (probably either float, int or str)
        :return: annotations
        """
        annotations = {}
        i = 0
        for line in csv_annotations.readlines():
            i += 1
            if i == 300:
                break
            words = line[:-1].split(",")

            zooniverse_id = words[zooniverse_id_index][1:-1]

            next_annotation = []
            error = False
            for param_index,t in zip(to_extract,param_type):
                #remove the quotation marks
                value = words[param_index][1:-1]
                try:
                    #try converting the value to necessary type - if that fails don't include this annotation
                    #most often will fail when we try to cast an empty string into a float or int
                    #if so, it just means that this annotation is blank - casting a blank string into a string
                    #should not cause any trouble
                    next_annotation.append(t(value))
                except ValueError:
                    error = True
                    break

            if not error:
                #check to see if we are filtering on type
                if type_index is not None:
                    value = words[type_index][1:-1]
                    if value in filter_dict:

                if not(zooniverse_id in annotations):
                    annotations[zooniverse_id] = [next_annotation[:]]
                else:
                    annotations[zooniverse_id].append(next_annotation[:])

        return annotations

    def dbscan(annotations,param_dict):
        """

        :param annotations: a list of annotations, each of the format (type, x, y)
        :return: penguins - a list of the center of each penguin according to DBSCAN
        """
        epsilon = param_dict["epsilon"]
        min_pts = param_dict["min_pts"]

        for zooniverse_id in annotations:

        #extract the x and y coordinates of all adults and penguins
        #and convert to np.array which DBSCAN needs
        adult_chick_annotations = np.array([(x,y) for t,x,y in annotations if t in ["\"adult\"","\"chick\""]])

        #do the actual DBSCAN - hard coded parameters for simple example only!
        db = DBSCAN(eps=20, min_samples=2).fit(adult_chick_annotations)

        #get the set of unique labels (each corresponds to a penguin or noise)
        labels = db.labels_
        unique_labels = set(labels)

        penguins = []

        #go through cluster
        for k in unique_labels:
            #skip noise = which corresponds to label == -1
            if k != -1:
                #find out which annotations belong to this cluster
                class_member_mask = (labels == k)
                this_cluster = adult_chick_annotations[class_member_mask]

                #get the center of the cluster this is will be the main representation
                #of the penguin
                x_set,y_set = zip(*this_cluster)
                penguins.append(("adult",np.mean(x_set),np.mean(y_set)))

        return penguins


    def __read_annotations2__(self,param_dict,csv_annotations,run_number):
        """
        read through the associated csv file and collect the values needed for individual aggregation run
        :param param_dict: the set of all parameters - global plus ones for each individual aggregation run - stored as dict
               csv_annotations: the csv file containing the user annotations
               run_number: which run we are on - used to identify which set of values to extract
        :return annotations
        """
        aggregation_param = param_dict["algorithms"][str(run_number)]

        #read in the header - used to identify where certain columns are
        header = csv_annotations.readline()
        #chop off the newline
        column_headers = header[:-1].split(",")

        #find out where the zooniverse id is stored - MUST be there somewhere
        zooniverse_id_index = column_headers.index("\"subject_zooniverse_id\"")
        to_extract = [zooniverse_id_index]

        #depending on what aggregation algorithm we are using, where we store the appropriate params to extract
        #may differ
        if aggregation_param["name"] == "dbscan":
            #do we care about the type associated with the annotations? (e.g. with penguins do we
            #care about adult vs. chick)
            if "type_groupings" in aggregation_param:
                #check to see if the "type_given_by" param is defined - since we are grouping by it, it definitely
                #should be defined. This param will either be defined for all runs or none - since it is a function
                #of the csv file, not the individual run - so check the global param
                assert("type_given_by" in param_dict)
                to_extract.append(column_headers.index("\""+param_dict["type_given_by"]+"\""))

            #which parameters will we cluster on? definitely need to extract that
            cluster_on = aggregation_param["cluster_on"]
            to_extract.extend([column_headers.index("\"" + axis + "\"") for axis in cluster_on])

        annotations = []


        i = 0
        #read through the annotations
        for line in csv_annotations.readlines():
            words = line.split(",")
            annotations.extend()
            i += 1
            if i == 1000:
                break
            words = line.split(",")
            zooniverse_id = words[2]

            animals_present = words[12]

            if animals_present == "\"yes\"":
                #extract the necessary elements
                value = words[15]
                try:
                    x = float(words[16][1:-1])
                except ValueError:
                    print line
                    print words[16]
                    print words[16][1:-1]
                    raise

                y = float(words[17][1:-2])

                #if this is the first time we've come across this image, create a new entry
                if not(zooniverse_id in annotations):
                    annotations[zooniverse_id] = [(value,x,y)]
                #otherwise, just append
                else:
                    annotations[zooniverse_id].append((value,x,y))

        for zooniverse_id in annotations:
            penguins = overly_simple_penguin_dbscan(annotations[zooniverse_id])

            for penguin_type,penguin_center_x,penguin_center_y in penguins:
                result_string += zooniverse_id + ",\"0\",\"" + penguin_type + "\",\""+str(penguin_center_x)+"\",\""+str(penguin_center_y)+"\"\n"





def main():
    engine = AggregationEngine()

    json_data = open('/home/greg/Documents/aggregation_param.json')
    param = json.load(json_data)

    engine.__aggregate__(param)

if __name__ == '__main__':
    main()