__author__ = 'ggdhines'
from penguin import Penguins,SubjectGenerator
from cassandra.concurrent import execute_concurrent
import urllib
import json
from aggregation_api import base_directory
import os
import matplotlib.pyplot as plt

class Analysis(Penguins):
    def __init__(self):
        Penguins.__init__(self)

    def __image_setup__(self,subject_id,download=True):
        """
        get the local file name for a given subject id and downloads that image if necessary
        :param subject_id:
        :return:
        """
        subject = self.subject_collection.find_one({"zooniverse_id":subject_id})
        url = subject["location"]["standard"]

        # print data["subjects"]
        # assert False

        image_path = base_directory+"/Databases/images/"+subject_id+".jpg"

        if not(os.path.isfile(image_path)):
            if download:
                print "downloading"
                urllib.urlretrieve(url, image_path)

        return image_path


    def __analyze__(self):
        statements_and_params = []
        select_statement = self.cassandra_session.prepare("select * from penguins where zooniverse_id = ?")

        for subject_set in SubjectGenerator(self):
            for zooniverse_id in subject_set:
                statements_and_params.append((select_statement, [zooniverse_id]))

                if len(statements_and_params) == 50:
                    results = execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=False)
                    statements_and_params = []

                    for zooniverse_id2,(success,record) in zip(subject_set,results):
                        if record != []:
                            # self.__image_setup__(zooniverse_id)
                            self.__plot_image__(zooniverse_id2)

                            aggregation = json.loads(record[0].aggregations)

                            for pt_index,pt in aggregation["1"]["point"].items():
                                if pt_index in ["param","all_users"]:
                                    continue
                                x,y= pt["center"]
                                plt.plot([x],[y],'.',color="blue")

                            plt.show()

            break

project = Analysis()
project.__analyze__()