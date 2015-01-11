__author__ = 'greg'
import mongo
import sys
import os
import bisect
import csv
import matplotlib.pyplot as plt
import urllib
import matplotlib.cbook as cbook

#add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/classifier")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
    sys.path.append("/home/greg/github/reduction/experimental/classifier")

from divisiveKmeans import DivisiveKmeans

#for Greg - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"
sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc


def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


class CondorMongo(mongo.Mongo):
    def __init__(self):
        mongo.Mongo.__init__("condor","2014-11-23")

    def __readin_users(self):
        for user_record in self.user_collection.find():
            if "name" in user_record:
                user = user_record["name"]

                bisect.insort(self.all_users,user)

    def __readin_subject__(self, zooniverse_id):
        #records relating to the individual annotations
        #first - the actual XY markings, then what species are associated with the annotations,
        #then who made each marking
        self.markings_list[zooniverse_id] = []
        self.user_list[zooniverse_id] = []
        self.what_list[zooniverse_id] = []

        #keep track of which users annotated this. Users per subject will contain all users - ip_addresses
        #will contain just those users who were not logged in so we only have the ip address to identify them
        #we need to deal with non-logged in users slightly differently
        self.ips_per_subject[zooniverse_id] = []
        self.users_per_subject[zooniverse_id] = []

        for user_index, classification in enumerate(self.classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
            #get the name of this user
            if "user_name" in classification:
                user = classification["user_name"]
            else:
                user = classification["user_ip"]
                self.ips_per_subject[zooniverse_id].append(user)

            #check to see if we have already encountered this subject/user pairing
            #due to some double classification errors
            if user in self.users_per_subject[zooniverse_id]:
                continue
            self.users_per_subject[zooniverse_id].append(user)

            #read in all of the markings this user made - which might be none
            try:
                mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks", ])
                markings = classification["annotations"][mark_index].values()[0]

                for animal in markings.values():
                    scale = 1.875
                    x = scale*float(animal["x"])
                    y = scale*float(animal["y"])

                    animal_type = animal["animal"]
                    if not(animal_type in ["carcassOrScale", "carcass", "other", ""]):
                        self.markings_list[zooniverse_id].append((x,y))
                        #print annotation_list
                        self.user_list[zooniverse_id].append(user)
                        self.what_list[zooniverse_id].append(animal_type)
            except (ValueError, KeyError):
                pass

    def __cluster_subject__(self,zooniverse_id,clustering_alg,correction_alg=None):
        assert zooniverse_id in self.markings_list

        if self.markings_list[zooniverse_id] != []:
            #cluster results will be a 3-tuple containing a list of the cluster centers, a list of the points in each
            #cluster and a list of the users who marked each point
            self.clusterResults[zooniverse_id] = clustering_alg(self.markings_list[zooniverse_id],self.user_list[zooniverse_id])

            #make sure we got a 3 tuple and since there was a least one marking, we should have at least one cluster
            #pruning will come later
            assert len(self.clusterResults[zooniverse_id]) == 3
            assert self.clusterResults[zooniverse_id][0] != []

            #fix the cluster if desired
            if correction_alg is not None:
                self.clusterResults[zooniverse_id] = correction_alg(self.clusterResults[zooniverse_id])

    def __signal_ibcc__(self):
        #run ibcc on each cluster to determine if it is a signal (an actual animal) or just noise
        #run ibcc on all of the subjects that have been processed (read in and clustered) so far
        #each cluster needs to have a universal index
        cluster_count = -1

        #need to give the ip addresses unique indices, so update ip_index after every subject
        ip_index = 0

        #needed for determining priors for IBCC
        real_animals = 0
        fake_animals = 0
        true_pos = 0
        false_neg = 0
        false_pos = 0
        true_neg = 0

        #intermediate holder variable
        #because ibcc needs indices to be nice and ordered with no gaps, we have to make two passes through the data
        to_ibcc = []

        for zooniverse_id in self.clusterResults:
            for cluster_center,cluster_markings,user_per_cluster in self.clusterResults[zooniverse_id]:
                #moving on to the next animal so increase counter
                #universal counter over all images
                cluster_count += 1

                #needed for determining priors for IBCC
                pos = 0
                neg = 0

                #check whether or not each user marked this cluster
                for u in self.users_per_subject[zooniverse_id]:
                    #was this user logged in or not?
                    #if not, their user name will be an ip address
                    try:
                        i = self.ips_per_subject[zooniverse_id].index(u) + ip_index


                        if u in user_per_cluster:
                            to_ibcc.append((-i,cluster_count,1))
                            pos += 1
                        else:
                            to_ibcc.append((-i,cluster_count,0))
                            neg += 1
                    #if a ValueError was thrown, the user name was not in the list of ip addresses
                    #and therefore, the user name was not an ip address, which means the user was logged in
                    except ValueError as e:

                        if u in user_per_cluster:
                            to_ibcc.append((index(self.all_users,u),cluster_count,1))
                            pos += 1
                        else:
                            to_ibcc.append((index(self.all_users,u),cluster_count,0))
                            neg += 1

                if pos > neg:
                    real_animals += 1

                    true_pos += pos/float(pos+neg)
                    false_neg += neg/float(pos+neg)
                else:
                    fake_animals += 1

                    false_pos += pos/float(pos+neg)
                    true_neg += neg/float(pos+neg)

            ip_index += len(self.ips_per_subject[zooniverse_id])

        #now run through again - this will make sure that all of the indices are ordered with no gaps
        ibcc_user_list = []
        for u,animal_index,found in to_ibcc:
            #can't use bisect or the indices will be out of order
            if not(u in ibcc_user_list):
                ibcc_user_list.append(u)

        #write out the input file for IBCC
        with open(base_directory+"/Databases/condor_ibcc.csv","wb") as f:
            f.write("a,b,c\n")
            for u,animal_index,found in to_ibcc:
                i = ibcc_user_list.index(u)
                f.write(str(i)+","+str(animal_index)+","+str(found)+"\n")

        #create the prior estimate and the default confusion matrix
        prior = real_animals/float(real_animals + fake_animals)
        confusion = [[max(int(true_neg),1),max(int(false_pos),1)],[max(int(false_neg),1),max(int(true_pos),1)]]

        #create the config file
        with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
            f.write("import numpy as np\n")
            f.write("scores = np.array([0,1])\n")
            f.write("nScores = len(scores)\n")
            f.write("nClasses = 2\n")
            f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
            f.write("outputFile = \""+base_directory+"/Databases/condor_signal.out\"\n")
            f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
            f.write("nu0 = np.array(["+str(max(int((1-prior)*100),1))+","+str(max(int(prior*100),1))+"])\n")
            f.write("alpha0 = np.array("+str(confusion)+")\n")

        #start by removing all temp files
        try:
            os.remove(base_directory+"/Databases/condor_signal.out")
        except OSError:
            pass

        try:
            os.remove(base_directory+"/Databases/condor_ibcc.mat")
        except OSError:
            pass

        try:
            os.remove(base_directory+"/Databases/condor_ibcc.csv.dat")
        except OSError:
            pass

        #pickle.dump((big_subjectList,big_userList),open(base_directory+"/Databases/tempOut.pickle","wb"))
        ibcc.runIbcc(base_directory+"/Databases/condor_ibcc.py")

    def __process_signal__(self):
        self.signal_probability = []

        with open(base_directory+"/Databases/condor_signal.out","rb") as f:
            results = csv.reader(f, delimiter=' ')

            for row in results:
                self.signal_probability.append(float(row[2]))

    def __display_signal_noise(self):
        for ii,zooniverse_id in enumerate(self.clusterResults):
            print zooniverse_id
            subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})
            zooniverse_id = subject["zooniverse_id"]
            url = subject["location"]["standard"]

            slash_index = url.rfind("/")
            object_id = url[slash_index+1:]

            if not(os.path.isfile(base_directory+"/Databases/condors/images/"+object_id)):
                urllib.urlretrieve (url, base_directory+"/Databases/condors/images/"+object_id)

            image_file = cbook.get_sample_data(base_directory+"/Databases/condors/images/"+object_id)
            image = plt.imread(image_file)

            fig, ax = plt.subplots()
            im = ax.imshow(image)

            for center,animal_index,users_l,user_count in results_dict[zooniverse_id]:
                if ibcc_v[animal_index] >= 0.5:
                    print center[0],center[1],1
                    plt.plot([center[0],],[center[1],],'o',color="green")
                else:
                    print center[0],center[1],0
                    plt.plot([center[0],],[center[1],],'o',color="red")

            #print "==--"
            plt.show()
            plt.close()