import classification
import ibcc
import os

class IBCC(classification.Classification):
    def __init__(self,clustering_alg=None):
        classification.Classification.__init__(self,clustering_alg)

    def __task_aggregation__(self,raw_classifications,gold_standard_results=None):
        # do we actually need to run ibcc - no if there wasn't any confusion
        # borderline degenerate case but we need to be prepared for it
        # highest_class is needed for helping the degenerate cases
        run_ibcc,highest_class =self.__ibcc_setup__(raw_classifications)

        # run ibcc
        if run_ibcc:
            ibcc.load_and_run_ibcc("/tmp/config.py")

            # now analyze the results
            print "not degenerate"
            return self.__ibcc_analyze__(raw_classifications)
        else:
            print "degenerate case"
            return self.__degenerate_ibcc__(raw_classifications,highest_class)

    def __degenerate_ibcc__(self,raw_classifications,highest_classes):
        """
        handle cases which are borderline degenerate - i.e. everyone agrees on everything
        :param raw_classifications:
        :return:
        """
        results = {}
        for subject_id in raw_classifications:
            if subject_id == "param":
                    continue

            num_votes = len(raw_classifications[subject_id])
            users,votes = zip(*raw_classifications[subject_id])
            assert min(votes) == max(votes)
            probabilities = [0 for i in range(highest_classes+1)]
            probabilities[min(votes)] = 1
            results[subject_id] = probabilities,num_votes

        return results

    def __ibcc_analyze__(self,raw_classifications):
        global_cluster_index = 0
        # as long as raw classification are not changed, the ordering of the keys will not change
        # even if that ordering is completely weird

        results = {}

        with open("/tmp/ibcc_output.csv","rb") as csvfile:
            for subject_id in raw_classifications:
                if subject_id == "param":
                    continue

                s = csvfile.readline()
                assert s != ""

                # extract the probabilities of each classification
                # and convert into floats
                probabilities = s.split(" ")[1:]
                probabilities = [float(p) for p in probabilities]

                # this is the number of people who have voted on this classification
                num_votes = len(raw_classifications[subject_id])
                results[subject_id] = probabilities,num_votes

        return results

    def __ibcc_setup__(self,raw_classifications):
        """
        if raw_classifications correspond to a simple task then each each should just be a subject_id
        if raw_classifications corresponds to marking then each key should be a combination of
        subject id and cluster index
        :param raw_classifications:
        :return:
        """
        global_cluster_index = -1
        global_user_list = []

        # for right now use plurality voting to estimate the priors and confusion matrix
        # use dictionary so we are not pre restricting how many classes there are - we can figure this
        # out as we go
        prior_estimates = {}
        # make this be a dictionary of dictionaries - gives us some flexibility
        confusion_estimate = {}

        # use this in case there are gaps in the class labels
        highest_class = 0

        with open("/tmp/ibcc_input.csv",'wb') as f:
            f.write("a,b,c\n")
            for subject_id in raw_classifications:#sorted(raw_classifications.keys()):
                if subject_id == "param":
                    continue
                votes = {}
                global_cluster_index += 1
                for user,ballot in raw_classifications[subject_id]:
                    # todo - may need to catch cases where users are able to select multiple options
                    assert isinstance(ballot,int)
                    if user not in global_user_list:
                        global_user_list.append(user)

                    # write out the ibcc input file
                    f.write(str(global_user_list.index(user))+","+str(global_cluster_index)+","+str(ballot)+"\n")


                    # update the prior estimate
                    if ballot not in votes:
                        votes[ballot] = 0
                    votes[ballot] += 1

                highest_class = max(highest_class,max(votes.keys()))
                # what was the most likely classification according to plurality voting
                most_likely = max(votes.items(),key = lambda x:x[1])
                # most_likely will also contain the vote count which we don't really care about
                most_likely_classification = most_likely[0]

                if most_likely_classification not in prior_estimates:
                    prior_estimates[most_likely_classification] = 0
                prior_estimates[most_likely_classification] += 1

                # what do we estimate the confusion matrix to be with respect to this subject?
                vote_percentages = {candidate:v/float(sum(votes.values())) for candidate,v in votes.items()}

                if most_likely_classification not in confusion_estimate:
                    confusion_estimate[most_likely_classification] = {}

                for candidate,per in vote_percentages.items():
                    if candidate not in confusion_estimate[most_likely_classification]:
                        confusion_estimate[most_likely_classification][candidate] = 0

                    confusion_estimate[most_likely_classification][candidate] += per

        # now convert to an actual confusion matrix - there may be values missing
        # either because a particular class was never seen or a particular confusion never
        # took place - relatively rare in either case but we need to be careful
        # see step 8 of configuration on page
        # https://github.com/CitizenScienceInAstronomyWorkshop/pyIBCC/wiki
        # for details - obviously this approach might be not optimal for each project
        # but as first start, should be good
        weight = (highest_class+1) * 10
        confusion_matrix = []

        # if there hs been no confusion at all - we don't need to bother running IBCC
        if max([len(row) for row in confusion_estimate.values()]) == 1:
            return False,highest_class

        for true_class in range(highest_class+1):
            if true_class in confusion_estimate:
                s = float(sum(confusion_estimate[true_class].values()))
                row = []
                for reported_class in range(highest_class+1):
                    if reported_class in confusion_estimate[true_class]:
                        # the count has to be at least 1
                        row.append(int(max(1,confusion_estimate[true_class][reported_class]*weight/s)))
                        # row.append(int(confusion_estimate[true_class][reported_class]*weight/s))
                    else:
                        row.append(1)
                confusion_matrix.append(row[:])
            else:
                confusion_matrix.append([1 for i in range(highest_class+1)])

        # extract the estimate count
        # if there are now counts, give a value of 1
        prior_counts = [prior_estimates[i] if i in prior_estimates else 1 for i in range(highest_class+1)]
        # prior_counts = zip(*sorted(prior_estimates.items(), key = lambda x:x[0]))[1]
        # scale the counts so that they add up to 100-ish
        prior_counts = [int(max(1,c*100/sum(prior_counts))) for c in prior_counts]

        # now create the config file
        self.__create_config__(prior_counts,confusion_matrix)

        return True,highest_class

    def __create_config__(self,priors,confusion_matrix):
        """
        write out the config file for running IBCC
        :return:
        """
        try:
            os.remove("/tmp/ibcc_input.csv.dat")
        except OSError:
            pass

        num_classes = len(priors)
        assert len(priors) == len(confusion_matrix)

        with open("/tmp/config.py",'wb') as f:
            f.write("import numpy as np\n")
            f.write("scores = np.array("+str(range(num_classes))+")\n")
            f.write("nScores = len(scores)\n")
            f.write("nClasses = "+str(num_classes)+"\n")
            f.write("inputFile = \"/tmp/ibcc_input.csv\"\n")
            f.write("outputFile = \"/tmp/ibcc_output.csv\"\n")
            f.write("confMatFile = \"/tmp/ibcc.mat\"\n")
            f.write("nu0 = np.array("+str(priors)+")\n")
            f.write("alpha0 = np.array("+str(confusion_matrix)+")\n")

    # def __classify__(self,subject_ids,gold_standard=False):
    #     self.results = {}
    #     # might be over doing the elections analogy but can't think of a better way to describe things
    #     # ridings is a list of tuples (subject_ids, cluster_center) so we can match up the results from IBCC
    #     # if no clustering was involved (so only one classification per subject_id) then cluster_center should
    #     # be None
    #     ridings = []
    #     # ridings_dict stores the "ridings" by subject id - that way, we don't need to search through all
    #     # of the ridings, everytime we want to find the "elections" for a given subject_id
    #     ridings_dict = {}
    #     # candidates = []
    #     users = []
    #     agreement = 0
    #     nonagreement = 0
    #     notenough = 0
    #     # all_elections = {}
    #     # self.create_configfile(len(self.species))
    #     nclasses = len(self.species)
    #     nu0 = [100/nclasses for i in range(nclasses)]
    #     confusion_matrix = [[0.2 for i in range(nclasses)] for j in range(nclasses)]
    #
    #
    #
    #     # classifer = ibcc.IBCC(nclasses=nclasses,nscores=nclasses,alpha0=confusion_matrix,nu0=nu0)
    #
    #     priors = {s:1 for s in self.candidates}
    #     # confusion = [[1 for i in self.candidates] for j in self.candidates]
    #
    #     # for i in range(nclasses):
    #     #     confusion[i][i] = 20
    #
    #     with open(self.base_directory+"Databases/plankton_ibcc.csv",'wb') as f:
    #         f.write("a,b,c\n")
    #         for subject_id in subject_ids:
    #             # print "-----"
    #             # print self.project.gold_annotations[subject_id]
    #             self.results[subject_id] = []
    #
    #             # cluster centers only make sense if we have a clustering setup - otherwise they should just be empty
    #             cluster_centers,polls = self.project.__get_classifications__(subject_id,cluster_alg=self.cluster_alg,gold_standard=gold_standard)
    #
    #             for poll_index,(center,poll) in enumerate(zip(cluster_centers,polls)):
    #                 print center
    #                 print poll
    #                 print
    #                 # local_candidates = set()
    #                 vote_counts = {}
    #                 if len(poll) >=4:
    #                     # classification_counter  += 1
    #                     ridings.append((subject_id,center))
    #                     if not(subject_id in ridings_dict):
    #                         ridings_dict[subject_id] = [center]
    #                     else:
    #                         ridings_dict[subject_id].append(center)
    #
    #                     for user,vote,pt in poll:
    #                         # assert isinstance(vote,unicode)
    #                         # local_candidates.add(vote)
    #
    #                         # use majority voting to establish priors
    #                         if not(vote in vote_counts):
    #                             vote_counts[vote] = 1
    #                         else:
    #                             vote_counts[vote] += 1
    #                         # if not(vote in candidates):
    #                         #     candidates.append(vote)
    #                         if not(user in users):
    #                             users.append(user)
    #                         # print vote,self.species[vote.lower()],pt
    #                         f.write(str(users.index(user))+","+str(len(ridings)-1)+","+str(self.candidates.index(vote.lower()))+"\n")
    #                         # print users.index(user),classification_counter,self.candidates.index(vote)
    #
    #                     most_votes = max(vote_counts,key=lambda x:vote_counts[x])
    #                     priors[most_votes.lower()] += 1
    #
    #                     # now that we know what the majority vote estimate is, estimate the confusion matrix
    #                     most_votes_index = self.candidates.index(most_votes.lower())
    #                     for user,vote,pt in poll:
    #                         confusion_matrix[most_votes_index][self.candidates.index(vote.lower())] += 1/float(len(poll))
    #
    #                     if len(vote_counts) ==1:
    #                         agreement +=1
    #                     else:
    #                         nonagreement += 1
    #                     # print local_candidates
    #                     # local_candidates = tuple(sorted(list(local_candidates)))
    #                     # if not(local_candidates in all_elections):
    #                     #     all_elections[local_candidates] = 1
    #                     # else:
    #                     #     all_elections[local_candidates] += 1
    #                 else:
    #                     notenough +=1
    #
    #     # confusion_matrix = []
    #     print "^^^^^"
    #     for i,row in enumerate(confusion_matrix):
    #         # print c
    #         confusion_matrix[i] = [int(a/min(row)) for a in row]
    #
    #         # print
    #     print
    #     print sum(priors.values())
    #     self.create_configfile(priors,confusion_matrix)
    #
    #     # ibcc.runIbcc(self.base_directory+"Databases/config.py")
    #     ibcc.load_and_run_ibcc(self.base_directory+"Databases/config.py")
    #     results = {}
    #     with open(self.base_directory+"Databases/plankton_ibcc.out","rb") as f:
    #         for i,l in enumerate(f.readlines()):
    #             # print "===-----"
    #             subject_id,center = ridings[i]
    #
    #             if not(subject_id in results):
    #                 results[subject_id] = []
    #
    #             # print elections[i]
    #             probabilities = [float(p) for j,p in enumerate(l.split(" ")[1:])]
    #             results[subject_id].append(probabilities)
    #             # print probabilities
    #             # ibcc_most_likely = max(probabilities, key= lambda x:x[1])
    #             # print ibcc_most_likely
    #             # print self.candidates[ibcc_most_likely[0]]
    #             # self.results[subject_id].append(max(vote_counts,key=lambda x:vote_counts[x]))
    #     # print all_elections
    #     # G=nx.Graph()
    #     # species_keys = self.species.keys()
    #     # G.add_nodes_from(range(len(species_keys)))
    #     # for e in all_elections.keys():
    #     #     for a,b in findsubsets(e,2):
    #     #         G.add_edge(species_keys.index(a.lower()),species_keys.index(b.lower()))
    #     #
    #     # nx.draw(G)
    #     # plt.show()
    #     # print agreement,nonagreement,notenough
    #     return self.candidates,ridings_dict,results