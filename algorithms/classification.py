__author__ = 'greg'
import clustering
import ouroboros_api


class Classification:
    def __init__(self,project,clustering_alg=None):
        assert isinstance(project,ouroboros_api.OuroborosAPI)
        self.project = project

        if clustering_alg is not None:
            assert isinstance(clustering_alg,clustering.Cluster)
        self.cluster_alg = clustering_alg

    def __classify__(self,subject_ids):
        pass

class MajorityVote(Classification):
    def __init__(self,project,clustering_alg=None):
        Classification.__init__(self,project,clustering_alg)

    def __classify__(self,subject_ids):
        self.results = {}
        for subject_id in subject_ids:
            self.results[subject_id] = []
            for poll in self.project.__get_classifications__(subject_id,self.cluster_alg):
                vote_counts = {}
                for user,vote in poll:
                    if vote in vote_counts:
                        vote_counts[vote] += 1
                    else:
                        vote_counts[vote] = 1

                self.results[subject_id].append(max(vote_counts,key=lambda x:vote_counts[x]))

        return self.results


class AllOrNothing(Classification):
    def __init__(self,project,clustering_alg=None):
        Classification.__init__(self,project,clustering_alg)

    def createConfigFile(self,classID):
        f = open(baseDir+"ibcc/"+str(classID)+"config.py",'wb')
        # print("import numpy as np\nscores = np.array([0,1])", file=f)
        # print("nScores = len(scores)", file=f)
        # print("nClasses = 2",file=f)
        # print("inputFile = '"+baseDir+"ibcc/"+str(classID)+".in'", file=f)
        # print("outputFile =  '"+baseDir+"ibcc/"+str(classID)+".out'", file=f)
        # print("confMatFile = '"+baseDir+"ibcc/"+str(classID)+".mat'", file=f)
        # print("nu0 = np.array([45.0,55.0])", file=f)

    def __fit__(self,subject_ids,gold_standard=False):
        self.results = {}
        classification_counter = -1
        candidates = []
        users = []
        agreement = 0
        nonagreement = 0
        notenough = 0
        for subject_id in subject_ids:
            # print "-----"
            # print self.project.gold_annotations[subject_id]
            self.results[subject_id] = []
            for poll_index,poll in enumerate(self.project.__get_classifications__(subject_id,self.cluster_alg,gold_standard)):
                vote_counts = {}
                if len(poll) > 2:
                    classification_counter  += 1
                    for user,vote in poll:
                        if not(vote in vote_counts):
                            vote_counts[vote] = 1
                        else:
                            vote_counts[vote] += 1
                    #     if not(vote in candidates):
                    #         candidates.append(vote)
                    #     if not(user in users):
                    #         users.append(user)
                    #     print users.index(user),classification_counter,candidates.index(vote)
                    # print
                    if len(vote_counts) ==1:
                        agreement +=1
                    else:
                        print poll
                        nonagreement += 1
                else:
                    notenough +=1

                # self.results[subject_id].append(max(vote_counts,key=lambda x:vote_counts[x]))
        print agreement,nonagreement,notenough
        return self.results