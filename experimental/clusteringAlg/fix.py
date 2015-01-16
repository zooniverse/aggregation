__author__ = 'greg'
#the base of a series of classes which can fix clusters which have been split but which should not have been
#the difference is how we find "abnormally" close clusters

class Fix:
    def __init__(self):
        pass

    def __fix__(self,centers,clusters,users_per_cluster,threshold,f_name=None):
        #look for abnormally close clusters with only one or zero users in common
        #zero is possible as a result of how k-means splits points
        #users per cluster must in the same order as the points per cluster
        while True:
            #compare every pair of clusters - returns only those clusters with 0 or 1 users in common
            #within the threshold
            relations = self.calc_relations(centers,users_per_cluster,threshold)
            if relations == []:
                break

            #look at the closest pair
            overlap = relations[0][1]
            distance = relations[0][0]
            c1_index = relations[0][2]
            c2_index = relations[0][3]

            #make sure to pop in the right order so else the indices will get messed up
            if c2_index > c1_index:
                cluster_2 = clusters.pop(c2_index)
                cluster_1 = clusters.pop(c1_index)

                cent_2 = centers.pop(c2_index)
                cent_1 = centers.pop(c1_index)

                users_2 = users_per_cluster.pop(c2_index)
                users_1 = users_per_cluster.pop(c1_index)

            else:
                cluster_1 = clusters.pop(c1_index)
                cluster_2 = clusters.pop(c2_index)

                cent_1 = centers.pop(c1_index)
                cent_2 = centers.pop(c2_index)

                users_1 = users_per_cluster.pop(c1_index)
                users_2 = users_per_cluster.pop(c2_index)

            #the distance < threshold check may be redundant - since calc_relations can also check
            #but doesn't hurt
            if (overlap != []) and (distance < threshold):
                assert(len(overlap) == 1)
                overlap_user = overlap[0]
                #the point in each cluster which corresponds to the overlapping user
                p1 = cluster_1.pop(users_1.index(overlap_user))
                #double check that there no other equaivalent points in the array
                #shouldn't happen - unless by really bad luck
                assert sum([1 for p in cluster_1 if (p == p1)]) == 0

                p2 = cluster_2.pop(users_2.index(overlap_user))
                #double check that there no other equaivalent points in the array
                #shouldn't happen - unless by really bad luck
                assert sum([1 for p in cluster_1 if (p == p2)]) == 0

                #since we have removed the points, we now remove the users themselves
                users_1.remove(overlap_user)
                users_2.remove(overlap_user)

                #now merge
                #for now, only deal with 2D points - which should always be the case
                assert len(p1) == 2
                avg_pt = ((p1[0]+p2[0])/2., (p1[1]+p2[1])/2.)
                joint_cluster = cluster_1[:]
                joint_cluster.extend(cluster_2)
                joint_cluster.append(avg_pt)

                joint_users = users_1[:]
                joint_users.extend(users_2)
                joint_users.append(overlap_user)

                #recalculate the center
                X,Y = zip(*joint_cluster)
                center = (np.mean(X),np.mean(Y))

                #add this new merged cluster back
                centers.append(center)
                clusters.append(joint_cluster)
                users_per_cluster.append(joint_users)

            else:
                assert overlap == []
                #we have no nearby clusters with no common users - should be easy to merge

                joint_cluster = cluster_1[:]
                joint_cluster.extend(cluster_2)

                joint_users = users_1[:]
                joint_users.extend(users_2)

                #recalculate the center
                X,Y = zip(*joint_cluster)
                center = (np.mean(X),np.mean(Y))

                #add this new merged cluster back
                centers.append(center)
                clusters.append(joint_cluster)
                users_per_cluster.append(joint_users)

        return centers,clusters,users_per_cluster