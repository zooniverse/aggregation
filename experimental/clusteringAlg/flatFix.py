__author__ = 'greg'
from fix import Fix


class FlatFix(Fix):
    def __init__(self):
        pass

    def calc_relations(self,centers,users_per_cluster,threshold):
        relations = []
        for c1_index in range(len(centers)):
            for c2_index in range(c1_index+1,len(centers)):
                c1 = centers[c1_index]
                c2 = centers[c2_index]

                u1 = users_per_cluster[c1_index]
                u2 = users_per_cluster[c2_index]

                dist = math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)

                overlap = [u for u in u1 if u in u2]
                #print (len(overlap),dist)

                #print (len(overlap),dist)
                if (len(overlap) <= 1) and (dist <= threshold):
                    relations.append((dist,overlap,c1_index,c2_index))

        relations.sort(key= lambda x:x[0])
        relations.sort(key= lambda x:len(x[1]))

        return relations
