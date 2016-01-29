__author__ = 'ggdhines'
from transcription import Tate, TextCluster
import numpy as np
import heapq

class HeuristicTextCluster(TextCluster):
    def __init__(self,shape,param_dict):
        TextCluster.__init__(self,shape,param_dict)

    def __line_alignment__(self,lines):
        print lines
        print [len(l) for l in lines]

        self.M_matrices = []

        Sigma = set()
        for l in lines:
            Sigma.update(l)
        Sigma = list(Sigma)

        # M_T is used to find the a-successor of p
        # created and described in the first paragraph on page 1290
        M_T = np.zeros((len(Sigma),max([len(l) for l in lines]),len(lines)))
        M_T.fill(1+max([len(l) for l in lines]))

        self.parent = {}

        for i,l in enumerate(lines):
            for index in range(len(l)):
                a = l[index]
                for j in range(0,index):
                    M_T[Sigma.index(a),j,i] = min(M_T[Sigma.index(a),j,i],index)

        for l_index in range(len(lines)-1):
            s_i = lines[l_index]
            s_j = lines[l_index+1]

            n_i = len(s_i)
            n_j = len(s_j)

            M = np.zeros((n_i,n_j))

            for x in range(n_i-2,-1,-1):
                for y in range(n_j-2,-1,-1):
                    if s_i[x+1] == s_j[y+1]:
                        M[(x,y)] = M[(x+1,y+1)]+1
                    else:
                        M[(x,y)] = max(M[(x+1,y)],M[(x,y+1)])

            self.M_matrices.append(M)

        p = [0 for i in lines]

        self.parent[tuple(p)] = None

        Q = []

        h_p = self.h(p)

        Q.append(((h_p,h_p),p))
        # line 7 of Algorithm MLCS-A* page 1289
        while Q != []:
            (f_p,h_p),p = heapq.heappop(Q)

            # line 9
            if h_p == 0:
                break

            # to determine if q in Q (line 12) we need to discard the (f,h)'s
            if Q == []:
                p_list = []
            else:
                _,p_list = zip(*Q)

            for q in self.Suc(p,Sigma,M_T):
                print q
                # line 12
                if q not in p_list:
                    (f_q,h_q) = self.UpdateSuc(p,q,f_p,h_p)
                    Q.append((((f_q,h_q)),q))
                else:
                    q_index = p_list.index(q)
                    (f_q,h_q),_ = Q[q_index]
                    g_q = f_q - h_q
                    g_p = f_p - h_p

                    # line 15
                    if g_q < (g_p+1):
                        (f_q,h_q) = self.UpdateSuc(p,q,f_p,h_p)
                        Q[q_index] = (f_q,h_q),q

            heapq.heapify(Q)
            print Q
            print




        assert False

    def h(self,p):
        h = float("inf")
        for l_index in range(len(self.M_matrices)):
            h = min(h,self.M_matrices[l_index][(p[l_index],p[l_index+1])])

        return h

    def Suc(self,p,Sigma,M_T):
        for a in Sigma:
            q = [int(M_T[(Sigma.index(a),p[i],i)]) for i in range(len(p))]
            yield q

        raise StopIteration()

    def UpdateSuc(self,p,q,f_p,h_p):
        # from the definition just under Corollary 2
        g_p = f_p-h_p

        # line 19
        g_q = g_p + 1
        h_q = self.h(q)
        f_q = g_q + h_q

        self.parent[tuple(q)] = p

        return f_q,h_q

class HeuristicTranscription(Tate):
    def __init__(self,project_id,environment):
        Tate.__init__(self,project_id,environment)

        self.default_clustering_algs["text"] = HeuristicTextCluster

if __name__ == "__main__":
    with HeuristicTranscription(245,"development") as project:
        project.__aggregate__()