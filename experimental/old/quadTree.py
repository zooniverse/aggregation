__author__ = 'greg'
import numpy as np


class Node:
    def __init__(self,min_x,min_y,max_x,max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.markings = []
        self.users = []
        self.children = None
        self.split_x = None
        self.split_y = None


    def __give_to_child__(self,m,u):
        assert(len(self.children) == 4)
        if m[0] <= self.split_x:
            if m[1] <= self.split_y:
                self.children[0].__add_point__((m,u))
            else:
                self.children[1].__add_point__((m,u))
        else:
            if m[1] <= self.split_y:
                self.children[2].__add_point__((m,u))
            else:
                self.children[3].__add_point__((m,u))


    def __add_point__(self,(m,u)):
        #does this node have children, if so we should pass this pt onto the appropriate child
        if self.children is not None:
            self.__give_to_child__(m,u)
        #else we are doing our own insertion
        else:
            self.markings.append(m)
            self.users.append(u)
            #have we now exceeded our capacity - hard code for now to be 10
            if len(self.markings) == 10:
                #find out split points
                self.split_x = np.median(zip(*self.markings)[0])
                self.split_y = np.median(zip(*self.markings)[1])

                #create the children
                self.children = []
                self.children.append(Node(self.min_x,self.min_y,self.split_x,self.split_y))
                self.children.append(Node(self.min_x,self.split_y,self.split_x,self.max_y))
                self.children.append(Node(self.split_x,self.min_y,self.max_x,self.split_y))
                self.children.append(Node(self.split_x,self.split_y,self.max_x,self.max_y))

                for m2,u2 in zip(self.markings,self.users):
                    self.__give_to_child__(m2,u2)

                #just keep the memory clean
                self.markings = []
                self.users = []

    def __ward_traverse__(self):
        if self.children is None:
            
            print self.markings
            print self.users
        else:
            self.children[0].__ward_traverse__()
            self.children[1].__ward_traverse__()

