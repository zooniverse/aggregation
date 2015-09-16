import os, sys
import Tkinter as tkinter
from PIL import ImageTk, Image
from penguin import Penguins,global_workflow_id
from matplotlib.widgets import Slider, Button, RadioButtons
import ttk
import matplotlib.pyplot as plt
import random
from aggregation_api import base_directory
import math
import numpy

DIR_IMGS = base_directory+'/Databases/images/'
DIR_THUMBS = base_directory+'/Databases/thumbnails/'

class Marmot:
    def __init__(self):
        self.project = Penguins()
        # self.project.__migrate__()

        # tkinter stuff
        self.root = tkinter.Tk()
        self.root.geometry('900x700')
        self.root.title("Marmot")
        self.root.resizable(False,False)

        # ttk stuff - congrads if you understand the difference between tkinter and ttk
        self.mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        self.root.option_add('*tearOff', False)

        self.links = []

        self.true_probabilities = {}
        self.false_probabilities = {}

        self.percentage_thresholds = {}
        self.probability_threshold = {}
        self.weightings = {}

        # might have to truly random - but this way, we don't always download new images
        random.seed(1)
        # store all of the subjects in a random order
        self.subjects = []
        # self.subjects = self.project.__get_retired_subjects__(1,True)
        #
        # random.shuffle(self.subjects)
        self.page_index = 0
        self.step_size = 45


        # see for deleting previous thumbnails when we go to a new page
        self.thumbnails = []

        self.true_positives = {}
        self.false_positives = {}
        self.unknown_positives = {}

        self.run_mode = None

        # used that we can store values when we go back to an image
        self.matplotlib_points = {}
        self.probabilities = {}

        # by default no
        self.require_gold_standard = False

    def __create_thumb__(self,subject_id):
        fname = self.project.__image_setup__(subject_id)
        openImg = Image.open(fname)
        openImg.thumbnail((250, 250))
        openImg.save(DIR_THUMBS + subject_id+".jpg")

    def __image_select__(self,require_gold_standard=False):
        # setup for when the user wants to explore results that don't have a gold standard
        return self.project.__get_retired_subjects__(1,False)
        # random.shuffle(self.subjects)

    def __run__(self):
        # create the welcome window
        run_type = None
        t = tkinter.Toplevel(self.root)
        t.resizable(False,False)
        frame = ttk.Frame(t, padding="3 3 12 12")
        frame.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        ttk.Label(frame,text="Welcome to Marmot.").grid(column=1,row=1)
        def setup(require_gold_standard):
            # this will determine the whole run mode from here on in
            self.require_gold_standard = require_gold_standard
            self.subjects = self.__image_select__(require_gold_standard)
            # if r == "a":
            #     self.subjects = self.project.__get_retired_subjects__(1,True)
            #     self.run_mode = "a"
            # else:
            #     # when we want to explore subjects which don't have gold standard
            #     # basically creating some as we go
            #     # False => read in all subjects, not just those with gold standard annotations
            #     # todo - takes a while in read in all subjects. Better way?
            #     self.subjects = self.project.__get_retired_subjects__(1,False)
            #     self.run_mode = "b"
            random.shuffle(self.subjects)

            self.__thumbnail_display__()
            self.__add_buttons__()

            t.destroy()

        ttk.Button(frame, text="Explore results using existing expert annotations", command = lambda : setup(True)).grid(column=1, row=2)
        ttk.Button(frame, text="Explore and create gold standard on the fly", command = lambda : setup(False)).grid(column=1, row=3)

        t.lift(self.root)

        # self.outputButtons()
        self.root.mainloop()

    def __calculate__(self):
        if self.percentage_thresholds != {}:
            plt.close()

            subject_ids = self.probability_threshold.keys()
            X = [self.probability_threshold[s] for s in subject_ids]
            Y = [self.weightings[s] for s in subject_ids]

            plt.plot(X,Y,'.')
            plt.xlim((-0.02,1.02))
            plt.show()

    def __thumbnail_display__(self):
        # destroy any previously existing thumbnails - for when we're flipping through pages
        for thumb_index in range(len(self.thumbnails)-1,-1,-1):
            old_thumb = self.thumbnails.pop(thumb_index)
            old_thumb.destroy()

        for ii,subject_id in enumerate(self.subjects[9*self.page_index:9+9*self.page_index]):
            # do we already have a thumb for this file?
            thumb_path = DIR_THUMBS+str(subject_id)+".jpg"
            if not os.path.exists(thumb_path):
                self.__create_thumb__(subject_id)

            render_image = ImageTk.PhotoImage(file=thumb_path)

            but = ttk.Button(self.root, image=render_image)
            but.grid(column=ii/3+1, row=(1+ii)%3,sticky=tkinter.W)

            # the interaction with the subject will depend on whether we have gold standard data for it or not
            # if not, the user will need to create some
            if self.require_gold_standard:
                assert False
                # but.bind('<Button-1>', lambda event,t=thumb_path: self.(t) if self.run_mode == "a" else self.__create_gold_standard__(t))
            else:
                but.bind('<Button-1>', lambda event,t=thumb_path: self.__create_gold_standard__(t))

            self.thumbnails.append(but)

            # sigh - I hate having to do this
            # MUST keep - otherwise garbage collection in Python will remove it
            self.links.append(render_image)

        # todo - this window is not actually popping up
        # determine which of the subjects we are interested in have actually been processed
        # we may need to do some additional aggregation
        aggregated_subjects = self.project.__get_aggregated_subjects__(-1)

        not_aggregated = [s for s in self.subjects[:self.step_size] if s not in aggregated_subjects]

        # print not_aggregated
        if not_aggregated != []:
            self.project.__aggregate__([-1],self.subjects[:self.step_size])


    def __increment__(self):
        self.page_index += 1
        self.__thumbnail_display__()

    def __decrement__(self):
        self.page_index -= 1
        self.__thumbnail_display__()

    def __add_buttons__(self):
        # for ii,thumbfile in enumerate(thumbfiles[:3]):

        ttk.Button(self.root, text="<--", command=self.__decrement__).grid(column=2, row=4)
        ttk.Button(self.root, text="-->", command=self.__increment__).grid(column=2, row=5)
        # ttk.Button(self.root, text="Threshold Plot", command=self.__calculate__).grid(column=1, row=5)
        # ttk.Button(self.root, text="Re-aggregate", command=self.__reaggregate__).grid(column=1, row=6)
        ttk.Button(self.root, text="ROC estimate", command=self.__roc_estimate__).grid(column=1, row=6)

    def __roc_estimate__(self):
        plt.close()
        true_positives = []
        false_positives = []
        unknowns = []

        for subject_id in self.true_positives:
            for pt in self.true_positives[subject_id]:
                true_positives.append(self.probabilities[subject_id][pt])

            for pt in self.false_positives[subject_id]:
                false_positives.append(self.probabilities[subject_id][pt])

            for pt in self.unknown_positives[subject_id]:
                unknowns.append(self.probabilities[subject_id][pt])

        overall_probabilities = true_positives[:]
        overall_probabilities.extend(false_positives)

        # remove duplicates and sort
        overall_probabilities = sorted(list(set(overall_probabilities)),reverse=True)

        X = []
        Y = []

        for p in overall_probabilities:
            num_true = sum([1 for p1 in true_positives if p1 >= p])
            num_false = sum([1 for p1 in false_positives if p1 >= p])

            # treat them as positives
            num_true += sum([1 for p1 in unknowns if p1 >= p])

            Y.append(num_true)
            X.append(num_false)

        X = [x/float(max(X)) for x in X]
        Y = [y/float(max(Y)) for y in Y]

        plt.plot(X,Y,"o-")

        X = []
        Y = []

        for p in overall_probabilities:
            num_true = sum([1 for p1 in true_positives if p1 >= p])
            num_false = sum([1 for p1 in false_positives if p1 >= p])

            # treat them as negatives
            num_false += sum([1 for p1 in unknowns if p1 >= p])

            Y.append(num_true)
            X.append(num_false)

        X = [x/float(max(X)) for x in X]
        Y = [y/float(max(Y)) for y in Y]

        plt.plot(X,Y,"o-")

        plt.show()


    def __reaggregate__(self):
        """
        just rerun aggregation stuff for the current selection of subjects
        :return:
        """
        # todo - just rid of the hard coded 45
        assert (self.true_positives != {}) or (self.false_positives != {})
        self.project.__aggregate__([-1],self.subjects[:45],(self.true_positives,self.false_positives))

    def __update_threshold__(self,new_percentile_threshold,matplotlib_objects):
        """
        update whether we think objects are true positives or not - without using gold standard data
        so this is for creating gold standard
        returns a tuple of all TP probabilities and the FP ones too, according to the given threshold
        :param new_percentile_threshold:
        :return:
        """
        assert isinstance(matplotlib_objects,dict)
        # TP = []
        # FP = []

        # if self.probabilities == []:
        #     return None,([],[],[],[])
        objects,individual_probabilities = zip(*matplotlib_objects.values())
        prob_threshold = numpy.percentile(individual_probabilities,(1-new_percentile_threshold)*100)
        print new_percentile_threshold
        print individual_probabilities
        print prob_threshold
        print

        # clusters we have corrected identified as true positivies
        green_pts = []
        # clusters we have incorrectly identified as true positives
        red_pts = []
        # clusters have incorrectly idenfitied as false positivies
        yellow_pts = []
        # etc.
        blue_pts = []


        for center,(obj,prob_existence) in matplotlib_objects.items():
            # x,y = matplotlib_pt.get_data()
            # x = x[0]
            # y = y[0]
            # if correct_pts is not None:
            #     if prob_existence >= prob_threshold:
            #         # based on the threshold - we think this point exists
            #         if center in correct_pts:
            #             # woot - we were right
            #             matplotlib_pt.set_color("green")
            #             # green_pts.append(prob_existence)
            #         else:
            #             # boo - we were wrong
            #             matplotlib_pt.set_color("red")
            #             # green_pts.append(prob_existence)
            #     else:
            #         # we think this point is a false positive
            #         if center in correct_pts:
            #             matplotlib_pt.set_color("yellow")
            #             # green_pts.append(prob_existence)
            #         else:
            #             matplotlib_pt.set_color("blue")
            #             # green_pts.append(prob_existence)
            # else:
            # in this case, with no expert data, we are assuming that all points accepted
            # are correctly accepted and making no judgement about rejected points
            # do not change any points which have been assigned to be a false positive
            if prob_existence >= prob_threshold:
                if obj.get_color() != "red":
                    obj.set_color("green")

                    green_pts.append(prob_existence)
            else:
                if obj.get_color() != "red":
                    obj.set_color("yellow")
                    print "yellow"
                    yellow_pts.append(prob_existence)
        print (1-new_percentile_threshold)*100
        print prob_threshold
        print green_pts
        print yellow_pts
        print

        return prob_threshold

    def __create_gold_standard__(self,thumb_path):
        print "hello world"
        slash_index = thumb_path.rindex("/")
        subject_id = thumb_path[slash_index+1:-4]

        # get the clusters

        # close any previously open graph
        plt.close()
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

        # get the cluster markings from the aggregation api
        self.probabilities[subject_id] = self.project.__get_cluster_markings__(-1,subject_id,1,"point")

        # if this is the first time we are seeing this particular image
        # by default, accept the 50% in terms of probability threshold
        if subject_id not in self.percentage_thresholds:
            self.percentage_thresholds[subject_id] = 0.5

        # plot centers
        matplotlib_objects = self.__plot_cluster_markings__(self.probabilities[subject_id],"point",axes,self.percentage_thresholds[subject_id])

        # if this is the first time we've seen this image
        # accept all points above the threshold
        # any point below the threshold is just unknown - NOT a false positive

        if subject_id not in self.true_positives:
            # ,"point",axes,self.percentage_thresholds[subject_id]

            # use the colour to determine if a point has been labelled as true positive
            self.true_positives[subject_id] = [pt for pt,(obj,prob) in matplotlib_objects.items() if obj.get_color() == "green"]
            # by default any point above the threshold is just unknown - NOT a false positive
            self.false_positives[subject_id] = []
            self.unknown_positives[subject_id] = [pt for pt in matplotlib_objects if pt not in self.true_positives[subject_id]]

        else:
            # immediately reload previous colours
            for pt,(obj,prob) in matplotlib_objects.items():
                if pt in self.true_positives[subject_id]:
                    obj.set_color("green")
                elif pt in self.false_positives[subject_id]:
                    obj.set_color("red")
                else:
                    obj.set_color("yellow")
            fig.canvas.draw_idle()

        dimensions = self.project.__plot_image__(subject_id,axes)

        plt.subplots_adjust(bottom=0.2)
        # axcolor = 'lightgoldenrodyellow'

        # todo - 1.92 is a hack for penguin watch
        # this rescaling must happen before we add the slider axes - because why not
        if dimensions is not None:
            plt.axis((0,dimensions["width"]/1.92,dimensions["height"]/1.92,0))
            # plt.xlim((dimensions["height"],0))
            # plt.ylim((0,dimensions["width"]))
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
        # todo - has got to be a better to initalize this

        # self.weightings[subject_id] = len(matplotlib_points)

        threshold_silder = Slider(axfreq, 'Percentage', 0., 1., valinit=self.percentage_thresholds[subject_id])

        # self.project.__get_expert_annotations__(-1,subject_id)

        # needs to be an inner function - grrrr
        def update(val):
            new_threshold = threshold_silder.val
            # store this new threshold for future use i.e. when the image is reopened
            self.percentage_thresholds[subject_id] = new_threshold
            self.probability_threshold[subject_id] = self.__update_threshold__(new_threshold,matplotlib_objects)

            # if a point is not a true positive - we make no claim about what it is
            self.true_positives[subject_id] = [pt for pt,(obj,prob) in matplotlib_objects.items() if obj.get_color() == "green"]
            self.unknown_positives[subject_id] = [pt for pt,(obj,prob) in matplotlib_objects.items() if obj.get_color() == "yellow"]
            # false positives should not change
            fig.canvas.draw_idle()

        def onpick3(event):
            x = event.xdata
            y = event.ydata
            if (x is None) or (y is None):
                return

            # pt = event.xdata,event.ydata
            nearest_pt = None
            closest_dist = float("inf")
            for x2,y2 in matplotlib_objects.keys():
                dist = math.sqrt((x-x2)**2+(y-y2)**2)
                if dist < closest_dist:
                    closest_dist = dist
                    nearest_pt = (x2,y2)

            if closest_dist < 20:
                obj,prob = matplotlib_objects[nearest_pt]
                # new_color = color_list[(color_list.index(color)+1)%3]

                # matplotlib_objects[nearest_pt] = pt,prob,new_color
                # pt.set_color(new_color)

                old_colour = obj.get_color()

                # first is going from red to green
                if old_colour == "red":
                    # gone from a false positive to a true positive
                    self.false_positives[subject_id].remove(nearest_pt)
                    self.true_positives[subject_id].append(nearest_pt)
                    obj.set_color("green")
                elif old_colour == "green":
                    # gone from true positive to don't know
                    self.true_positives[subject_id].remove(nearest_pt)
                    self.unknown_positives[subject_id].append(nearest_pt)
                    obj.set_color("yellow")
                else:
                    # gone from don't know to false positive
                    self.unknown_positives[subject_id].remove(nearest_pt)
                    self.false_positives[subject_id].append(nearest_pt)
                    obj.set_color("red")

                fig.canvas.draw_idle()

        threshold_silder.on_changed(update)
        fig.canvas.mpl_connect('button_press_event', onpick3)

        plt.show()

    def __compare_aggregations__(self,agg1,agg2):
        X = []
        Y = []
        plt.close()
        for subject_id in agg1:
            if subject_id == "param":
                continue
            print agg1[subject_id]

            assert subject_id in agg2
            for task_id in agg1[subject_id]:
                if task_id == "param":
                    continue

                print agg1[subject_id][task_id]

                assert task_id in agg2[subject_id]
                for shape in agg1[subject_id][task_id]:
                    if shape == "param":
                        continue

                    print agg1[subject_id][task_id][shape]

                    assert shape in agg2[subject_id][task_id]

                    for cluster_index in agg1[subject_id][task_id][shape]:
                        if (cluster_index == "param") or (cluster_index == "all_users"):
                            continue

                        assert cluster_index in agg2[subject_id][task_id][shape]
                        print cluster_index

                        X.append(agg1[subject_id][task_id][shape][cluster_index]["existence"][0][1])
                        Y.append(agg2[subject_id][task_id][shape][cluster_index]["existence"][0][1])
        plt.close()
        plt.plot(X,Y,"o")
        plt.show()

    def __plot_cluster_markings__(self,cluster_list,shape,axes,percentile_threshold=None,correct_pts=None,incorrect_pts=None):
        """
        take a listing of cluster centers and mark them on the image - cluster list is a dictionary with
        the keywords the centers and values the probabilities of existence. With these, we can figure out which
        colour to make each image
        """
        # main thing we are returning so that objects can be updated
        matplotlib_objects = {}

        # convert from a percentile threshold to a probability threshold
        if percentile_threshold is not None:
            prob_threshold = numpy.percentile(cluster_list.values(),(1-percentile_threshold)*100)
            marker = '.'
        else:
            prob_threshold = None
            marker = '^'

        for center,prob_existence in cluster_list.items():
            if shape == "point":
                # with whatever alg we used, what do we think the probability is that
                # this cluster actually exists?
                # if we have gold standard to compare to - use that to determine the colour
                if correct_pts is not None:
                    # if is equal to None - just compared directly against gold standard with out threshold
                    if prob_threshold is not None:
                        # we have both a threshold and gold standard - gives us four options
                        if prob_existence >= prob_threshold:
                            # based on the threshold - we think this point exists
                            if center in correct_pts:
                                # woot - we were right
                                color = "green"
                            else:
                                # boo - we were wrong
                                color = "red"
                        else:
                            # we think this point is a false positive
                            if center in correct_pts:
                                # boo - we were wrong
                                color = "yellow"
                            else:
                                # woot
                                color = "blue"
                    else:
                        # we have just the gold standard - so we are purely reviewing the expert results
                        if center in correct_pts:
                            color = "green"
                        else:
                            color = "red"
                    matplotlib_objects[center] = axes.plot(center[0],center[1],marker=marker,color=color)[0],prob_existence
                else:
                    # we have nothing to compare against - so we are not showing correctness so much
                    # as just showing which points would be rejected/accepted with the default understanding
                    # that points will be correctly accepted - points that are rejected - we make no statement about
                    # they will not be included in the gold standard
                    if prob_existence >= prob_threshold:
                        color = "green"
                        # matplotlib_cluster[center] = axes.plot(center[0],center[1],".",color="green"),prob_existence
                    else:
                        # we think this is a false positive
                        color = "yellow"
                        # matplotlib_cluster[center] = axes.plot(center[0],center[1],".",color="red"),prob_existence
                    matplotlib_objects[center] = axes.plot(center[0],center[1],marker=marker,color=color)[0],prob_existence
        return matplotlib_objects


    # def gold_update(self,thumb_path):
    #     slash_index = thumb_path.rindex("/")
    #     subject_id = thumb_path[slash_index+1:-4]
    #
    #
    #
    #     # if we are looking at a image for the first time
    #     if subject_id not in self.true_positives:
    #         self.true_positives[subject_id] = self.project.__get_correct_points__(global_workflow_id,subject_id,1,"point")
    #         print "slowly but surely"
    #         print self.true_positives[subject_id]
    #
    #     # close any previously open graph
    #     plt.close()
    #     fig = plt.figure()
    #     axes = fig.add_subplot(1, 1, 1)
    #
    #     # grrr - inner function needed again
    #     def onpick3(event):
    #         color_list = ["green","yellow","red"]
    #
    #         x = event.xdata
    #         y = event.ydata
    #         pt = event.xdata,event.ydata
    #         nearest_pt = None
    #         closest_dist = float("inf")
    #         for x2,y2 in matplotlib_points.keys():
    #             dist = math.sqrt((x-x2)**2+(y-y2)**2)
    #             if dist < closest_dist:
    #                 closest_dist = dist
    #                 nearest_pt = (x2,y2)
    #
    #         if closest_dist < 20:
    #             pt,prob,color = matplotlib_points[nearest_pt]
    #             new_color = color_list[(color_list.index(color)+1)%3]
    #
    #             matplotlib_points[nearest_pt] = pt,prob,new_color
    #             pt.set_color(new_color)
    #             fig.canvas.draw_idle()
    #
    #             print self.false_positives[subject_id]
    #             # update the gold standard accordingly
    #             if new_color == "green":
    #                 print 1
    #                 # gone from a false positive to a true positive
    #                 self.false_positives[subject_id].remove(nearest_pt)
    #                 self.true_positives[subject_id].append(nearest_pt)
    #             elif new_color == "yellow":
    #                 print 2
    #                 # gone from true positive to don't know
    #                 self.true_positives[subject_id].remove(nearest_pt)
    #             else:
    #                 print 3
    #                 # gone from don't know to false positive
    #                 self.false_positives[subject_id].append(nearest_pt)
    #
    #     def ibcc_check(event):
    #         """
    #         for when we want to see the affects of using gold standard data
    #         :param event:
    #         :return:
    #         """
    #         related_subjects = self.project.__get_related_subjects__(global_workflow_id,subject_id)[:10]
    #
    #         base_aggregations = self.project.__aggregate__([global_workflow_id],related_subjects,store_values=False)
    #
    #         # now try it with gold standard
    #         print (self.true_positives[subject_id],self.false_positives[subject_id])
    #
    #         pos_gold_standard = {subject_id :self.true_positives[subject_id]}
    #         neg_gold_standard = {subject_id :self.false_positives[subject_id]}
    #
    #
    #         extra_aggregations = self.project.__aggregate__([global_workflow_id],related_subjects,gold_standard_clusters=(pos_gold_standard,neg_gold_standard),expert=self.project.experts[0],store_values=False)
    #         self.__compare_aggregations__(base_aggregations,extra_aggregations)
    #
    #     print "here maybe>?"
    #     self.project.__plot_image__(subject_id,axes)
    #     matplotlib_points = self.project.__plot_cluster_results__(-1,subject_id,1,"point",axes,correct_pts=self.true_positives[subject_id]) #self.percentage_thresholds[subject_id],
    #
    #     # we can only set the false positives once we have seen all the points
    #     if subject_id not in self.false_positives:
    #         self.false_positives[subject_id] = [pt for pt in matplotlib_points.keys() if pt not in self.true_positives[subject_id]]
    #         print self.false_positives[subject_id]
    #     plt.subplots_adjust(left=0.25, bottom=0.25)
    #     fig.canvas.mpl_connect('button_press_event', onpick3)
    #     # axcolor = 'lightgoldenrodyellow'
    #
    #     dimensions = self.project.__plot_image__(subject_id,axes)
    #     if dimensions is not None:
    #         plt.axis((0,dimensions["width"]/1.92,dimensions["height"]/1.92,0))
    #
    #     axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    #     bprev = Button(axfreq, 'IBCC check')
    #     bprev.on_clicked(ibcc_check)
    #
    #     plt.show()



m = Marmot()
m.__run__()





