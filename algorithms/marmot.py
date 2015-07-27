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

DIR_IMGS = base_directory+'/Databases/images/'
DIR_THUMBS = base_directory+'/Databases/thumbnails/'

class Marmot:
    def __init__(self):
        self.project = Penguins()

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

        self.run_mode = None

    def __create_thumb__(self,subject_id):
        fname = self.project.__image_setup__(subject_id)
        openImg = Image.open(fname)
        openImg.thumbnail((250, 250))
        openImg.save(DIR_THUMBS + subject_id+".jpg")

    def __create_gold_standard__(self):
        # setup for when the user wants to explore results that don't have a gold standard
        self.subjects = self.project.__get_retired_subjects__(1,False)
        random.shuffle(self.subjects)

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
        def setup(r):
            if r == "a":
                self.subjects = self.project.__get_retired_subjects__(1,True)
                self.run_mode = "a"
            else:
                # when we want to explore subjects which don't have gold standard
                # basically creating some as we go
                # False => read in all subjects, not just those with gold standard annotations
                # todo - takes a while in read in all subjects. Better way?
                self.subjects = self.project.__get_retired_subjects__(1,False)
                self.run_mode = "b"
            random.shuffle(self.subjects)

            self.__thumbnail_display__()
            self.outputButtons()

            t.destroy()

        ttk.Button(frame, text="Explore results using existing expert annotations", command = lambda : setup("a")).grid(column=1, row=2)
        ttk.Button(frame, text="Explore and create gold standard on the fly", command = lambda : setup("b")).grid(column=1, row=3)

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
        for thumb_index in range(len(self.thumbnails)-1,-1,-1):
            old_thumb = self.thumbnails.pop(thumb_index)
            old_thumb.destroy()

        # for ii,subject_id in enumerate(self.project.__get_aggregated_subjects__(1)[:9]):
        for ii,subject_id in enumerate(self.subjects[9*self.page_index:9+9*self.page_index]):
            # do we already have a thumb for this file?
            thumb_path = DIR_THUMBS+str(subject_id)+".jpg"
            if not os.path.exists(thumb_path):
                self.__create_thumb__(subject_id)

            render_image = ImageTk.PhotoImage(file=thumb_path)

            # but = tkinter.Button(self.root, image=render_image)
            # but.pack(side="left")

            # but.destroy

            but = ttk.Button(self.root, image=render_image)
            but.grid(column=ii/3+1, row=(1+ii)%3,sticky=tkinter.W)
            but.bind('<Button-1>', lambda event,t=thumb_path: self.gold_update(t) if self.run_mode == "a" else self.create_gold(t))

            self.thumbnails.append(but)

            # sigh - I hate having to do this
            self.links.append(render_image)

        # todo - this window is not actually popping up
        # determine which of the subjects we are interested in have actually been processed
        # we may need to do some additional aggregation
        aggregated_subjects = self.project.__get_aggregated_subjects__(-1)
        print aggregated_subjects


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

    def outputButtons(self):
        # for ii,thumbfile in enumerate(thumbfiles[:3]):

        ttk.Button(self.root, text="<--", command=self.__decrement__).grid(column=2, row=4)
        ttk.Button(self.root, text="-->", command=self.__increment__).grid(column=2, row=5)
        ttk.Button(self.root, text="Threshold Plot", command=self.__calculate__).grid(column=1, row=5)
        ttk.Button(self.root, text="Re-aggregate", command=self.__reaggregate__).grid(column=1, row=6)

    def __reaggregate__(self):
        """
        just rerun aggregation stuff for the current selection of subjects
        :return:
        """
        # todo - just rid of the hard coded 45
        assert (self.true_positives != {}) or (self.false_positives != {})
        self.project.__aggregate__([-1],self.subjects[:45],(self.true_positives,self.false_positives))

    def create_gold(self,thumb_path):
        print "hello world"
        slash_index = thumb_path.rindex("/")
        subject_id = thumb_path[slash_index+1:-4]

        # if we are looking at a image for the first time
        if subject_id not in self.percentage_thresholds:
            self.percentage_thresholds[subject_id] = 0.5

        if subject_id not in self.false_positives:
            self.false_positives[subject_id] = []

        # close any previously open graph
        plt.close()
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

        dimensions = self.project.__plot_image__(subject_id,axes)
        matplotlib_points = self.project.__plot_cluster_results__(-1,subject_id,1,"point",axes,self.percentage_thresholds[subject_id])
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
        if subject_id not in self.true_positives:
            prob_threshold,colours = self.probability_threshold[subject_id] = self.project.__update_threshold__(self.percentage_thresholds[subject_id],matplotlib_points)
            self.true_positives[subject_id] = colours[0]

        self.weightings[subject_id] = len(matplotlib_points)

        threshold_silder = Slider(axfreq, 'Percentage', 0., 1., valinit=self.percentage_thresholds[subject_id])

        # self.project.__get_expert_annotations__(-1,subject_id)

        # needs to be an inner function - grrrr
        def update(val):
            new_threshold = threshold_silder.val
            self.percentage_thresholds[subject_id] = new_threshold
            prob_threshold,colours = self.probability_threshold[subject_id] = self.project.__update_threshold__(new_threshold,matplotlib_points)
            # green pts are true positives
            self.true_positives[subject_id] = colours[0]

            fig.canvas.draw_idle()

        def onpick3(event):
            color_list = ["green","yellow","red"]

            x = event.xdata
            y = event.ydata
            pt = event.xdata,event.ydata
            nearest_pt = None
            closest_dist = float("inf")
            for x2,y2 in matplotlib_points.keys():
                dist = math.sqrt((x-x2)**2+(y-y2)**2)
                if dist < closest_dist:
                    closest_dist = dist
                    nearest_pt = (x2,y2)

            if closest_dist < 20:
                pt,prob,color = matplotlib_points[nearest_pt]
                new_color = color_list[(color_list.index(color)+1)%3]

                matplotlib_points[nearest_pt] = pt,prob,new_color
                pt.set_color(new_color)
                fig.canvas.draw_idle()

                print self.false_positives[subject_id]
                # update the gold standard accordingly
                if new_color == "green":
                    print 1
                    # gone from a false positive to a true positive
                    self.false_positives[subject_id].remove(nearest_pt)
                    self.true_positives[subject_id].append(nearest_pt)
                elif new_color == "yellow":
                    print 2
                    # gone from true positive to don't know
                    self.true_positives[subject_id].remove(nearest_pt)
                else:
                    print 3
                    # gone from don't know to false positive
                    self.false_positives[subject_id].append(nearest_pt)

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

    def gold_update(self,thumb_path):
        slash_index = thumb_path.rindex("/")
        subject_id = thumb_path[slash_index+1:-4]



        # if we are looking at a image for the first time
        if subject_id not in self.true_positives:
            self.true_positives[subject_id] = self.project.__get_correct_points__(global_workflow_id,subject_id,1,"point")
            print "slowly but surely"
            print self.true_positives[subject_id]

        # close any previously open graph
        plt.close()
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

        # grrr - inner function needed again
        def onpick3(event):
            color_list = ["green","yellow","red"]

            x = event.xdata
            y = event.ydata
            pt = event.xdata,event.ydata
            nearest_pt = None
            closest_dist = float("inf")
            for x2,y2 in matplotlib_points.keys():
                dist = math.sqrt((x-x2)**2+(y-y2)**2)
                if dist < closest_dist:
                    closest_dist = dist
                    nearest_pt = (x2,y2)

            if closest_dist < 20:
                pt,prob,color = matplotlib_points[nearest_pt]
                new_color = color_list[(color_list.index(color)+1)%3]

                matplotlib_points[nearest_pt] = pt,prob,new_color
                pt.set_color(new_color)
                fig.canvas.draw_idle()

                print self.false_positives[subject_id]
                # update the gold standard accordingly
                if new_color == "green":
                    print 1
                    # gone from a false positive to a true positive
                    self.false_positives[subject_id].remove(nearest_pt)
                    self.true_positives[subject_id].append(nearest_pt)
                elif new_color == "yellow":
                    print 2
                    # gone from true positive to don't know
                    self.true_positives[subject_id].remove(nearest_pt)
                else:
                    print 3
                    # gone from don't know to false positive
                    self.false_positives[subject_id].append(nearest_pt)

        def ibcc_check(event):
            """
            for when we want to see the affects of using gold standard data
            :param event:
            :return:
            """
            related_subjects = self.project.__get_related_subjects__(global_workflow_id,subject_id)[:10]

            base_aggregations = self.project.__aggregate__([global_workflow_id],related_subjects,store_values=False)

            # now try it with gold standard
            print (self.true_positives[subject_id],self.false_positives[subject_id])

            pos_gold_standard = {subject_id :self.true_positives[subject_id]}
            neg_gold_standard = {subject_id :self.false_positives[subject_id]}


            extra_aggregations = self.project.__aggregate__([global_workflow_id],related_subjects,gold_standard_clusters=(pos_gold_standard,neg_gold_standard),expert=self.project.experts[0],store_values=False)
            self.__compare_aggregations__(base_aggregations,extra_aggregations)

        print "here maybe>?"
        self.project.__plot_image__(subject_id,axes)
        matplotlib_points = self.project.__plot_cluster_results__(-1,subject_id,1,"point",axes,correct_pts=self.true_positives[subject_id]) #self.percentage_thresholds[subject_id],

        # we can only set the false positives once we have seen all the points
        if subject_id not in self.false_positives:
            self.false_positives[subject_id] = [pt for pt in matplotlib_points.keys() if pt not in self.true_positives[subject_id]]
            print self.false_positives[subject_id]
        plt.subplots_adjust(left=0.25, bottom=0.25)
        fig.canvas.mpl_connect('button_press_event', onpick3)
        # axcolor = 'lightgoldenrodyellow'

        dimensions = self.project.__plot_image__(subject_id,axes)
        if dimensions is not None:
            plt.axis((0,dimensions["width"]/1.92,dimensions["height"]/1.92,0))

        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
        bprev = Button(axfreq, 'IBCC check')
        bprev.on_clicked(ibcc_check)

        plt.show()



m = Marmot()
m.__run__()





