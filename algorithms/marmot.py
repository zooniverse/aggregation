import os, sys
import Tkinter as tkinter
from PIL import ImageTk, Image
from penguin import Penguins
from matplotlib.widgets import Slider, Button, RadioButtons
import ttk
import matplotlib.pyplot as plt

DIR_IMGS = '/home/ggdhines/Databases/images/'
DIR_THUMBS = '/home/ggdhines/Databases/thumbnails/'

class Marmot:
    def __init__(self):
        self.project = Penguins()

        # tkinter stuff
        self.root = tkinter.Tk()
        self.root.geometry('900x700')
        self.root.title("Marmot")

        # ttk stuff - congrads if you understand the difference between tkinter and ttk
        self.mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        self.links = []

        self.true_probabilities = {}
        self.false_probabilities = {}

        self.percentage_thresholds = {}
        self.probability_threshold = {}
        self.weightings = {}
    def __create_thumb__(self,subject_id):
        fname = self.project.__image_setup__(subject_id)
        openImg = Image.open(fname)
        openImg.thumbnail((250, 250))
        openImg.save(DIR_THUMBS + subject_id+".jpg")

    def __run__(self):
        self.outputButtons()
        self.root.mainloop()

    def outputButtons(self):
        # for ii,thumbfile in enumerate(thumbfiles[:3]):

        # for ii,subject_id in enumerate(self.project.__get_aggregated_subjects__(1)[:9]):
        for ii,subject_id in enumerate(self.project.__get_retired_subjects__(1,True)[:9]):
            # do we already have a thumb for this file?
            thumb_path = DIR_THUMBS+str(subject_id)+".jpg"
            print thumb_path
            if not os.path.exists(thumb_path):
                self.__create_thumb__(subject_id)

            render_image = ImageTk.PhotoImage(file=thumb_path)

            # but = tkinter.Button(self.root, image=render_image)
            # but.pack(side="left")

            # but.destroy

            but = ttk.Button(self.root, image=render_image)
            but.grid(column=ii/3+1, row=(1+ii)%3,sticky=tkinter.W)
            but.bind('<Button-1>', lambda event,t=thumb_path:self.m(t))

            # sigh - I hate having to do this
            self.links.append(render_image)

    def m(self,thumb_path):

        # print event
        # plt.plot(bins, y, 'r--')
        # plt.xlabel('Smarts')
        # plt.ylabel('Probability')
        # plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
        #
        # # Tweak spacing to prevent clipping of ylabel
        # plt.subplots_adjust(left=0.15)
        # plt.show()

        slash_index = thumb_path.rindex("/")
        subject_id = thumb_path[slash_index+1:-4]

        # if we are looking at a image for the first time
        if subject_id not in self.percentage_thresholds:
            self.percentage_thresholds[subject_id] = 0.5



        plt.close()
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

        self.project.__plot_image__(subject_id,axes)
        correct_pts = self.project.__get_correct_points__(1,subject_id,1,"point")
        matplotlib_points = self.project.__plot_cluster_results__(1,subject_id,1,"point",axes,self.percentage_thresholds[subject_id],correct_pts)
        plt.subplots_adjust(left=0.25, bottom=0.25)
        # axcolor = 'lightgoldenrodyellow'
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])


        self.weightings[subject_id] = len(matplotlib_points)


        threshold_silder = Slider(axfreq, 'Percentage', 0., 1., valinit=self.percentage_thresholds[subject_id])

        self.project.__get_expert_annotations__(1,subject_id)

        # needs to be an inner function - grrrr
        def update(val):
            new_threshold = threshold_silder.val
            self.percentage_thresholds[subject_id] = new_threshold
            self.probability_threshold[subject_id] = self.project.__update_threshold__(new_threshold,correct_pts,matplotlib_points)


        threshold_silder.on_changed(update)

        plt.show()



m = Marmot()
m.__run__()





