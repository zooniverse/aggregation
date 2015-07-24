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
        for ii,subject_id in enumerate(self.project.__get_aggregated_subjects__(1)[:2]):
            # do we already have a thumb for this file?
            thumb_path = DIR_THUMBS+str(subject_id)+".jpg"
            print thumb_path
            if not os.path.exists(thumb_path):
                self.__create_thumb__(subject_id)

            render_image = ImageTk.PhotoImage(file=thumb_path)
            print render_image
            # but = tkinter.Button(self.root, image=render_image)
            # but.pack(side="left")

            # but.destroy

            but = ttk.Button(self.root, image=render_image)
            but.grid(column=1, row=1+ii,sticky=tkinter.W)
            but.bind('<Button-1>', lambda event:self.m(thumb_path))

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
        print subject_id
        plt.close()
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

        self.project.__plot_image__(subject_id,axes)
        self.project.__plot_cluster_results__(1,subject_id,1,"point",axes,0.5)
        plt.subplots_adjust(left=0.25, bottom=0.25)
        # axcolor = 'lightgoldenrodyellow'
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
        threshold_silder = Slider(axfreq, 'Percentage', 0., 1., valinit=0.5)

        # needs to be an inner function - grrrr
        def update(val):
            new_threshold = threshold_silder.val
            self.project.__update_threshold__(new_threshold)

        threshold_silder.on_changed(update)

        plt.show()



m = Marmot()
m.__run__()





