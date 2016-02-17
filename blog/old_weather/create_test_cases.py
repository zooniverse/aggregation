from backup_weather.learning import NearestNeighbours
try:
    import Image
except ImportError:
    from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np

image_directory = "/home/ggdhines/Databases/old_weather/aligned_images/"
# log_pages = list(os.listdir(image_directory))

from cassandra.cluster import Cluster

cluster = Cluster()


cassandra_session = cluster.connect("active_weather")

classification_algorithm = NearestNeighbours(collect_gold_standard=False)

# cell_columns = [(713,821),(821,890),(1067,1252),(1527,1739),(1739,1837),(1837,1949),(1949,2053),(2053,2156)] #(510,713),
# cell_rows = [(1226,1320),(1320,1377),(1377,1437),(1437,1495),(1495,1555),(1555,1613),(1613,1673),(1673,1731),(1731,1790),(1790,1848),(1848,1907),(1907,1967)]

aligned_subjects = list(os.listdir(image_directory))

# print aligned_subjects[:10]
# assert False

cell_columns = [(a.lb,a.ub) for a in cassandra_session.execute("select * from columns")]
cell_rows = [(a.lb,a.ub) for a in cassandra_session.execute("select * from rows")]


for f_count,f_name in enumerate(aligned_subjects[:10]):
    if not f_name.endswith(".JPG"):
        continue

    image_file = cbook.get_sample_data(image_directory+f_name)
    image = plt.imread(image_file)

    fig = plt.figure()
    fig.set_size_inches(52,78)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(image)

    p_index = f_name.rfind(".")
    base_fname = f_name[:p_index]

    print f_name

    for row_index,(row_lb,row_ub) in enumerate(cell_rows):
        for column_index,(column_lb,column_ub) in enumerate(cell_columns):




                sizes = np.shape(image)
                height = float(sizes[0])
                width = float(sizes[1])
                # fig.set_size_inches(width/height, 1, forward=False)
                # ax = plt.Axes(fig, [0., 0., 1., 1.])


                ax.set_axis_off()
                # ax.set_xticks()

                plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off')

                plt.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelleft='off')

                # fig.add_axes(ax)

                # fig.add_axes(ax)
                ax.set_xlim((0,width))
                ax.set_ylim((height,0))
                ax.plot([column_lb,column_ub],[row_lb,row_lb],"-r",linewidth=3.0)
                ax.plot([column_lb,column_ub],[row_ub,row_ub],"-r",linewidth=3.0)
                ax.plot([column_lb,column_lb],[row_lb,row_ub],"-r",linewidth=3.0)
                ax.plot([column_ub,column_ub],[row_lb,row_ub],"-r",linewidth=3.0)
                # ax.tight_layout()
                plt.subplots_adjust(bottom=0, right =1, top=1,left=0)
    plt.savefig("/home/ggdhines/Databases/"+f_name,bbox_inches='tight', pad_inches=0,dpi=72)
    raw_input("enter something")
