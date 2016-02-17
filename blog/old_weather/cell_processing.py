from backup_weather.learning import NearestNeighbours
try:
    import Image
except ImportError:
    from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
import shutil
image_directory = "/home/ggdhines/Databases/old_weather/pruned_cases/"
# log_pages = list(os.listdir(image_directory))

classification_algorithm = NearestNeighbours(collect_gold_standard=False)

cell_columns = [(713,821),(821,890),(1067,1252),(1527,1739),(1739,1837),(1837,1949),(1949,2053),(2053,2156)] #(510,713),
cell_rows = [(1226,1320),(1320,1377)]

aligned_subjects = list(os.listdir(image_directory))

# print aligned_subjects[:10]
# assert False

for f_count,f_name in enumerate(aligned_subjects[:10]):
    if not f_name.endswith(".JPG"):
        continue

    image_file = cbook.get_sample_data(image_directory+f_name)
    image = plt.imread(image_file)

    p_index = f_name.rfind(".")
    base_fname = f_name[:p_index]

    print f_name

    for row_index,(row_lb,row_ub) in enumerate(cell_rows):
        for column_index,(column_lb,column_ub) in enumerate(cell_columns):
            offset = 8
            r = range(row_lb-offset,row_ub+offset)
            c = range(column_lb-offset,column_ub+offset)

            sub_image = image[np.ix_(r, c)]

            img = Image.fromarray(sub_image, 'RGB')
            # img.convert("L")
            cell_name = "/home/ggdhines/Databases/old_weather/cells/"+base_fname+"_"+str(row_index)+"_"+str(column_index)+".png"
            print row_index,column_index
            img.save(cell_name)

            retval = classification_algorithm.__process_cell__(cell_name,plot=False)

            if (retval != None) and (retval != -1):
                # print retval
                print "hard"
                if retval < 0.78:
                    if not(os.path.isfile("/home/ggdhines/Databases/old_weather/hard_cases/"+f_name)):
                        shutil.copyfile(image_directory+f_name,"/home/ggdhines/Databases/old_weather/hard_cases/"+f_name)

                    image_file = cbook.get_sample_data("/home/ggdhines/Databases/old_weather/hard_cases/"+f_name)
                    image = plt.imread(image_file)

                    fig = plt.figure()

                    sizes = np.shape(image)
                    height = float(sizes[0])
                    width = float(sizes[1])
                    # fig.set_size_inches(width/height, 1, forward=False)
                    # ax = plt.Axes(fig, [0., 0., 1., 1.])
                    fig.set_size_inches(52,78)
                    ax = fig.add_subplot(1, 1, 1)

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
                    im = ax.imshow(image)
                    # fig.add_axes(ax)
                    ax.set_xlim((0,width))
                    ax.set_ylim((height,0))
                    ax.plot([column_lb,column_ub],[row_lb,row_lb],"-r",linewidth=4.0)
                    ax.plot([column_lb,column_ub],[row_ub,row_ub],"-r",linewidth=4.0)
                    ax.plot([column_lb,column_lb],[row_lb,row_ub],"-r",linewidth=4.0)
                    ax.plot([column_ub,column_ub],[row_lb,row_ub],"-r",linewidth=4.0)
                    # ax.tight_layout()
                    plt.subplots_adjust(bottom=0, right =1, top=1,left=0)
                    plt.savefig("/home/ggdhines/Databases/old_weather/hard_cases/"+f_name,bbox_inches='tight', pad_inches=0,dpi=72)
                    plt.close()

# for f_count,f_name in enumerate(log_pages[:40]):
#     if not f_name.endswith(".png"):
#         continue
#     # f_name = "Bear-AG-29-1941-0545_0_6.png"
#     print f_name
#
#     # im = Image.open(image_directory+f_name)
#     # im = im.convert('L')#.convert('LA')
#     # image = np.asarray(im)
#
#     retval = classification_algorithm.__process_cell__(image_directory+f_name,plot=False)
#
#     if retval == -1:
#         break
#
# # for d in range(0,10):
# #     print str(d) + " -- " + str(len(classification_algorithm.transcribed_digits[d]))
#
# print len(classification_algorithm.cells_to_process)
# print len(classification_algorithm.completed_cells)