from __future__ import print_function
import cv2
import csv
import paper_quad
import math
import tesserpy
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage.morphology import disk
import sqlite3 as lite
import numpy as np
import glob
import warnings
from skimage.filters import threshold_otsu, rank

warnings.simplefilter("error", RuntimeWarning)


__author__ = 'ggdhines'



def __identity__(img,line):
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask,[line],0,255,-1)

    return mask

def __non_extreme_regression__(img,line,horizontal):
    # draw on the contour
    grid_contour = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(grid_contour,[line],0,255,-1)

    min_dict = {}
    max_dict = {}
    all_values = {}

    if horizontal:
        for x,y in line:
            try:
                min_dict[x] = min(min_dict[x],y)
            except KeyError:
                min_dict[x] = y

            try:
                max_dict[x] = max(max_dict[x],y)
            except KeyError:
                max_dict[x] = y

            try:
                all_values[x].append(y)
            except KeyError:
                all_values[x] = [y]
    else:
        for y,x in line:
            try:
                min_dict[x] = min(min_dict[x],y)
            except KeyError:
                min_dict[x] = y

            try:
                max_dict[x] = max(max_dict[x],y)
            except KeyError:
                max_dict[x] = y

            try:
                all_values[x].append(y)
            except KeyError:
                all_values[x] = [y]




    # # assert sorted(min_dict.keys()) == sorted(max_dict.keys())
    x_vals = sorted(min_dict.keys())
    # lower_bound = [min_dict[x] for x in x_vals]
    # upper_bound = [max_dict[x] for x in x_vals]

    dist = {x:max_dict[x]-min_dict[x] for x in x_vals}
    median_dist = np.mean(dist.values())
    # print(median_dist)
    # print("===---")
    fitted_x = []
    fitted_y = []
    for x in x_vals:

        if dist[x] <= median_dist:
            fitted_x.extend([x,x])
            fitted_y.extend([max_dict[x],min_dict[x]])

    degrees = 3
    coeff = list(reversed(np.polyfit(fitted_x,fitted_y,degrees)))

    y_bar = [sum([coeff[p]*x_**p for p in range(degrees+1)]) for x_ in fitted_x]
    std = math.sqrt(np.mean([(y1-y2)**2 for (y1,y2) in zip(fitted_y,y_bar)]))*1

    def y_bar(x_,upper):
        return int(round(sum([coeff[p]*x_**p for p in range(degrees+1)]) + upper*std))


    domain = sorted(all_values.keys())
    y_vals = [y_bar(x,-1) for x in domain]
    y_vals.extend([y_bar(x,1) for x in list(reversed(domain))])
    x_vals = list(domain)
    x_vals.extend(list(reversed(domain)))

    if horizontal:
        pts = np.asarray(zip(x_vals,y_vals))
    else:
        pts = np.asarray(zip(y_vals,x_vals))

    mask2 = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask2,[pts],0,255,-1)

    overlap = np.min([grid_contour,mask2],axis=0)

    return overlap



# def __polynomial_correct__(img,line,horizontal,recurse=0):
#     __min_and_max__(line)
#
#
#     if horizontal:
#         x,y = zip(*line)
#         domain = sorted(set(line[:,0]))
#
#
#     else:
#         y,x = zip(*line)
#         domain = sorted(set(line[:,1]))
#
#
#
#     init_mask = np.zeros(img.shape[:2],np.uint8)
#     cv2.drawContours(init_mask,[line],0,255,-1)
#
#     mask2 = np.zeros(img.shape[:2],np.uint8)
#
#     degrees = 2
#     # coeff = list(reversed(np.polyfit(x,y,degrees)))
#
#
#
#     # degrees = 2
#     coeff = list(reversed(np.polyfit(x,y,degrees)))
#     y_bar = [sum([coeff[p]*x_**p for p in range(degrees+1)]) for x_ in x]
#     if recurse == 0:
#         std = math.sqrt(np.mean([(y1-y2)**2 for (y1,y2) in zip(y,y_bar)]))*1
#     else:
#         std = math.sqrt(np.mean([(y1-y2)**2 for (y1,y2) in zip(y,y_bar)]))*1.4
#
#     def y_bar(x_,upper):
#         return int(sum([coeff[p]*x_**p for p in range(degrees+1)]) + upper*std)
#
#
#     y_vals = [y_bar(x,-1) for x in domain]
#     y_vals.extend([y_bar(x,1) for x in list(reversed(domain))])
#     x_vals = list(domain)
#     x_vals.extend(list(reversed(domain)))
#
#     ymax = np.max(y_vals) + 50
#     ymin = np.min(y_vals) - 50
#
#     if horizontal:
#         pts = np.asarray(zip(x_vals,y_vals))
#     else:
#         pts = np.asarray(zip(y_vals,x_vals))
#
#
#     cv2.drawContours(mask2,[pts],0,255,-1)
#
#
#     mask3 = np.min([init_mask,mask2],axis=0)
#
#
#     if recurse == 4:
#         #
#         pass
#         plt.plot(line[:,1],line[:,0])
#
#
#     _,contour, hier = cv2.findContours(mask3.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contour:
#         x,y,w,h = cv2.boundingRect(cnt)
#         perimeter = cv2.arcLength(cnt,True)
#         if min(h,w) > 1 and (perimeter > 500):
#             s = cnt.shape
#             f = np.reshape(cnt,(s[0],s[2]))
#
#
#     return mask3


def __db__(img):
    shape = img.shape
    x,y = np.where(img==0)
    X = np.asarray(zip(x,y))
    db = DBSCAN(eps=1, min_samples=2).fit(X)

    unique_labels = set(db.labels_)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    x_values = []

    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue

        class_member_mask = (db.labels_ == k)

        xy = X[class_member_mask]
        xvalue_range= max(xy[:,1])-min(xy[:,1])
        yvalue_range= max(xy[:,0])-min(xy[:,0])
        x_values.append(xvalue_range)

        if ((xvalue_range/float(yvalue_range)) <= 5 ) and ((yvalue_range/float(xvalue_range)) <= 5 ):
            plt.plot(xy[:, 1], -xy[:, 0], 'o', markerfacecolor=col,markeredgecolor='k', markersize=3)

    print(x_values)
    t = np.median([x for x in x_values if x > 0])
    print(t)
    plt.show()


    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue

        class_member_mask = (db.labels_ == k)

        xy = X[class_member_mask]
        xvalue_range= max(xy[:,1])-min(xy[:,1])
        if xvalue_range == 0:
            continue

        num_in_bins,_ = np.histogram(xy[:,1],bins=xvalue_range)
        bin_diff = [num_in_bins[i+1]-num_in_bins[i] for i in range(num_in_bins.shape[0]-1)]
        if bin_diff == []:
            continue
        if xvalue_range > 2*t:

            print(xvalue_range)
            # print(num_in_bins)
            # print(max(bin_diff))
            plt.plot(xy[:, 1], -xy[:, 0], 'o', markerfacecolor=col,markeredgecolor='k', markersize=3)
            plt.show()

    # plt.show()



def __otsu_bin__(img,invert):
    # with scikit-image
    thresh = threshold_otsu(img)
    # threshed_image = img > thresh

    threshed_image = np.zeros(img.shape,np.uint8)
    threshed_image[np.where(img > thresh)] = 255

    # with open cv
    # _,threshed_image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return threshed_image,thresh


def __binary_threshold_curry__(threshold):
    assert isinstance(threshold,int) or isinstance(threshold,float)

    def __binary_threshold__(img,invert):
        if invert:
            ret,threshed_image = cv2.threshold(img,255-threshold,255,cv2.THRESH_BINARY)
        else:
            ret,threshed_image = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
        return threshed_image,threshold

    return __binary_threshold__

def __pca__(img,threshold_alg):
    """
    convert an image from RGB to "gray" scale using pca
    :param img:
    :return:
    """
    pca = PCA(n_components=1)
    s = img.shape
    flatten_table = np.reshape(img,(s[0]*s[1],3))

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    X_r = pca.fit_transform(flatten_table)

    pca_img = np.reshape(X_r,s[:2])
    # print(pca.explained_variance_ratio_)

    normalized_image = np.uint8(cv2.normalize(pca_img,pca_img,0,255,cv2.NORM_MINMAX))
    # threshold_global_otsu,threshold_value = threshold_otsu(normalized_image)
    # print(threshold_global_otsu)
    # flip if necessary so that black is text/grid lines
    if np.mean(X_r) < 125:
        print("inverted")
        normalized_image = 255 - normalized_image


        threshed_image,threshold_value = threshold_alg(normalized_image,True)
        threshold_value = 255 - threshold_value
        inverted = True

    else:
        print("not inverted")
        inverted = False
        threshed_image,threshold_value = threshold_alg(normalized_image,False)
    #     ret2,threshed_image = cv2.threshold(res,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     # ret,threshed_image = cv2.threshold(res,90,255,cv2.THRESH_BINARY)
    #
    #     threshed_image = 255 - threshed_image
    #     inverted = True
    #
    #     if display:
    #         plt.imshow(threshed_image,cmap="gray")
    #         plt.show()
    #
    #         ret,threshed_image = cv2.threshold(res,255-200,255,cv2.THRESH_BINARY)
    #         threshed_image = 255 - threshed_image
    #         plt.imshow(threshed_image,cmap="gray")
    #         plt.show()
    # else:
    #     print("not inverted")
    #     # ret2,th2 = cv2.threshold(res,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     ret,threshed_image = cv2.threshold(res,threshold_value,255,cv2.THRESH_BINARY)
    #
    #     inverted = False


    return threshed_image,threshold_value,inverted


def __dbscan_threshold__(img):
    ink_pixels = np.where(img>0)
    X = np.asarray(zip(ink_pixels[1],ink_pixels[0]))
    print("doing dbscan: " + str(X.shape))
    db = DBSCAN(eps=1, min_samples=5).fit(X)

    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    return_image = np.zeros(gray.shape,np.uint8)
    return_image.fill(255)

    print("going through dbscan results")
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            continue

        class_member_mask = (labels == k)
        # temp = np.zeros(X.shape)

        xy = X[class_member_mask]

        max_value = gray[xy[:, 1], xy[:, 0]].max()
        median = np.median(gray[xy[:, 1], xy[:, 0]])
        mean = np.mean(gray[xy[:, 1], xy[:, 0]])
        # print(max_value,median,mean)

        if True:#median > 120:
            x_max,y_max = np.max(xy,axis=0)
            x_min,y_min = np.min(xy,axis=0)
            if min(x_max-x_min,y_max-y_min) >= 10:
                return_image[xy[:, 1], xy[:, 0]] = gray[xy[:, 1], xy[:, 0]]

def __create_mask__(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    horizontal_lines = paper_quad.__extract_grids__(gray,True)

    mask = np.zeros(gray.shape,np.uint8)

    for l in horizontal_lines:
        #     corrected_l = __identity__(gray,l)
        corrected_l = __non_extreme_regression__(gray,l,True)
        # corrected_l = __correct__(gray,l,True)
        mask = np.max([mask,corrected_l],axis=0)

    vertical_lines = paper_quad.__extract_grids__(gray,False)

    for l in vertical_lines:
        # corrected_l = __identity__(gray,l)
        corrected_l = __non_extreme_regression__(gray,l,False)
        # corrected_l = __correct__(gray,l,False)
        mask = np.max([mask,corrected_l],axis=0)

    return mask



def __mask_lines__(img,mask):

    # thresh1 = __threshold_image__(gray)
    masked_image = np.max([img,mask],axis=0)

    # cv2.imwrite("/home/ggdhines/2.jpg",masked_image)
    # assert False

    return masked_image

def __gen_columns__(masked_image,gray):
    vertical_contours = paper_quad.__extract_grids__(gray,False)

    vertical_lines = []

    for v in vertical_contours:
        x_val,_ = np.mean(v,axis=0)
        vertical_lines.append(x_val)

    vertical_lines.extend([0,img.shape[1]])
    vertical_lines.sort()
    for column_index in range(len(vertical_lines)-1):
        sub_image = masked_image[:,vertical_lines[column_index]:vertical_lines[column_index+1]]

        colour_sub_image = np.zeros([sub_image.shape[0],sub_image.shape[1],3],np.uint8)
        colour_sub_image[:,:,0] = sub_image
        colour_sub_image[:,:,1] = sub_image
        colour_sub_image[:,:,2] = sub_image
        yield colour_sub_image

def __ocr_image__(image):
    tess = tesserpy.Tesseract("/home/ggdhines/github/tessdata/",language="eng")



    # print(tess_directory,language)
    # tess = tesserpy.Tesseract(tess_directory,language=language)
    # print(vars(tess))
    tess.tessedit_pageseg_mode = tesserpy.PSM_SINGLE_BLOCK
    # tess.tessedit_ocr_engine_mode = tesserpy.OEM_TESSERACT_CUBE_COMBINED
    # tess.tessedit_pageseg_mode = tesserpy.PSM_SINGLE_WORD
    # tess.tessedit_ocr_engine_mode = tesserpy.OEM_CUBE_ONLY
    # tess.tessedit_page_iteratorlevel = tess.RIL_SYMBOL
    tess.tessedit_char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.abcdefghijkmnopqrstuvwxyz-"

    tess.set_image(image)
    tess.get_utf8_text()
    transcribed = []

    # image_height,image_width = image.shape[:2]
    temp_image = np.zeros((image.shape[0],image.shape[1],3),np.uint8)
    temp_image[:,:,0] = image
    temp_image[:,:,1] = image
    temp_image[:,:,2] = image
    # print(image.shape)
    # cv2.imwrite("/home/ggdhines/tmp.jpg",temp_image)
    # assert False
    # plt.imshow(image,cmap="gray")

    for word in tess.symbols():
    # for word in tess.words():
    #     print(word.text,word.confidence)
        bb = word.bounding_box
        height = abs(bb.top-bb.bottom)
        width = bb.right - bb.left

        if max(height,width) > 7:
            # print(word.text,height,width)
            transcribed.append((word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
            # if word.text == "M":
            #     print(word.text,word.confidence)
            #     plt.imshow(image,cmap="gray")
            # plt.plot([bb.left,bb.right,bb.right,bb.left,bb.left],[bb.top-1,bb.top-1,bb.bottom-1,bb.bottom-1,bb.top-1],color="blue")
            # plt.show()
            # print("{}\t{}\tt:{}; l:{}; r:{}; b:{}".format(word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
        else:
            pass
            # print(word.text,height,width,word.confidence)
        # confidences.append(word.confidence)
        # text.append(word.text)
        # boxes.append(word.bounding_box)
    # plt.show()
    return transcribed

def __cell_boundaries__(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    horizontal_grid = []
    vertical_grid = []

    height,width = image.shape[:2]

    horizontal_lines = paper_quad.__extract_grids__(gray,True)

    # plt.imshow(image)
    for row_index in range(len(horizontal_lines)):
        c = np.median(horizontal_lines[row_index],axis=0)[1]
        # lb = np.max(horizontal_lines[row_index+1],axis=0)[1]

        # if horizontal_grid
        horizontal_grid.append(c)

        # plt.plot((0,width),(lb,lb))
        # plt.plot((0,width),(ub,ub))

    horizontal_grid.extend([0,height])
    horizontal_grid.sort()

    vertical_lines = paper_quad.__extract_grids__(gray,False)
    for column_index in range(len(vertical_lines)):
        c = np.median(vertical_lines[column_index],axis=0)[0]
        # ub = np.max(vertical_lines[column_index+1],axis=0)[0]

        vertical_grid.append(c)

        # plt.plot((lb,lb),(0,height))
        # plt.plot((ub,ub),(0,height))
    vertical_grid.extend([0,width])
    vertical_grid.sort()


    return horizontal_grid,vertical_grid


def __place_in_cell__(transcriptions,horizontal_grid,vertical_grid,id_):
    """
    :param transcriptions:
    :param image:
    :param id_:
    :param save_to_db: do we actually want to save data to the database?
    :return:
    """
    # horizontal_grid,vertical_grid = __cell_boundaries__(image)
    cell_contents = {}

    for (t,c,top,left,right,bottom) in transcriptions:
        if t == None:
            continue



        # print(t,c)
        mid_y = (int(top)+int(bottom))/2.
        mid_x = (int(right)+int(left))/2.

        in_row = False

        for row_index in range(len(horizontal_grid)-1):
            lb = horizontal_grid[row_index]
            ub = horizontal_grid[row_index+1]
            if lb <= mid_y <= ub:
                in_row = True
                break

        assert in_row



        # assert in_row

        in_column = False
        for column_index in range(len(vertical_grid)-1):
            lb = vertical_grid[column_index]
            ub = vertical_grid[column_index+1]
            if lb <= mid_x <= ub:
                in_column = True
                break
        assert in_column


        key = (row_index,column_index)
        if key not in cell_contents:
            cell_contents[key] = [(mid_x,t,c,(top,left,right,bottom))]
        else:
            cell_contents[key].append((mid_x,t,c,(top,left,right,bottom)))

    confidence_array = []

    problems = 0

    # print("*******")
    cur = con.cursor()
    for key in cell_contents:
        sorted_contents = sorted(cell_contents[key], key = lambda x:x[0])
        _,text,confidences,coordinates = zip(*sorted_contents)
        text = "".join(text)
        confidence = min(confidences)
        cell_contents[key] = (text,confidence)
        confidence_array.append(confidence)

        if confidence < 80:
            problems += 1

        if id_ is None:
            continue
        stmt = "insert into transcriptions values(\""+id_+"\",0,"+str(key[1])+","+str(key[0])+",\""+text+"\","+str(confidence)+")"
        cur.execute(stmt)

        # and now save the actual characters
        for char,cnf,crd in zip(text,confidences,coordinates):
            # cur.execute("create table characters(subject_id text, region int, column int, row int, characters text, confidence float,lb_x int,ub_x int, lb_y int,ub_y)")
            stmt = "insert into characters values(\""+id_+"\",0,"+str(key[1])+","+str(key[0])+",\""+char+"\","+str(cnf)+","+str(crd[1])+","+str(crd[2])+","+str(crd[0])+","+str(crd[3])+")"
            cur.execute(stmt)


    if True:#id_ is not None:
        print("problems " + str(problems))

    return cell_contents, confidence_array,problems

def __gold_standard_comparison__(transcriptions):
    total = 0
    correct_empty = 0
    empty = 0

    true_positives = []
    false_positives = []
    correct_by_column = {}
    with open("/home/ggdhines/gold_standard.txt","rb") as f:
        reader = csv.reader(f, delimiter=',')
        for row,column,gold_standard in reader:

            key = (int(row),int(column))

            if gold_standard == "":
                empty += 1
                if  key not in transcriptions:
                    correct_empty += 1
            else:
                if key in transcriptions:
                    t,c = transcriptions[key]

                    if gold_standard == t:
                        total += 1
                        true_positives.append(c)

                        if column not in correct_by_column:
                            correct_by_column[column] = 1
                        else:
                            correct_by_column[column] += 1
                    else:
                        false_positives.append(c)

    print("summary stats")
    print(total)
    print(correct_empty)
    # print(empty)
    print("")
    print([(i,correct_by_column[i]) for i in sorted(correct_by_column.keys())])


    return true_positives,false_positives

def __roc_plot__(true_positives,false_positives):
    alpha_list = true_positives[:]
    alpha_list.extend(false_positives)
    alpha_list.sort()

    roc_X = [0,1]
    roc_Y = [0,0]

    for alpha in alpha_list:
        positive_count = sum([1 for x in true_positives if x >= alpha])
        positive_rate = positive_count/float(len(true_positives))

        negative_count = sum([1 for x in false_positives if x >= alpha])
        negative_rate = negative_count/float(len(false_positives))

        if (len(roc_X) > 0) and (positive_rate == roc_Y[-1]) and (negative_rate == roc_X[-1]):
            continue

        else:
            roc_X.append(negative_rate)
            roc_Y.append(positive_rate)


    # roc_X = list(reversed(roc_X))
    # roc_Y = list(reversed(roc_Y))

    p = Polygon(zip(roc_X,roc_Y))
    print(p.area)
    print(len(false_positives))

    # plt.plot(roc_X,roc_Y)
    # plt.xlabel("% False Positives")
    # plt.ylabel("% True Positives")
    # plt.ylim((0,1.01))
    # plt.ylim((0,1.01))
    # plt.show()

    # fig = plt.figure(1, figsize=(5,5), dpi=90)
    # ax = fig.add_subplot(111)
    # ring_patch = PolygonPatch(p)
    # ax.add_patch(ring_patch)
    # plt.show()

def __extract_region__(fname,region_id = 0):
    regions =  [(559,3282,1276,2097)]

    img = cv2.imread(fname)
    region = regions[region_id]
    sub_image = img[region[2]:region[3],region[0]:region[1]]

    return sub_image


def __run__(img,mask,horizontal_grid,vertical_grid,threshold_alg,id_=None):
    pca_image,threshold,inverted = __pca__(img,threshold_alg)


    masked_image = __mask_lines__(pca_image,mask)




    if id_ is not None:
        cv2.imwrite("/home/ggdhines/to_upload"+db_id+"/"+id_+".jpg",masked_image)

    transcriptions = __ocr_image__(masked_image)
    _,confidence_values,problems = __place_in_cell__(transcriptions,horizontal_grid,vertical_grid,id_)

    return np.mean(confidence_values),threshold,inverted


def round_ten(x):
    # http://stackoverflow.com/questions/26454649/python-round-up-to-the-nearest-ten
    return int(round(x / 10.0)) * 10

if __name__ == "__main__":
    # cur.execute("create table transcriptions(subject_id text, region int, column int, row int, contents text, confidence float)")
    # cur.execute("create table characters(subject_id text, region int, column int, row int, characters text, confidence float,lb_x int,ub_x int, lb_y int,ub_y int)")


    # db_id = "5"

    tess_directory, language, db_id = "/home/ggdhines/github/tessdata/", "eng", "3"
    tess_directory, language, db_id = "/tmp/tessdata/", "active_weather", "4"

    con = lite.connect('/home/ggdhines/to_upload' + db_id + '/active.db')
    cur = con.cursor()

    cur.execute(
        "create table transcriptions(subject_id text, region int, column int, row int, contents text, confidence float)")
    cur.execute(
        "create table characters(subject_id text, region int, column int, row int, characters text, confidence float,lb_x int,ub_x int, lb_y int,ub_y int)")
    # con.commit()


    for fname in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:40]:
        # fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0329.JPG"
        img = __extract_region__(fname)
        id_ = fname.split("/")[-1][:-4]
        print(id_)


        # set a baseline for performance with otsu's binarization
        mask = __create_mask__(img)
        horizontal_grid,vertical_grid = __cell_boundaries__(img)


        otsu_peformance,otsu_threshold,inverted = __run__(img,mask,horizontal_grid,vertical_grid,__otsu_bin__,None)
        print(otsu_threshold,otsu_peformance)



        # # if the performance is good enough - just go with it
        if otsu_peformance >= 80:
            __run__(img,mask,horizontal_grid,vertical_grid,__otsu_bin__,id_)
        # otherwise, use binary thresholding and search for a good threshold value
        else:
            print("searching")
            print("******")
            best_threshold = otsu_threshold
            max_confidence = otsu_peformance

            previous_confidence = otsu_peformance

            step = 2
            if inverted:
                lb = round_ten(otsu_threshold-20)
                search_range = range(int(round(otsu_threshold))-step,lb,-step)
            else:
                ub = round_ten(otsu_threshold+20)
                search_range = range(int(round(otsu_threshold))+step,ub,step)

            for bin_threshold in search_range:
                # print("here " + str(bin_threshold))

                thres_alg = __binary_threshold_curry__(bin_threshold)
                try:
                    confidence,_,_ = __run__(img,mask,horizontal_grid,vertical_grid,thres_alg,None)
                except RuntimeWarning:
                    continue

                print(bin_threshold,confidence)
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_threshold = bin_threshold


                previous_confidence = confidence


            assert best_threshold is not None
            thres_alg = __binary_threshold_curry__(best_threshold)
            __run__(img,mask,horizontal_grid,vertical_grid,thres_alg,id_)


    con.commit()
    con.close()
