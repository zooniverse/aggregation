from __future__ import print_function
import numpy as np
import cv2
import csv
import paper_quad
import math
import tesserpy
# from scipy.integrate import simps
# import shapely
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from sklearn import mixture
import sqlite3 as lite

con = lite.connect('/home/ggdhines/active.db')


# cur.execute("create table transcriptions(subject_id text, region int, column int, row int, contents text, confidence float)")




__author__ = 'ggdhines'

def __upper_bounds__(line):
    sorted_l = sorted(line, key = lambda l:l[0])

    x_ret = []
    y_ret = []

    current_x = sorted_l[0][0]
    current_y = -float("inf")

    for x,y in sorted_l:
        if x != current_x:
            x_ret.append(current_x)
            y_ret.append(current_y)
            current_x = x
            current_y = -float("inf")
        current_y = max(current_y,y)

    x_ret.append(current_x)
    y_ret.append(current_y)

    return x_ret,y_ret

def __lower_bounds__(line):
    sorted_l = sorted(line, key = lambda l:l[0])

    x_ret = []
    y_ret = []

    current_x = sorted_l[0][0]
    current_y = float("inf")

    for x,y in sorted_l:
        if x != current_x:
            x_ret.append(current_x)
            y_ret.append(current_y)
            current_x = x
            current_y = float("inf")
        current_y = min(current_y,y)

    x_ret.append(current_x)
    y_ret.append(current_y)

    return x_ret,y_ret

def __median_kernel__(img,line,horizontal):
    if horizontal:
        x,y = __upper_bounds__(line)
        x2,y2 = __lower_bounds__(line)
    else:
        x,y = zip(*line)
        line = zip(y,x)
        x,y = __upper_bounds__(line)
        x2,y2 = __lower_bounds__(line)

    x2 = list(reversed(x2))
    y2 = list(reversed(y2))


    x_pts = x
    x_pts.extend(x2)

    y_pts = []
    step = 150
    for y_index in range(len(y)):
        m = np.median(y[max(y_index-step,0):y_index+step])
        assert not math.isnan(m)

        y_pts.append(int(round(m)))

    for y_index in range(len(y2)):
        m = np.median(y2[max(y_index-step,0):y_index+step])
        assert not math.isnan(m)

        y_pts.append(int(round(m)))

    mask = np.zeros(img.shape,np.uint8)

    # print zip(x,y)

    if horizontal:
        pts = np.asarray(zip(x_pts,y_pts))
    else:
        pts = np.asarray(zip(y_pts,x_pts))

    cv2.drawContours(mask,[pts],0,255,-1)
    # cv2.imshow("img",mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # assert False

    return mask

def __identity__(img,line):
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask,[line],0,255,-1)

    return mask

def __polynomial_correct__(img,line,horizontal):
    if horizontal:
        x,y = zip(*line)
        domain = sorted(set(line[:,0]))
    else:
        y,x = zip(*line)
        domain = sorted(set(line[:,1]))

    init_mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(init_mask,[line],0,255,-1)
    # plt.imshow(init_mask)
    # plt.show()

    mask2 = np.zeros(img.shape[:2],np.uint8)

    degrees = 2
    coeff = list(reversed(np.polyfit(x,y,degrees)))
    y_bar = [sum([coeff[p]*x_**p for p in range(degrees+1)]) for x_ in x]

    # degrees = 2
    coeff = list(reversed(np.polyfit(x,y,degrees)))
    std = math.sqrt(np.mean([(y1-y2)**2 for (y1,y2) in zip(y,y_bar)]))

    def y_bar(x_,upper):
        return int(sum([coeff[p]*x_**p for p in range(degrees+1)]) + upper*std)


    y_vals = [y_bar(x,-1) for x in domain]
    y_vals.extend([y_bar(x,1) for x in list(reversed(domain))])
    x_vals = list(domain)
    x_vals.extend(list(reversed(domain)))

    if horizontal:
        pts = np.asarray(zip(x_vals,y_vals))
    else:
        pts = np.asarray(zip(y_vals,x_vals))


    cv2.drawContours(mask2,[pts],0,255,-1)

    # plt.imshow(mask2)
    # plt.show()

    mask3 = np.min([init_mask,mask2],axis=0)
    # plt.imshow(mask3)
    # plt.show()

    # x,y = zip(*line)
    # plt.plot(x,y)
    # plt.plot(x_vals,y_vals)
    # plt.show()

    return mask3

def __correct__(img,line,horizontal,background=0,foreground=255):
    assert len(img.shape) == 2
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask,[line],0,255,-1)


    mask2 = __median_kernel__(img,line,horizontal)

    # plt.imshow(mask)
    # plt.show(0)

    overlap = np.min([mask,mask2],axis=0)

    # plt.imshow(overlap)
    # plt.show(0)

    return overlap

def __pca__(img):
    pca = PCA(n_components=1)
    s = img.shape
    flatten_table = np.reshape(img,(s[0]*s[1],3))

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    X_r = pca.fit_transform(flatten_table)
    # background = max(X_r)[0]
    # foreground = 0

    pca_img = np.reshape(X_r,s[:2])
    print(pca.explained_variance_ratio_)

    res = np.uint8(cv2.normalize(pca_img,pca_img,0,255,cv2.NORM_MINMAX))

    # plt.imshow(res)
    # plt.show()

    ink_pixels = np.where(res>90)
    template = np.zeros(img.shape[:2],np.uint8)
    # plt.plot(ink_pixels[1],-ink_pixels[0],".")
    # plt.show()

    template[ink_pixels[0],ink_pixels[1]] = gray[ink_pixels[0],ink_pixels[1]]
    # plt.imshow(template)
    # plt.show()

    # assert False
    template = 255 - template
    return template

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

# def __pca_mask__(img):
#     assert len(img.shape) ==3
#
#     pca_image = __pca__(img)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#     horizontal_lines = paper_quad.__extract_grids__(gray,True)
#
#     mask = np.zeros(gray.shape,np.uint8)
#     for l in horizontal_lines:
#         corrected_l = __polynomial_correct__(gray,l,True)
#         # corrected_l = 255 - corrected_l
#         # corrected_l = __correct__(gray,l,True)
#         pca_image = np.min([pca_image,corrected_l],axis=0)
#
#         # plt.imshow(pca_image,cmap="gray")
#         # plt.show()
#
#     vertical_lines = paper_quad.__extract_grids__(gray,False)
#     for l in vertical_lines:
#         corrected_l = __polynomial_correct__(gray,l,False)
#         # corrected_l = 255 -corrected_l
#         # corrected_l = __correct__(gray,l,False)
#         pca_image = np.min([pca_image,corrected_l],axis=0)
#
#
#     # masked_image = np.max([pca_image,mask],axis=0)
#
#
#     pca_image = 255 - pca_image
#     plt.imshow(pca_image,cmap="gray")
#     plt.show()
#
#
#     # ink_pixels = np.where(masked_image>2)
#     # # template = np.zeros(img.shape[:2],np.uint8)
#     # plt.plot(ink_pixels[1],-ink_pixels[0],".")
#     # plt.show()
#     return pca_image


def __mask_lines__(gray):
    # img = cv2.imread('/home/ggdhines/region.jpg')
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    horizontal_lines = paper_quad.__extract_grids__(gray,True)

    mask = np.zeros(gray.shape,np.uint8)
    for l in horizontal_lines:
        corrected_l = __identity__(gray,l)
        # corrected_l = __polynomial_correct__(gray,l,True)
        corrected_l = __correct__(gray,l,True)
        mask = np.max([mask,corrected_l],axis=0)

    cv2.imwrite("/home/ggdhines/testing.jpg",mask)
    vertical_lines = paper_quad.__extract_grids__(gray,False)
    for l in vertical_lines:
        corrected_l = __identity__(gray,l)
        # corrected_l = __polynomial_correct__(gray,l,False)
        corrected_l = __correct__(gray,l,False)
        mask = np.max([mask,corrected_l],axis=0)

    # cv2.imshow("img",mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    thresh1 = __threshold_image__(gray)
    masked_image = np.max([thresh1,mask],axis=0)

    plt.imshow(masked_image,cmap="gray")
    plt.title("masked image")
    plt.show()
    # cv2.imwrite("/home/ggdhines/2.jpg",masked_image)
    # assert False

    return masked_image

def __threshold_image__(img):
    assert len(img.shape) == 2
    # for i in img:
    #     print(list(i))
    ret,thresh1 = cv2.threshold(img,190,255,cv2.THRESH_BINARY)

    thresh3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,251,2)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return thresh3


    radius = 200
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)

    # print(img >= local_otsu)
    thresh2 = np.zeros(img.shape[:2],np.uint8)
    thresh2.fill(255)
    x,y = np.where(img < local_otsu)
    thresh2[x,y] = 0

    # plt.imshow(thresh1,cmap="gray")
    # plt.show()
    # plt.imshow(thresh2,cmap="gray")
    # plt.show()

    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu
    # print(img)
    # print(global_otsu)
    # print(threshold_global_otsu)
    thresh3 = np.zeros(img.shape[:2],np.uint8)
    x,y = np.where(img < threshold_global_otsu)
    thresh3.fill(255)
    # print(np.where(img< global_otsu))
    # for i in img:
    #     print(min(i))
    # plt.plot(x,y,".")
    # plt.show()
    thresh3[x,y] = 0

    return thresh3


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
    # print(vars(tess))
    tess.tessedit_pageseg_mode = tesserpy.PSM_SINGLE_BLOCK
    # tess.tessedit_ocr_engine_mode = tesserpy.OEM_CUBE_ONLY
    # tess.tessedit_page_iteratorlevel = tess.RIL_SYMBOL
    tess.tessedit_char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.abcdefghijkmnopqrstuvwxyz"

    tess.set_image(image)
    tess.get_utf8_text()
    transcribed = []

    # image_height,image_width = image.shape[:2]

    for word in tess.symbols():
    # for word in tess.words():
    #     print(word.text,word.confidence)
        bb = word.bounding_box
        height = abs(bb.top-bb.bottom)
        width = bb.right - bb.left

        if min(height,width) >= 10:

            transcribed.append((word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
            # print("{}\t{}\tt:{}; l:{}; r:{}; b:{}".format(word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
        # confidences.append(word.confidence)
        # text.append(word.text)
        # boxes.append(word.bounding_box)
    return transcribed

def __cell_boundaries__(image):
    horizontal_grid = []
    vertical_grid = []

    height,width = image.shape[:2]

    horizontal_lines = paper_quad.__extract_grids__(image,True)

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

    vertical_lines = paper_quad.__extract_grids__(image,False)
    for column_index in range(len(vertical_lines)):
        c = np.median(vertical_lines[column_index],axis=0)[0]
        # ub = np.max(vertical_lines[column_index+1],axis=0)[0]

        vertical_grid.append(c)

        # plt.plot((lb,lb),(0,height))
        # plt.plot((ub,ub),(0,height))
    vertical_grid.extend([0,width])
    vertical_grid.sort()


    return horizontal_grid,vertical_grid

def __place_in_cell__(transcriptions,image,id_):
    horizontal_grid,vertical_grid = __cell_boundaries__(image)
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

        if not in_row:
            plt.imshow(image,cmap="gray")
            # plt.plot(mid_x,mid_y,"o")
            print(mid_x,mid_y)
            print((t,c,top,left,right,bottom))
            plt.show()
            assert False
            continue

        # assert in_row

        in_column = False
        for column_index in range(len(vertical_grid)-1):
            lb = vertical_grid[column_index]
            ub = vertical_grid[column_index+1]
            if lb <= mid_x <= ub:
                in_column = True
                break
        if not in_column:
            plt.imshow(image,cmap="gray")
            # plt.plot(mid_x,mid_y,"o")
            print(mid_x,mid_y)
            print((t,c,top,left,right,bottom))
            plt.show()
            assert False
            continue

        # plt.imshow(image,cmap="gray")
        # plt.plot(mid_x,mid_y,"o")
        # plt.show()

        key = (row_index,column_index)
        if key not in cell_contents:
            cell_contents[key] = [(mid_x,t,c)]
        else:
            cell_contents[key].append((mid_x,t,c))

    confidence_array = []

    problems = 0
    print("*******")
    cur = con.cursor()
    for key in cell_contents:
        sorted_contents = sorted(cell_contents[key], key = lambda x:x[0])
        _,text,confidence = zip(*sorted_contents)
        text = "".join(text)
        confidence = min(confidence)
        cell_contents[key] = (text,confidence)
        confidence_array.append(confidence)

        stmt = "insert into transcriptions values(\""+id_+"\",0,"+str(key[1])+","+str(key[0])+",\""+text+"\","+str(confidence)+")"
        print(stmt)
        cur.execute(stmt)

        if confidence < 80:
            print(text)
            problems += 1
    print("problems " + str(problems))

    con.commit()
    # con.close()

    plt.hist(confidence_array, bins=20, normed=1, histtype='step', cumulative=1)
    plt.show()


    return cell_contents

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

    plt.plot(roc_X,roc_Y)
    plt.xlabel("% False Positives")
    plt.ylabel("% True Positives")
    plt.ylim((0,1.01))
    plt.show()

    # fig = plt.figure(1, figsize=(5,5), dpi=90)
    # ax = fig.add_subplot(111)
    # ring_patch = PolygonPatch(p)
    # ax.add_patch(ring_patch)
    # plt.show()

def __gmm__(img):
    x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = np.reshape(x,x.shape[0]*x.shape[1])

    for i in range(2,20):
        g = mixture.GMM(n_components=i)

        g.fit(x)

        print(g.aic(x),g.means_,g.weights_)

if __name__ == "__main__":
    img = cv2.imread('/home/ggdhines/region.jpg')

    # gray = __pca_mask__(img)
    #
    # img = cv2.GaussianBlur(img,(5,5),0)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # # mask = np.zeros((gray.shape),np.uint8)
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #
    # close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    # div = np.float32(gray)/(close)
    # res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    # res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

    # plt.imshow(res2)
    # plt.show()
    # assert False

    # x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # x = np.reshape(x,x.shape[0]*x.shape[1])
    # n, bins, patches = plt.hist(x, 20, normed=1, facecolor='green', alpha=0.5)
    # plt.show()

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret,thresh1 = cv2.threshold(gray,198,255,cv2.THRESH_BINARY)
    # plt.imshow(thresh1)
    # plt.show()
    # __gmm__(img)
    # assert False

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = __pca__(img)


    # assert False

    masked_image = __mask_lines__(gray)
    # masked_image = __pca_mask__(img)

    transcriptions = __ocr_image__(masked_image)
    transcriptions_in_cells = __place_in_cell__(transcriptions,gray)

    true_positives,false_positives = __gold_standard_comparison__(transcriptions_in_cells)

    __roc_plot__(true_positives,false_positives)





    con.commit()
    con.close()