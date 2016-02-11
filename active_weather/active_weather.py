import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import pytesseract
import Image
import tesserpy
tess = tesserpy.Tesseract("/home/ggdhines/github/tessdata/", language="eng")
tess.tessedit_char_whitelist = """'"!@#$%^&*()_+-=[]{};,.<>/?`~abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"""

horizontal = []

reference_image = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0019.JPG")
refer_shape = reference_image.shape

def __get_lines__(horizontal):
    # todo - swap
    lined_images = []

    for f in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:5]:
        img = cv2.imread(f,0)

        if horizontal:
            dy = cv2.Sobel(img,cv2.CV_16S,0,2)
            dy = cv2.convertScaleAbs(dy)
            cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
            ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        else:
            dx = cv2.Sobel(img,cv2.CV_16S,2,0)
            dx = cv2.convertScaleAbs(dx)
            cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
            ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

        close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,kernel)

        lined_images.append(close)

    average_image = np.percentile(lined_images,40,axis=0)

    average_image = average_image.astype(np.uint8)

    cv2.imwrite("/home/ggdhines/results.jpg",average_image)

    contours_to_return = []

    _,contour, hier = cv2.findContours(average_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if (horizontal and w/h > 5) or ((not horizontal) and h/w > 5):
            cv2.drawContours(average_image,[cnt],0,255,-1)
            contours_to_return.append(cnt)
        else:
            cv2.drawContours(average_image,[cnt],0,0,-1)

    average_image = cv2.morphologyEx(average_image,cv2.MORPH_DILATE,None,iterations = 2)

    return average_image,contours_to_return


def __get_grid__():

    horizontal_image,horizontal_contours = __get_lines__(True)
    cv2.imwrite("/home/ggdhines/horizontal.jpg",horizontal_image)

    horizontal_lines = []
    vertical_lines = []

    a = np.zeros(refer_shape,np.uint8)
    delta = 50

    for cnt in horizontal_contours:
        shape = cnt.shape
        t = cnt.reshape((shape[0],shape[2]))
        max_x,max_y = np.max(t,axis=0)
        min_x,min_y = np.min(t,axis=0)

        if (min_y>=1276-delta) and (max_y<=2097+delta):
            perimeter = cv2.arcLength(cnt,True)

            if perimeter > 100:
                cv2.drawContours(a,[cnt],0,255,3)
                horizontal_lines.append(cnt)

    horizontal_lines.sort(key = lambda l:l[0][0][1])

    vertical_image,vertical_contours = __get_lines__(False)

    delta = 400
    for cnt in vertical_contours:
        shape = cnt.shape
        t = cnt.reshape((shape[0],shape[2]))
        max_x,max_y = np.max(t,axis=0)
        min_x,min_y = np.min(t,axis=0)

        interior_line = (min_x >= 563-100) and (max_x <= 3282+100)and(min_y>=1276-delta) and (max_y<=2097+delta)
        through_line = (min_x >= 563-100) and (max_x <= 3282+100) and (min_y < 1276) and(max_y > 2097)

        if interior_line or through_line:

            perimeter = cv2.arcLength(cnt,True)
            if perimeter > 1000:
                print min_y,max_y
                # cv2.drawContours(masks,[cnt],0,255,3)
                vertical_lines.append(cnt)
                cv2.drawContours(a,[cnt],0,255,3)

    vertical_lines.sort(key = lambda l:l[0][0][0])

    # todo - deal with lines that go all the way through the window
    # cv2.imwrite("/home/ggdhines/testtest.jpg",a)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',a)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return horizontal_lines,vertical_lines

horizontal_grid,vertical_grid = __get_grid__()

image = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG",0)
# image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

print len(horizontal_grid)
print len(vertical_grid)

total = 0.
confident = 0

for h_index in range(len(horizontal_grid)-1):
    # todo - resizing might be not be necessary
    # todo - DEFINITELY precalculate these
    shape = horizontal_grid[h_index].shape
    t = horizontal_grid[h_index].reshape((shape[0],shape[2]))
    _,min_y = np.max(t,axis=0)

    shape = horizontal_grid[h_index+1].shape
    t = horizontal_grid[h_index+1].reshape((shape[0],shape[2]))
    _,max_y = np.min(t,axis=0)

    for v_index in range(len(vertical_grid)-1):
        shape = vertical_grid[v_index].shape
        t = vertical_grid[v_index].reshape((shape[0],shape[2]))
        min_x,_ = np.max(t,axis=0)

        shape = vertical_grid[v_index+1].shape
        t = vertical_grid[v_index+1].reshape((shape[0],shape[2]))
        max_x,_ = np.min(t,axis=0)


        mask = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
        cv2.drawContours(mask,horizontal_grid,h_index,255,-1)
        cv2.drawContours(mask,horizontal_grid,h_index+1,255,-1)
        cv2.drawContours(mask,vertical_grid,v_index,255,-1)
        cv2.drawContours(mask,vertical_grid,v_index+1,255,-1)

        _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # contours are probably in sorted order but just to be sure
        for c,h in zip(contours,hier[0]):
            if h[-1] != -1:
                continue

            cv2.drawContours(mask,[c],0,255,-1)

        res = cv2.bitwise_and(mask,image)

        cv2.imwrite("/home/ggdhines/intersection.jpg",res)

        # now go back over and draw in contours in white - any black inside the cell that is on a grid
        # line is mostly likely grid, not ink
        # most of the time this wouldn't make a difference - but be sure
        # todo - double check
        cv2.drawContours(res,horizontal_grid,h_index,255,-1)
        cv2.drawContours(res,horizontal_grid,h_index+1,255,-1)
        cv2.drawContours(res,vertical_grid,v_index,255,-1)
        cv2.drawContours(res,vertical_grid,v_index+1,255,-1)

        res2 = res[min_y:max_y+1,min_x:max_x+1]

        cv2.imwrite("/home/ggdhines/cell.jpg",res2)

        res3 = np.zeros((res2.shape[0],res2.shape[1],3),np.uint8)
        res3[:,:,0] = res2[:,:]
        res3[:,:,1] = res2[:,:]
        res3[:,:,2] = res2[:,:]

        # print res2.shape
        # print res2
        image_ = cv2.imread("/home/ggdhines/cell.jpg")
        # print image_.shape
        # print image_
        tess.set_image(res3)
        tess.get_utf8_text()
        found = False

        words = list(tess.words())
        words_in_cell = [w.text for w in words if w.text is not None]
        conf_in_cell = [w.confidence for w in words if w.text is not None]

        word = "".join(words_in_cell)

        if word == "":
            print "%8s" % "",
        else:
            overall_confidence = np.mean(conf_in_cell)
            total += 1

            if overall_confidence >= 80:
                print "%8s" % word,
            else:
                print "%8s" % "",
                confident += 1



        # for word in tess.words():
        #     t = word.text
        #
        #     conf = word.confidence
        #
        #
        #
        #     if (t is None) or (conf < 80):
        #         # continue
        #         print '%8s' % ""
        #         break
        #
        #     else:
        #         t = t.strip()
        #         print '%8s' % t
        #
        #         if t == "":
        #             continue
        #
        #         print t,
        #         for _ in range(7-len(t)):
        #             print " ",
            # print "|"
                # print t,word.confidence
                # raw_input("enter something and check file")
            # print word.confidence
    print
    # assert False
print confident, total
assert False


# _,image = cv2.threshold(image,160,255,cv2.THRESH_BINARY)
#
# for m in masks:
#
#     shape = m.shape
#     t = m.reshape((shape[0],shape[2]))
#     max_x,max_y = np.max(t,axis=0)
#     min_x,min_y = np.min(t,axis=0)
#
#     template = np.zeros(image.shape,np.uint8)
#     cv2.drawContours(template,[m],0,255,-1)
#
#     res = cv2.bitwise_and(template,image)
#
#     cv2.imwrite("/home/ggdhines/results.jpg",res)
#
#     res2 = res[min_y:max_y+1,min_x:max_x+1]
#     cv2.imwrite("/home/ggdhines/results2.png",res2)
#
#     print pytesseract.image_to_string(Image.open("/home/ggdhines/results2.png"))
#     break


