import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from subprocess import call
import collections
import xmltodict
import matplotlib.cbook as cbook
import cv2

img = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0245.JPG")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,175,255,cv2.THRESH_BINARY_INV)
# cv2.imwrite("/home/ggdhines/greg.jpg",thresh)
# assert False
# call(["tesseract","/home/ggdhines/Downloads/Hooper_John-Certeine_comfortable_expositions-STC-13743-548_05-p3.tif","/home/ggdhines/tess","hocr"])

# image_file = cbook.get_sample_data("/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1941-0813.JPG")
# image_file = cbook.get_sample_data("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0245.JPG")
# image = plt.imread(image_file)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# im = ax.imshow(image)
# plt.imshow(image)
# fig.set_size_inches(52,78)



def process_page(node):
    print "page length is " + str(len(node))
    for child in node:
        if child.attrib["class"] == "ocr_carea":
            process_carea(child)
        else:
            assert False



def process_carea(node):
    paragraphs = []
    for child in node:
        if child.attrib["class"] == "ocr_par":
            r = process_paragraph(child)
            if r is not None:
                paragraphs.append(r)
                for (x0,y0,x1,y1) in r[1]:
                    # ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color="red")
                    cv2.rectangle(img,(x0,y0),(x1,y1),color=(255,0,0),thickness=3)

        else:
            assert False

    # print "****"
    return paragraphs

def process_word(node):
    if len(node) > 0:
        assert len(node) == 1
        child = node[0]

        if "strong" in child.tag:
            text = process_strong(child)
        elif "em" in child.tag:
            text = process_em(child)
        else:
            print child.tag
            assert False
    else:
        text = node.text

    if text != " ":
        title = node.attrib["title"]
        title_params = title.split()
        coords = title_params[1:4]
        coords.append(title_params[4][:-1])
        coords = [int(c) for c in coords]

        confidence = int(title_params[-1])
        if False: #text in [""," ","  ","   "]:
            return None
        else:
            # assert (text != " ") and (text != "") and (text != "  ")
            return text,coords,confidence
    else:
        return None

def process_strong(node):
    if len(node) > 0:
        assert len(node) == 1
        child = node[0]
        assert "em" in child.tag
        return process_em(child)
    else:
        return node.text

def process_em(node):
    return node.text


def process_line(node):
    line = []
    for child in node:
        if child.attrib["class"] == "ocrx_word":

            word = process_word(child)
            if word is not None:
                # print "&&&",word,type(word)

                line.append(word)
        else:
            assert False

    return line

def process_paragraph(node):
    lines = []
    bounding_boxes = []
    confidence = []
    # print "****"
    for child in node:
        if child.attrib["class"] == "ocr_line":
            for l,b,c in process_line(child):

                if l != []:
                    lines.append(l)
                    bounding_boxes.append(b)
                    confidence.append(c)
        else:
            assert False

    if lines != []:

        return lines,bounding_boxes,confidence
    else:
        return None

import xml.etree.ElementTree as ET

tree = ET.parse('/home/ggdhines/tess.hocr')
root = tree.getroot()

# print root[1].tag
for child in root[1]:
    if child.attrib["class"] == "ocr_page":
        process_page(child)
        pass
    else:
        assert False


# with open("/home/ggdhines/Documents/gold_standard.txt","rb") as f:
#     gold_paragraphs = []
#     current_p = ""
#     for l in f.readlines():
#         l = l.strip()
#         if l != "":
#             current_p += l + " "
#         else:
#             gold_paragraphs.append(current_p[:-1])
#             current_p = ""
#
#     if current_p != "":
#         gold_paragraphs.append(current_p[:-1])

    # print gold_paragraphs
    # print ocr
# ax.set_axis_off()
# ax.set_xticks()

# ax.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='off')
#
# ax.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelleft='off')
# plt.savefig("/home/ggdhines/grrr.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
# fig.subplots_adjust(bottom=0, right =1, top=1,left=0,wspace=0,hspace=0)
# fig.tight_layout()
# fig.savefig("/home/ggdhines/old_weather.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
# plt.show()
cv2.imwrite("/home/ggdhines/test.jpg",img)