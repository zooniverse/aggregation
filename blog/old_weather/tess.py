from subprocess import call
import collections
import xmltodict

# call(["tesseract","/home/ggdhines/Downloads/Hooper_John-Certeine_comfortable_expositions-STC-13743-548_05-p3.tif","/home/ggdhines/tess","hocr"])

def process_page(node):
    print "page length is " + str(len(node))
    for child in node:
        if child.attrib["class"] == "ocr_carea":
            process_carea(child)
        else:
            assert False

    assert False

def process_carea(node):
    paragraphs = []
    for child in node:
        print "a"
        if child.attrib["class"] == "ocr_par":
            r = process_paragraph(child)
            if r is not None:
                paragraphs.append(r)
        else:
            assert False

    # print "****"
    # print paragraphs
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
        assert text != " "
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
    print "^^^^"
    for child in node:
        if child.attrib["class"] == "ocrx_word":

            word = process_word(child)
            if word is not None:
                print "a"
                # print "&&&",word,type(word)

                line.append(word)
        else:
            assert False
    print "done " + str(line)
    # print line
    return line

def process_paragraph(node):
    lines = []
    bounding_boxes = []
    confidence = []
    # print "****"
    for child in node:
        if child.attrib["class"] == "ocr_line":
            for l,b,c in process_line(child):
                # print type(l)
                print l

                if l != []:
                    lines.append(l)
                    bounding_boxes.append(b)
                    confidence.append(c)
        else:
            assert False

    if lines == []:
        print lines
        return lines,bounding_boxes,confidence
    else:
        return None

import xml.etree.ElementTree as ET

tree = ET.parse('/home/ggdhines/tess.hocr')
root = tree.getroot()

# print root[1].tag
for child in root[1]:
    if child.attrib["class"] == "ocr_page":
        ocr = process_page(child)
    else:
        assert False


with open("/home/ggdhines/Documents/gold_standard.txt","rb") as f:
    gold_paragraphs = []
    current_p = ""
    for l in f.readlines():
        l = l.strip()
        if l != "":
            current_p += l + " "
        else:
            gold_paragraphs.append(current_p[:-1])
            current_p = ""

    if current_p != "":
        gold_paragraphs.append(current_p[:-1])

    # print gold_paragraphs
    # print ocr