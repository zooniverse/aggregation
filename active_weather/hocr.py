import xmltodict
import os

def scan():
    os.system("tesseract -psm 7 /home/ggdhines/row.jpg /home/ggdhines/active hocr 2> /dev/null")

    with open('/home/ggdhines/active.hocr',"r") as fd:
        doc = xmltodict.parse(fd.read())

    cells = []
    confidences = []

    try:
        for word in doc["html"]["body"]["div"]["div"]["p"]["span"]["span"]:
            title = word["@title"]
            _,bb1,bb2,bb3,bb4,_,conf = title.split(" ")


            if "#text" in word:
                text = word["#text"]
            elif "em" in word:
                text = word["em"]
            elif "strong" in word:
                text = word["strong"]
            else:
                assert False

            cells.append(text)
            confidences.append(conf)
    except KeyError:
        print doc
        raise

    return cells,confidences