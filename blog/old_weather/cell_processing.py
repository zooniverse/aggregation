from learning import NearestNeighbours
try:
    import Image
except ImportError:
    from PIL import Image
import os

image_directory = "/home/ggdhines/Databases/old_weather/cells/"
log_pages = list(os.listdir(image_directory))

classification_algorithm = NearestNeighbours()

for f_count,f_name in enumerate(log_pages):
    if not f_name.endswith(".png"):
        continue
    # f_name = "Bear-AG-29-1941-0545_0_6.png"
    print f_name

    im = Image.open(image_directory+f_name)
    # im = im.convert('L')#.convert('LA')
    # image = np.asarray(im)

    retval = classification_algorithm.__process_cell__(im,plot=True)

    if retval == -1:
        break

for d in range(0,10):
    print str(d) + " -- " + str(len(classification_algorithm.transcribed_digits[d]))

