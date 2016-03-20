# import mnist
# from mnist import MNIST
from sklearn import neighbors
from sklearn.decomposition import PCA

n_neighbors = 15

mndata = MNIST('/home/ggdhines/Databases/mnist')
training = mndata.load_training()

weight = "distance"
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)

pca = PCA(n_components=80)
T = pca.fit(training[0])
reduced_training = T.transform(training[0])
print sum(pca.explained_variance_ratio_)
# clf.fit(training[0], training[1])
clf.fit(reduced_training, training[1])

# todo - refactor digit rescaling probably only matters for NN

def extract(image):
    digits = []
    confidence = []
    x_location = []
    assert isinstance(image,np.ndarray)
    num_col,num_row = image.shape

    # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # blur = cv2.GaussianBlur(image,(5,5),0)
    _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pts = np.where(image>0)

    # # assert False
    #
    # colours = {}
    #
    # for c in range(num_col):
    #     for r in range(num_row):
    #         pixel_colour = tuple(image[c,r])
    #         if pixel_colour not in colours:
    #             colours[pixel_colour] = 1
    #         else:
    #             colours[pixel_colour] += 1
    #
    # most_common_colour,_ = sorted(colours.items(),key = lambda x:x[1],reverse=True)[0]
    # pts = []
    # for c in range(num_col):
    #     for r in range(num_row):
    #         pixel_colour = tuple(image[c,r])
    #
    #         dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(pixel_colour,most_common_colour)]))
    #
    #         if dist > 40:
    #             # plt.plot(r,c,"o",color="black")
    #             pts.append((r,c))

    # x,y = zip(*pts)
    # plt.plot(x,y,"o")
    # plt.show()

    # hopefully corresponds to an empty cell
    if pts == []:
        return [],[]

    rows,columns = pts
    min_r =min(rows)
    max_r =max(rows)

    min_c =min(columns)
    max_c =max(columns)

    # pts = np.asarray([(r,c) for (r,c) in pts if (r>(min_r))and(r<(max_r-2))and(c>(min_c+2))and(c<(max_c-2))])
    pts = np.asarray([(r,c) for (r,c) in zip(rows,columns) if (r>=min_r)and(r<=max_r)and(c>=min_c)and(c<=max_c)])

    db = DBSCAN(eps=3, min_samples=5).fit(pts)
    labels = db.labels_
    unique_labels = set(labels)

    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            col = 'k'
            continue
        else:
            col = "blue"
        class_member_mask = (labels == k)

        xy = pts[class_member_mask]
        X_l,Y_l = zip(*xy)

        x_location.append(np.median(X_l))

        max_x = max(X_l)
        max_y = max(Y_l)

        min_x = min(X_l)
        min_y = min(Y_l)

        desired_height = 20.

        width_ratio = (max_x-min_x)/desired_height
        height_ratio = (max_y-min_y)/desired_height

        if width_ratio > height_ratio:
            # wider than taller
            # todo - probably not a digit
            width = int(desired_height)
            height = int(desired_height*(max_y-min_y)/float(max_x-min_x))
        else:
            height = int(desired_height)
            # print (max_y-max_y)/float(max_x-min_x)
            width = int(desired_height*(max_x-min_x)/float(max_y-min_y))

        template = [[[1,1,1] for i in range(min_x,max_x+1)] for j in range(min_y,max_y+1)]
        for x,y in xy:
            template[y-min_y][x-min_x] = image[y][x]

        digit_image = Image.fromarray(np.uint8(np.asarray(template)))
        # plt.show()
        cv2.imwrite("/home/ggdhines/aa.png",np.uint8(np.asarray(template)))

        digit_image = digit_image.resize((width,height),Image.ANTIALIAS)
        digit_image = digit_image.convert('L')

        # # we need to center subject
        # if height == 28:
        #     # center width wise
        #
        #     y_offset = 0
        # else:
        #
        #     x_offset = 0

        x_offset = int(28/2 - width/2)
        y_offset = int(28/2 - height/2)

        digit_array = np.asarray(digit_image)


        centered_array = [0 for i in range(28**2)]

        try:
            darkest_pixel = 0
            for y in range(len(digit_array)):
                for x in range(len(digit_array[0])):
                    darkest_pixel = max(darkest_pixel,digit_array[y][x])
        except TypeError:
            # print "problem skipping this one"
            continue

        # darkest_pixel = max(darkest_pixel,100)

        for y in range(len(digit_array)):
            for x in range(len(digit_array[0])):
                # dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(digit_array[y][x],ref1)]))
                # if dist1 > 10:
                # if digit_array[y][x] > 0.4:
                #     plt.plot(x+x_offset,y+y_offset,"o",color="blue")
                # digit_array[y][x] = digit_array[y][x]/255.
                if digit_array[y][x] > 10:
                    centered_array[(y+y_offset)*28+(x+x_offset)] = digit_array[y][x]#/float(darkest_pixel)
                else:
                    centered_array[(y+y_offset)*28+(x+x_offset)] = 0

        # for index,i in enumerate(centered_array):
        #     if i > 0:
        #         x = index%28
        #         y = index/28
        #         plt.plot(x,y,"o",color="blue")
        #
        # plt.ylim((28,0))
        # plt.xlim((0,28))
        # plt.savefig("/home/ggdhines/tmp.png")
        # plt.close()

        centered_array = np.asarray(centered_array)
        # print centered_array.shape
        # centered_array = T.transform(centered_array)
        # print centered_array
        # probabilities = clf.predict_proba(centered_array)
        # confidence.append(np.max(probabilities))
        # digits.append(str(int(np.argmax(probabilities))))



        # raw_input("enter something")

    # plt.xlim((0,num_row))
    # plt.ylim((num_col,0))
    # line_removal(pts,num_col,num_row)
    # plt.savefig("/home/ggdhines/tmp.png")
    # plt.close()
    if digits == []:
        return [],[]
    digits_confidence_location = zip(digits,confidence,x_location)
    digits_confidence_location.sort(key = lambda x:x[2])
    digits,confidence,_ = zip(*digits_confidence_location)

    return digits,confidence