**************************
Aggregation with Ouroboros
**************************

Ouroboros is the old Zooniverse platform. Chances are that if you didn't already know that, this page isn't relevant to you. There are still some projects such as Penguin Watch and Snapshot Serengeti which run on Ouroboros. We definitely would love to move those projects over to Panoptes but in the mean time, these projects will still be on Ouroboros. This is important for aggregation since Ouroboros stores data in a very different way than Panoptes does. (Even after we move everything over to Panoptes, it may be the case we can't move old classification data from Ouroboros so, Ouroboros is here to stay for a while longer.)

In this page, I'll talk about how to access classifications in Ouroboros. I'll also talk about how to make use of the Panoptes based aggregation tools - it will take a bit of messing about.

Ouroboros DB dumps
==================

All of the data (classifications, subject data) needed for aggregation is stored in Mongo DB. There are daily dumps created for Ouroboros project stored on AWS. If you're an external researcher, we'll need to write a script to that provides you with the data (ask Adam). If you're a Zooniverse developer, ask Cam or Adam where the dumps are.
Once you've copied the dump to your computer, make sure that mongo DB is running locally. You need to use "mongorestore" to restore the database (take the data from the dump files and put it into mongo db). If you have mongorestore version 3.2 or later, you shouldn't need to decompress the files. If you do need to decompress the files the commands are: (Obviously update the date in the file name to whenever you are running this.)

.. code-block:: console

    tar -xvf penguin_2016-03-22.tar.gz
    mongorestore --db penguin penguin_2016-03-22

So we've restored this database. If you go to the mongo DB interface (there are probably some decent GUIs for Mongo DB but I've always used the command) via the command "mongo" (this works in Linux) and enter "show dbs", you should now see the database "penguin". (There are plenty of good online tutorials for how to explore the database via the CLI)

Connecting to the database and iterating through classifications with Python is easy:

.. code-block:: python

    import pymongo

    client = pymongo.MongoClient()
    db = client['penguin_2015-06-08']
    classification_collection = db["penguin_classifications"]
    subject_collection = db["penguin_subjects"]

    for c in classification_collection.find({"user_name":expert})[:25]:

The classification "c" is a dictionary with a couple of important keys

* annotations - the actual annotations made by their user (in the case of Penguin Watch, the markings for each of the penguins)
* tutorial - if this annotation was made as part of a tutorial - should probably just skip those
* subjects - contains the zooniverse ids, allows you to match annotations from different users on the same subject and the image's location on AWS (in case you want to download it)
* user_name - for logged in users, this is their user name (so the above code searches for 25 classifications made by a given user). Field does not exist if user is not logged in

Let's look at some annotations for Penguin Watch - annotations in all projects are stored in JSON format.

.. code-block:: json

    [
      {u'value': u'no', u'key': u'animalsPresent'},
      ...
    ]

So this annotation is for a subject which the user has said does not contain any penguins.

.. code-block:: json

    [
      {u'value': {u'1': {u'y': u'118.132', u'x': u'-60.491', u'frame': u'0', u'value': u'adult'}, u'0': {u'y': u'167.988', u'x': u'127.011', u'frame': u'0', u'value': u'adult'}}, u'key': u'animalsPresent'},
    ]

And this annotation is for an image where the user has marked two penguins. Each penguins has 3 important fields

* x,y - coordinates
* value - adult or chick

In the above example, we see that for Penguin 1 there is a negative x coordinate - this is due to a problem with the UI and this marking should be ignored. Note that as always for images because computer graphics is a bit silly, 0,0 (the origin) for images is the top left hand corner.

If we wanted to find all classifications for a given subject id (say zooniverse_id), we would use

.. code-block:: python

    for classification in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):

This is really not efficient code - there is no index created for zooniverse_id (I'm not sure that one can be created when "zooniverse_id" is stored in the above manner). So we will have to repeatedly search through the whole DB. We could limit our searches with

.. code-block:: python

    for classification in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}).limit(10):

So this would return only 10 - still not very efficient (especially if somehow an image didn't get 10 classifications - this is especially important for something like Snapshot Serengeti where subjects may be retired with different numbers of views). To see just how bad this could be, let's figure out how many classifications we have in the database

.. code-block:: console

    use penguin;
    db.penguin_classifications.count();

Note that in Mongodb terms - penguin is the database (or db) and penguin_classifications is a "collection" (kinda like a table).  The above is for the Mongodb CLI. For Python use

.. code-block:: python

    print classification_collection.count()

Nice :) To improve things, let's create an index. We'll start with adding a "zooniverse_id" field to every classification

.. code-block:: python

    for c in classification_collection.find():
      _id = c["_id"]
      zooniverse_id = c["subjects"][0]["zooniverse_id"]

      classification_collection.update_one({"_id":_id},{"$set":{"zooniverse_id":zooniverse_id}})

That takes a while. But now we can search for a given zooniverse_id with

.. code-block:: python

    for classification in collection.find({"zooniverse_id":zooniverse_id}

Now to create the index

.. code-block:: python

    db.profiles.create_index([('zooniverse_id', pymongo.ASCENDING)],unique=False)

That part is pretty quick. Searching for all classifications for a given subject still takes a little bit but seems to be better (a quantitative difference would be nice - if it is actually still really bad, we might need to move the db over to postgres or something - but only as a last resort).

Ourboros to Panoptes
####################

Now to the actual clustering - we want to use the agglomerative clustering available through panoptes. (Link to be inserted later talking about the whole theory behind that) But we don't have to create an instance of AggregationAPI (which would mean basically whole "fake" panoptes project) - we can skip all of that.
Agglomerative clustering is available through engine/agglomerative.api. We can easily import Agglomerative (the class in agglomerative.api that can do the clustering for penguin marking).

.. code-block:: python

    import sys
    sys.path.append("/home/ggdhines/github/aggregation/engine")
    from agglomerative import Agglomerative

The code above adds the directory to the Python path (make sure to change it to the correct directory for your computer). The constructor for Agglomerative takes two parameters, either of which matters for Penguin Watch so feel free to pass in some dummy variables. The method within Agglomerative that we will class to do the actual clustering is

.. code-block:: python

    def __cluster__(self,markings,user_ids,tools,reduced_markings,image_dimensions,subject_id):

So we have to take the annotations from mongodb and convert them into the above format. The parameters for __cluster__ are

* markings - the raw x,y coordinates
* user_ids - probably go with ip addresses - that way you guarantee that everyone has a id, even if they are not logged in
* tools - either "adult" or "chick". This isn't actually used in the clustering algorithm. this is used later on to determine what type of penguin each cluster is mostly likely to be
* reduced_markings - doesn't matter for just point markings - just make it equal to the markings
* image_dimensions - also doesn't matter for Agglomerative
* subject_id - doesn't matter for Agglomerative (Agglomerative is a subclass of Clustering and there are other sub classes of Clustering for which image_dimensions and subject_id matter)

For a given zooniverse id, the code for converting the Ourboros annotations into Panoptes ones, and calling the clustering algorithm is::

    for c2 in classification_collection.find({"zooniverse_id":zooniverse_id}):
        if "finished_at" in c2["annotations"][1]:
            continue

        if "user_name" in c2:
            id_ = c2["user_name"]
        else:
            id_ = c2["user_ip"]

        try:
            for penguin in c2["annotations"][1]["value"].values():
                x = float(penguin["x"])
                y = float(penguin["y"])
                penguin_type = penguin["value"]

                markings.append((x,y))
                user_ids.append(id_)
                tools.append(penguin_type)
        except AttributeError:
            continue

    if markings != []:
        clustering_results = clustering_engine.__cluster__(markings,user_ids,tools,markings,None,None)

The first if statement inside the loop checks to see if the user marked any penguins at all (just using some knowledge about the structure of the annotations dictionary). We then extract the user id.
The try statement surrounds the extraction of the individual coordinates - occasionally we may get some badly formed annotations due to browser issues. We'll just skip those annotations. Note that all of the values (including x and y coordinates) associated with each marking are stored in string format so we need to convert them to float values.

Let's look at the results. The variable clustering_results is a tuple with the second value being the time needed for the algorithm to run - this is only really useful for papers etc. so we'll ignore it. The first item in clustering_results is the actual results we are interested in. This is a list of clusters - one cluster (hopefully) per one penguin. We can use the Python json library to print out the results for one pengin

.. code-block:: json

    {
    "center": [
        529.71000000000004,
        42.536999999999999
    ],
    "cluster members": [
        [
            523.387,
            40.582
        ],
        [
            523.649,
            40.776
        ],
        [
            529.712,
            42.063
        ],
        [
            528.786,
            42.844
        ],
        [
            528.824,
            41.469
        ],
        [
            526.054,
            48.076
        ],
        [
            526.69,
            38.973
        ],
        [
            527.087,
            42.537
        ],
        [
            527.83,
            40.357
        ],
        [
            530.179,
            44.801
        ],
        [
            529.71,
            45.932
        ],
        [
            531.925,
            44.746
        ],
        [
            531.803,
            43.478
        ],
        [
            541.235,
            38.68
        ],
        [
            536.761,
            43.378
        ],
        [
            533.883,
            44.69
        ],
        [
            534.46,
            41.449
        ]
    ],
    "num users": 17,
    "tool_classification": [
        {
            "adult": 1
        },
        -1
    ],
    "tools": [
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "chick",
        "adult",
        "adult",
        "adult"
    ],
    "users": [
        users
    ]
    }

So we have some fields to look at.

* center - the median center of this cluster
* cluster members - the individuals coordinates of each marking
* num users - how many people have marked this penguin
* tool_classification - ignore this - honestly not sure why this is here. Have made a note to double check
* tools - what tools (adult or chick) users have used to mark this penguin
* users - the list of users which marked this people. We've removed the list of users since that included some ip addresses.

For each cluster, we want to report three things in our csv output file.

* the center
* probability of true positive
* probability of "adult"

We can get the first field directly from the clustering results. Probability of true positive is how likely the cluster represents an actual penguin - as opposed to someone confusing some rocks and snow with a penguin. All things being equal, the markings a cluster contains, the more likely it is that that cluster is a true positive.
So for the "probability" of being a true positive, we'll report the percentage of users who have a marking in that cluster. (Quotations around probability there since it is a slight abuse of the term.) We'll also report the raw number of people who marked a penguin - sometimes the raw number is useful in addition to the percentage. Similarly for probability of adult we'll report the percentage of people who marked a penguin as an adult (as opposed to being a chick.)

Regions of Interest
*******************

To make things more interesting, with Penguin Watch, users are often asked to only mark penguins in a certain region of an image. The rest of the image is grayed out and it should, in theory, be impossible for people to not even make markings outside the region of interest (ROI).
However, things don't always work out in practice and we can have markings outside the ROI (most likely due to browser issues). So after we've found a cluster of markings - we need to double check that the center is inside of the ROI.

At the same time, we also need to convert zooniverse ids into the subject ids which the penguin watch team will understand. Each image has a "path" id which is how the researchers organized their data. To access these path ids::

    path = subject_collection.find_one({"zooniverse_id":zooniverse_id})["metadata"]["path"]

An example result would be - PETEa/PETEa2013b_000157.JPG. "PETEa" is the camera id which is how we can access the ROI for this image. To make things slightly more complicated, some of the path names have changed between what Zooniverse has and what the Penguin Watch researchers have. Below is the complete list of all name changes that Zooniverse is currently aware of.

=============   =================
Zooniverse ID   Pre-zooniverse ID
-------------   -----------------
BALIa2014a
BOOTa2012a	PCHAa2013
BOOTa2014a
BOOTb2013a	PCHb2013
BOOTb2014a
BOOTb2014b
BROWa2012a
CUVEa2013a
CUVEa2013b
CUVEa2014a
DAMOa2014a
DANCa2012a	DANCa2013
DANCb2013a
DANCb2014a
FORTa2011a
GEORa2013a
GEORa2013b
HALFa2012a
HALFa2013a
HALFb2013a
HALFc2013a
LOCKa2012a
LOCKa2012b
LOCKa2013a
LOCKb2013a
LOCKb2013b
MAIVb2012a	MAIVb2013
MAIVb2013a
MAIVb2013c
MAIVc2013
MAIVc2013b
MAIVd2014a
NEKOa2012a	NEKOa2013
NEKOa2013a
NEKOa2013b
NEKOa2013c
NEKOa2014a
NEKOb2013
NEKOc2013a
NEKOc2013b
NEKOc2013c
NEKOc2014b
PCHAc2013
PETEa2012a
PETEa2013a	PETEa2013a
PETEa2013b	PETEa2013a
PETEa2013c
PETEa2014b
PETEb2012a
PETEb2012b	PETEb2013
PETEb2013b
PETEc2013a
PETEc2013b
PETEc2014a
PETEc2014b
PETEd2013a
PETEd2013b
PETEe2013a
PETEe2013b
PETEf2014a
SALIa2012a
SALIa2013a
SALIa2013b
SALIa2013c
SALIa2013d
SALIa2013e
SIGNa2012a
SIGNa2013a	SIGNa2013
SPIGa2012a
SPIGa2013b
SPIGa2014a
SPIGa2014b
YALOa2013a
YALOa2014c
=============   =================

So the left hand side is that Zooniverse has and the right hand side gives any changes necessary for the researchers to make sense of the data. The ROIs are stored in the Penguins repo on the Zooniverse github site; under the public directory in the roi.tsv. To load the values from this file use the code::

    with open("/Penguins/public/roi.tsv","rb") as roiFile:
            roiFile.readline()
            reader = csv.reader(roiFile,delimiter="\t")
            for l in reader:
                path = l[0]
                t = [r.split(",") for r in l[1:] if r != ""]
                roi_dict[path] = [(int(x)/1.92,int(y)/1.92) for (x,y) in t]

The first readline above skips the header line. Then we read through each path one at a time. Each corner is represented by a x,y value (tab separated - so we set delimiter = "\t", see the Python csv library for more info). We scale each set of values by 1.92 which is the difference between the original image size and the size of the image shown to the users (forget which that number is documented).

To check if a given marking is inside of the ROI, we use the following code (remember that origin is at the top LHS of the image) ::

    def __in_roi__(self,site,marking):
        """
        does the actual checking
        :param object_id:
        :param marking:
        :return:
        """

        if site not in roi_dict:
            return True
        roi = roi_dict[site]

        x = float(marking["x"])
        y = float(marking["y"])


        X = []
        Y = []

        for segment_index in range(len(roi)-1):
            rX1,rY1 = roi[segment_index]
            X.append(rX1)
            Y.append(-rY1)

        # find the line segment that "surrounds" x and see if y is above that line segment (remember that
        # images are flipped)
        for segment_index in range(len(roi)-1):
            if (roi[segment_index][0] <= x) and (roi[segment_index+1][0] >= x):
                rX1,rY1 = roi[segment_index]
                rX2,rY2 = roi[segment_index+1]

                # todo - check why such cases are happening
                if rX1 == rX2:
                    continue

                m = (rY2-rY1)/float(rX2-rX1)
                rY = m*(x-rX1)+rY1

                if y >= rY:
                    # we have found a valid marking
                    # create a special type of animal None that is used when the animal type is missing
                    # thus, the marking will count towards not being noise but will not be used when determining the type

                    return True
                else:
                    return False

        # probably shouldn't happen too often but if it does, assume that we are outside of the ROI
        return False

An example of a site name is "BALIa2014a". If for whatever reason we don't have an ROI for the given site - just say yes. Don't have time right now for the full details of what's happening above. (Hopefully later.)
