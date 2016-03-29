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



