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
