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
* subject_ids - the zooniverse ids, allows you to match annotations from different users on the same subject
* subjects - contains metadata - most importantly file names which will make sense to the researchers (the zooniverse subject ids won't mean anything to them)

Let's look at some annotations for Penguin Watch - annotations in all projects are stored in JSON format.

.. code-block:: json

    [
      {u'value': u'no', u'key': u'animalsPresent'},
      ...
    ]
    