Zooniverse Aggregation Code

This repo allows you to do aggregation with Zooniverse projects. Aggregation is the process of taking classifications/markings/transcriptions from the multiple users who see each subject and combining them into a "final" answer that is useful for researchers. 
For example, if 3 out of 4 people say that an image contains a zebra, aggregation would report that there is a 75% probability that the image does indeed contain a zebra.

The directory to do all of this in "engine". This is the code base that runs every time you press "export aggregations" in the builder builder page. You can also run things locally if you want - this is especially useful if you have an Ourboros project (just ignore that if you don't already know what Ourboros is) or if you want to do bespoke aggregation or fix a bug.

The aggregation engine needs a bunch of stuff installed and running such as postgres and cassandra. While you can manually install all of the required packages and set them up, it is much easier to use [Docker](https://www.docker.com). Use the following steps to get the aggregation docker instance running on your computer.

1. If you don't have docker installed and running on your computer, go to the [site](https://www.docker.com) first and follow the instructions.
2. Clone the aggregation github repo on your computer, i.e. 'git clone git@github.com:zooniverse/aggregation.git'
3. In your cloned instance of the aggregation repo, run "docker-compose up". This may take a while the first time you run it. 

You should now have several docker containers running. You can check with "docker ps". You'll need a copy of the Panoptes database running which will give you the subject classifications/markings/transcriptions for all the projects. Talk to Adam or Cam about where to get them from - you'll get a file like "panoptes-panoptes-production-2016-04-25-04-56.dump" One of the containers that is running the postgres container. You'll need to put that file into the postgres container and run "pg_restore" to restore the image.

1. In the aggregation repo directory, there should now be a sub-directory called "data". Copy/move the .dump file into this directory. The data subdirectory is presistent - you'll only need to do this once and the database will continue to exist even after you rebuild any code. (Of course from time to time you may need to download an updated copy of the dump file.)
2. Connect to the postgres container with "docker exec -it aggregation_postgres_1 bash"
3. Now you'll need to create the panoptes database (this database will be empty until you run pg_restore) with the command "createdb -U postgres panoptes_development" (run this in the terminal not in psql)
4. Before restoring, you'll need to set a few environment variables
    * db='panoptes_development'
    * username='postgres'
    * local_backup_file="panoptes_staging.dump" (change this to be whatever the specific name of the .dump file you have is)
5. Finally run "pg_restore --clean --verbose -Fc -h localhost -U ${username} -d ${db} ${local_backup_file}". This will take a while (but again you only need to do this once)

The aggregation engine is now ready to be run. Exit the postgres container and use the following steps
1. docker exec -it aggregation_aggregation_1 bash
2. cd engine
3. ./aggregation_api.py project_id development 

Project_id is the numerical value. You can search for the number using lita (on slack) with something like "lita project wildcam" which will tell the project ids of all projects which have "wildcam" in their title. 
Assuming that everything worked - the aggregation_api will save the results to the /tmp directory in the docker image (no email will be sent out). There will be both a directory of results with your project id and tar.gz file. You can use "docker cp" to extract the results to your local directory.
