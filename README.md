# Zooniverse Aggregation Code

This repo allows you to do aggregation with Zooniverse projects. Aggregation is the process of taking classifications/markings/transcriptions from the multiple users who see each subject and combining them into a "final" answer that is useful for researchers. 
For example, if 3 out of 4 people say that an image contains a zebra, aggregation would report that there is a 75% probability that the image does indeed contain a zebra.

Documentation for the Zooniverse Aggregation code can be found at [https://developer.zooniverse.org/projects/aggregation/en/latest/](https://developer.zooniverse.org/projects/aggregation/en/latest/) (We are currently experiencing severe problems with loading (lots of 403 errors and pages just not loading). We've raised issue and are looking into alternative hosting options - but in the mean time just keep reloading until you get the site to show.)

There is a draft of a paper that I was planning on writing on ocr and Tesseract. There is lots of random code in the active-weather directory and a [link](https://drive.google.com/open?id=0By-LeWaBxt31ZVBfZ2R6SXl4NWs) to an unfinished paper I was working on - that paper has some good ideas on how to do line detection/extraction for tabular data.

The directory to do all of this in "engine". This is the code base that runs every time you press "export aggregations" in the builder builder page. You can also run things locally if you want - this is especially useful if you have an Ourboros project (just ignore that if you don't already know what Ourboros is) or if you want to do bespoke aggregation or fix a bug.

## Running the Aggregation Code Locally

The aggregation engine needs a bunch of stuff installed and running such as PostgreSQL and Cassandra. While you can manually install all of the required packages and set them up, it is much easier to use [Docker](https://www.docker.com). Use the following steps to get the aggregation docker instance running on your computer.

1. If you don't have Docker installed and running on your computer, go to the [site](https://www.docker.com) first and follow the instructions.
2. Clone the aggregation github repo on your computer, i.e. `git clone git@github.com:zooniverse/aggregation.git`
3. In your cloned instance of the aggregation repo, run `docker-compose up`. This may take a while the first time you run it. 

You should now have several docker containers running. You can check with `docker ps`. You'll need a copy of the Panoptes database running which will give you the subject classifications/markings/transcriptions for all the projects. Talk to Adam or Cam about where to get them from - you'll get a file like "panoptes-panoptes-production-2016-04-25-04-56.dump". One of the containers that is running the PostgreSQL container. You'll need to put that file into the PostgreSQL container and run `pg_restore` to restore the image.

1. In the aggregation repo directory, there should now be a sub-directory called "data". Copy/move the .dump file into this directory. The data subdirectory is presistent - you'll only need to do this once and the database will continue to exist even after you rebuild any code. (Of course from time to time you may need to download an updated copy of the dump file.)
2. Connect to the postgres container with `docker exec -it aggregation_postgres_1 bash`
3. Now you'll need to create the panoptes database (this database will be empty until you run `pg_restore`) with the command `createdb -U postgres panoptes_development` (run this in the terminal not in psql)
4. Before restoring, you'll need to set a few environment variables
    * db='panoptes_development'
    * username='postgres'
    * local_backup_file="panoptes_staging.dump" (change this to be whatever the specific name of the .dump file you have is)
5. Finally run `pg_restore --clean --verbose -Fc -h localhost -U ${username} -d ${db} ${local_backup_file}`. This will take a while (but again you only need to do this once)

You'll need a file called `aggregation.yml` to provide a bunch of setup values. A simple example file is in `config/aggregation.yml` in the aggregation repo and is already loaded for you into the docker image. In production, this YAML file allows us to make a secure (admin) connection to Panoptes as well as Rollbar (for reporting errors) and connecting to the postgres/cassandra db. 
For development, you just need to connect to the postgres/cassandra dbs in the docker images and the default values in the aggregation.yml file will allow you to do that. You only need a secure (possibly) admin connection to Panoptes to do a few specific things (such as for Annotate/Shakespeare's world being able to retire subjects via an API call). 
For the "development" environment in the YAML file we just use a public connection to Panoptes (which I don't think even really gets used). The YAML file used in production (ask Adam for details/location) has all the necessary password/tokens etc. to make all the necessary connections.

There is one password that you will need to set in the YAML file and that is the postgres password. The docker image comes with the default password which could probably just be included in the YAML file but to play it safe is not included. If you want to set the postgres user password, log into the postgres container and in psql use "alter user postgres password 'apassword';"

The aggregation engine is now ready to be run. Exit the postgres container and use the following steps 

1. `docker exec -it aggregation_aggregation_1 bash`
2. `cd engine`
3. `./aggregation_api.py project_id development`
4. The first time you run it, make sure that `__reset_cassandra_dbs__()` is executed, i.e. right after `project.__setup__()`, run `project.__reset_cassandra_dbs__()`. This only needs to be done once

`project_id` is the numerical identifier for a project. You can search for the number using lita (on Slack) with something like `lita project wildcam` which will tell the project IDs of all projects which have "wildcam" in their title. 
Assuming that everything worked - the aggregation_api will save the results to the `/tmp` directory in the Docker image (no email will be sent out). There will be both a directory of results with your project id and tar.gz file. You can use `docker cp` to extract the results to your local directory.

## Troubleshooting

Sometimes something goes wrong and the aggregation engine crashes. There are 3 reasons why this could happen.

1. A system crash - for example AWS goes down (but hopefully not). This is probably the rarest cause and Adam/Cam/etc. should probably realize that something is up with AWS
2. Newly deployed code interrupts an aggregation run - when deploying new code, there is no way of checking to see if there is currently an aggregation run in progress. Any such runs will automatically be stopped. The file `spot_check.py` runs every hours to see if there are any interrupted runs. If so an email is sent out to Zooniverse admin (the email is specified in the aggregation.yml file) (There is currently no way to automatically restart any such runs but this is again a relatively rare case.)
3. Edge cases - something about your classifications causes an expected case in the aggregation code and Python raises an exception. This is by far the most likely cause of error. There is currently no way of automatically emailing people to let them know that such as a case has occurred (definite future plans) but any such errors are reported automatically via Rollbar and github. To see the errors in Rollbar you need to have an account there and access to the Zooniverse_Aggregation dashboard. Currently Cam, Adam, Marten and Greg have such access. Rollbar will also automatically raise an issue on github (in the Aggregation repo) and show the Python exception. However, currently the Github issues do not report the project id so you may not be completely sure if your project caused a specific error. (The project ids are reported in Rollbar).  

Aggregation runs can take a little while - depending on how big your project is and what type of tools you used (polygon aggregation takes longer than simple classification aggregation). Depending on the current load on Panoptes, it might take a little while for the aggregation to start. Currently there is no way of checking to see the current state of the aggregation. If you haven't received an email with a link to your results in a couple of hours, check github to see if any new issues have been raised. In the future we aim to send out emails saying when an aggregation run has crashed.

## Using Classification CSV files as input

You can also use the classification csv files (available from the panoptes project builder page) as input for the aggregation engine. If you use the CSV files you do not need to connect to postgres or cassandra. However the current implementation is slightly inefficient since values are stored in a Pandas dataframe without indexing (this could definitely be improved in the future). You will need to also have a YAML file located in `/app/config/aggregation.yml` with the following code:

```yaml
development: &default
  panoptes_username: TBD
  panoptes_password: TBD
```
  
Since the aggregation code will need to connec to Panoptes to get some information. (The location could be updated in the future - /app/config is just where the file is on AWS) The code to run aggregation locally is in local_aggregation. The code to run it is:

```python
from local_aggregation import LocalAggregationAPI
project = LocalAggregationAPI(63,"/home/ggdhines/Downloads/copy-of-kitteh-zoo-subjects-classifications.csv")
project.__setup__()
project.__aggregate__()

with csv_output.CsvOut(project) as c:
    c.__write_out__()
```

## Important Notes

1. The aggregation code for Annotate and Shakespeare's world runs daily but sends out a weekly summary email to the project team. This email is sent out on [Tuesdays](https://github.com/zooniverse/aggregation/blob/281279b9367167c42920648e03dc65ba3e2be038/engine/text_aggregation.py#L452) (nothing special - just first set it up on a Tuesday). The code takes a while to run so probably best not to deploy new code on Tuesdays.
   
   You can also log on to the EC2 instance and check to see if the processes are running. Either `ps -auxf` or `top -c` will work. `top -c` might be a bit better since the aggregation code for AnnoTate and Shakespeare's World takes up a lot of CPU so they tend to be at the top of the list. 
