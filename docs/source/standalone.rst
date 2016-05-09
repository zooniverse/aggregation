Running a Standalone Aggregation Engine
#######################################

This chapter talks about what is needed to get the Aggregation Engine to run without access to the Panoptes DB. So the idea is that researchers can download a copy of their classifications which can be read directly into the aggregation engine. There are a whole bunch of required libraries for Python (openCV etc.). One possibility is to wrap all of this into a docker image (which is currently supported). The only problem is how to copy the csv input files into the aggregation docker image - definitely doable but library requirements aside, it seems a lot neater to have the aggregation engine just running locally. Ideally it would be nice to just have people do "pip install zooniverse-aggregation".

There are some required libraries for running the aggregation on AWS such as the postgres and cassandra libraries which would not be needed for a local run (or at least shouldn't be required). Not sure if you can install the cassandra library if you don't have cassandra actually locally installed.

Once you have downloaded the csv file for your classifications, loading them into python is trivial ::

    import pandas
    classifications = pandas.read_csv("/home/ggdhines/Downloads/copy-of-kitteh-zoo-subjects-classifications.csv")

To get the annotations, you simply do ::

    classifications["annotations"]

There are a couple of key methods in aggregation_api.py that you would need to overwrite in order to get aggregation running locally.

1. \__yield_annotations__ - This is what currently gets the annotations from cassandra, groups them by subject id, and then for each subject yields all of the annotations for that subject. A local instance of the aggregation engine should not rely on Cassandra (or even postgres) so we would need to rewrite this function to treat the csv file as our input database. Might be slightly inefficient since we would have to search through the whole dataframe for each subject_id.

You will still need to know the numerical project id - this allows the aggrgation engine to connect to panoptes to get the workflow structure(s). You will probably also need to provide your Panotpes userid and password in the yml file (don't think that workflow structure is public information). However, your log in doesn't need to have admin access.

2. \__migrate__ - this function was orginally created to copy classifications from postgres to cassandra when we thought that Cassandra was how we would eventually store the annotations. However, this has turned out to not be the case so even for the production AWS aggregation engine this function isn't really useful anymore. The one thing that it does do is return the list of all subjects which have annotations for the given workflow id - we can get this based purely on the csv file. So for the local aggregation engine, we would have ::

After we've loaded the classifications csv, we can filter based on the desired workflow with ::

    data_frame[data_frame["workflow_id"] == workflow_id]

Although subject_id is stored as part of the subject_data, to search/filer based on subject ids, we need it to be its own field/column (see https://github.com/zooniverse/Panoptes-Front-End/issues/2543 for current status). Once that's supported, searching for classifications for specific ids, or finding the list of all subjects which have classifications should be trivial.

3. \__setup__ this is what sets up everything for an aggregation run such as connecting to the postgres and cassandra db. Neither of these are needed for a local run so we should overwrite __setup__ - the only thing that __setup__ needs to do for a local run is connect to Panoptes
