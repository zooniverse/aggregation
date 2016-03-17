********************************************
Annotate and Shakespeare's World Aggregation
********************************************

Annotate and Shakespeare's World are the two projects which currently have text aggregation. They also have rectangle/box aggregation going for marking images on a page.
Aggregation for these files is run automatically on a regular basis via a crontab (ask Adam or Cam for its location). There are a couple of files involved with aggregation for these projects. Let's start with the main one.

text_aggregation.py
###################

This is the file that is called by the crontab. The format for calling this file is
./text_aggregation.py -i project_id -e environment

* project_id is either 245 (annotate) or 376 (shakespeare's world)
* environment is either production or development. This affects which databases the code tries to connect to. In development, the code relies on a local instance of both Cassandra and postgres

In text_aggregation.py, we have the class TranscriptionAPI which is a subclass of AggregationAPI (from aggregation_api.py). TranscriptionAPI has 3 main methods to call:

* __setup__ - connects to the various databases (mostly just calls __setup__ in AggregationAPI but also takes care of some things that are specific to Annotate and Shakespeare's world)
* __aggregate__ - does the actual aggregation - mostly just a slightly modified copy of AggregationAPI's __aggregate__ (some refactoring would be good)
* __summarize__ - both projects want emails sent out with the aggregation results. This happens in __summarize__