***************************************
Marking Tasks in Zooniverse Aggregation
***************************************

How does the aggregation engine handle marking tasks? In the __aggregate__ function in AggregateAPI, the key step is::

    if marking_tasks != {}:
        aggregations = self.__cluster__(used_shapes,raw_markings,image_dimensions)

Note that marking tasks are stored as a dictionary (more on that later) so if the dictionary is non-empty, we have markings to aggregate.

In AggregateAPI.__cluster__ we have the basic following code

.. literalinclude:: ../../engine/aggregation_api.py
    :language: python
    :lines: 465-504

This is the main code for aggregating markings. So what's going on? Aggregating markings is also known as clustering markings. Note that we only cluster over shapes that are actually used. Also note that we cluster on shapes and not individual tools. To understand why, consider an example of Penguin Watch where people can mark a penguin as either an adult or chick.
Both of these markings are made with a point marking tool. What happens if someone gets the type wrong? In the classification step of our code we'll return a probability of what type the cluster actually is. The alternative would be after each marking to have a follow up question which asks the user what type a "thing" - this would double the number of clicks a user would have to make.

There are several different types of marking tools available in Panoptes:

* point
* line
* ellipse
* rectangle
* arbitrary polygon

In the code above we iterate over each shape independently and do the clustering for each shape. We don't worry about things like false positives yet (e.g. thinking that a rock is a penguin) - this gets handled by the follow up classification.

In clustering.py, there are two main functions

* __aggregate__(self,raw_markings,image_dimensions)
* __cluster__(self,markings,user_ids,tools,reduced_markings,dimensions,subject_id)

The first function, __aggregate__ is what we call from the outside and further divides the set of markings up further to pass on to __cluster__, i.e. __aggregate__ takes all the markings for a given workflow over multiple subjects and ___cluster__ takes markings for a single subject and single task within that workflow.
