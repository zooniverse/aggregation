Understanding Aggregation Results in JSON Format
================================================

If you have the aggregation results from Annotate or Shakespeare's World, they are in JSON format. The advantage of JSON over CSV is flexibility. This wiki is a brief discussion of how to deal with them. The following code is in Python - JSON and Python are a natural fit and Greg (the person who wrote the Annotate/Shakespeare's World aggregation code) works in Python. That said, other languages should support JSON - it may just not be as simple.

To load a JSON file in Python, use the following code ::

    import json
    with open('aggregation_results.json') as data_file:
        aggregation_results = json.load(data_file)

In Python JSON variables are simply either a dictionary, list or literal values such as strings or numbers. JSON variables are recursively defined so each element of a dictionary/list is itself a JSON variable. Literal values are the base cases.

To print out the data in json we can use pretty print (we could also just do "print data" or "print(data)" in Python 3 but this would give an un-readable wall of text) ::

    print(json.dumps(aggregation_results, sort_keys=True, indent=4, separators=(',', ': ')))

Note while the aggregation engine is written in Python 2, for compatibility, all of the print statements have been done in Python 3. To allow for Python 3 style print statements (with the brackets) in Python 2, simply put at the top of your code the line ::

    from __future__ import print_function

Aggregation_results is massive and even with nice formatting, the output is going to be overwhelming. So let's break it down. The top level of the json results is a dictionary where each key is a subject id value which maps to the aggregation results for that subject. (The subject ids are just numerical values created by Zooniverse, for example "1281157". This numbers won't mean anything to you - they are just Zooniverse's way of uniquely identifying subjects (i.e. documents). This is probably not the best way to identify each subject but we never settled on a better way - so if you have a specific document you are searching for you will need to search through the whole list.)
The aggregation results will only contain subjects which have been retired. To see all of the subject ids with aggregation results simply do ::

    print(aggregation_results.keys())


