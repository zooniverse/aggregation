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

An example result might be ::

    [u'1274968', u'1274969', u'1276058', u'1279124', u'1273572', u'1274458', u'1273570', u'1274964', u'1273574', u'1273575' ...]

Note that the subject ids are actually strings (even though they are all numbers). The u'' simply means unicode.

To print out the aggregation results for one subject we can do ::

    print(json.dumps(aggregation_results["1274968"], sort_keys=True, indent=4, separators=(',', ': ')))

There is still a lot of output. aggregation_results["1274968"] is a dictionary so again we can look at the keys in the dictionary ::

    print(aggregation_results["1274968"].keys())


This will give ::

    ['text', 'raw transcriptions', 'metadata']

1. 'text' - the aggregated text
2. 'raw transcriptions' - the original transcriptions for the given subject. Useful if you want to know which transcriptions were ignored
3. 'metadata' - the metadata provided by Tate/Folger. Useful for figuring out which document each subject is

Looking at the text aggregation results - the aggregation results are stored in a list. Each element in the list refers to cluster of transcriptions, all transcribing the same text (each transcription in the cluster is by a different user).
Each cluster has a number of properties ::

    ['text', 'individual transcriptions', 'coordinates', 'accuracy']

1. 'text' - the aggregate text (probably what you are most interested in)
2. 'individual transcriptions' - the individual transcriptions in the cluster
3. 'accuracy' - how much agreement there is between all of the transcriptions in the given cluster

So to iterate over all of the aggregate transcriptions, we could do ::

    for line in aggregation_results["1274968"]["text"]:
        print(line)

Each aggregate line(cluster) will have a number of different properties.

1. 'aggregated_text' - this is the actual final text
2. 'coordinates' - the coordinates of this particular line (based on an average of all of the coordinates given by users). The format for these lines is [x1,y1,x2,y2] i.e. the first two values give the starting point of the line and the next two give the ending point.
3. 'individual transcriptions' - contains both the text and the coordinates of the transcriptions making up this cluster
4. 'accuracy' - for what percentage of characters did the users reach consensus on?
5. 'images' - present if there were any images marked in the document (format is [x1,y1,x2,y2] giving two opposite corners)
6. 'variants' - a list of variants, i.e. words that are not in the dictionary if you are looking for original words (key is always there but idea is only supported in Shakespeare's world)

In "aggregated_text", there will be the tags that Annotate/SW use. There will also be some special tags

1. <disagreement>...</disagreement> - these are used to show places where people have not reached consensus and we list every one of the possible options that people gave
2. <option>...</option> - inside of the disagreement tags these option tags give the different options (note that the options can be more than one character long)

In the individual transcriptions there are a couple of special characters as well. These characters have been inserted so that all of the users transcriptions are of the same length. So string[4] (for example) will refer to the same character for all transcriptions. These characters are non-printing ascii values. The list below gives the ASCII value for each of these special characters. So for example, 24 doesn't mean the characters "24" but the character with ASCII value 24 (which is a non-printing character).

1. 24 - the aggregation algorithm determined that a user skipped a character. For example if we have two transcriptions "hello" and "ello", then the secon transcription would be list as "\u0018ello". "\u0018" is a single character, specifically the character with ASCII value 24 listed in unicode format. (Converting to unicode was necessary for saving values in the database.)
2. 25 - for Folger only this corresponds to spaces before or after where the user started transcribing. So with Folger, the example of "hello" and "ello" would be "\x19ello" (\x19 is another way of saying character 25 in unicode with hex).

There is a subtle differnce between these two special characters. (Again this only applies to Shakespeare's world):

1. Consider "helo" vs "hello" - the first would be represented as "he\u0018l0". So here "\u0018" means that the first user thinks there is nothing in the spot.
2. Now consider "ello" and "hello" - the first would be presented as "\x19ello". Here "\x19" would mean "no option" - it might be that the first user didn't think there was a "h", but it also could have been that the first user merely decided to start transcribing at the "e". (Seems more realistic if you replace each letter with a whole word.) So "\x19" has no effect on the aggregation.

To get the average accuracy per line for a subject we would do::

    import numpy as np
    all_accuracies = [l["accuracy"] for l in aggregation_results["1274968"]["text"]
    print(np.mean(all_accuracies))

This would of course give a bias towards small lines of text.

For variants, we are looking for cases where at least two people have transcribed the same word. In that case, the display name of each user is given. (So we have a list of lists.)