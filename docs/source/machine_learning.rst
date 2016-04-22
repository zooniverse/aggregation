Suppose you have an image (such as the one below) and you can trying to use machine learning (OCR, neural networks etc.) to transcribe it.
We could use something like Tesseract (probably the most common OCR engine, and open source too). There are two issues with Tesseract

* (by design) only works for typed text
* teaching Tesseract a new font is a horrible experience. There are several blogs which talk about how to teach Tesseract a new font - and they all involve considerable user effort.
This goes against my idea of being able to just have documents posted online and volunteers being able to transcribe stuff right away. All of the steps in teaching Tesseract are, in theory, doable by a computer but
Tesseract does weird counter-intuitive things. For example, adding new training examples can affect old training examples. Teaching tesseract a new font involves training image and a box file - basically the labelled data
with what each character is and a bounding box around that character in the training image. You might think of the box file as the gold standard data - so if the box file says that there is an "A" at coordinates (x,y) you would hope
that Tesseract takes that as gold standard. Unfortunately this is not the case. For example, you could have a box file with just one character (the "A" at (x,y)) and every thing is fine (i.e. Tesseract accepts that character as a valid training example).
Then you add an other entry into the box file , say "B" at coordinates (x2,y2). All of a sudden, Tesseract no long recognizes the "A". An other problem case is when you're trying to teach tesseract what a period (".") looks like. So you have an image
file with a bunch of example periods and the box file which gives a bounding box around each of those periods. Again, you would hope that Tesseract would take this as gold standard but instead (assuming that the training image only has periods), Tesseract
claims that the training image is empty. The only help on forums amounts to "fiddle around with things until you get the results you want" or "make sure to do it correctly". None of that is helpful in the least.

An experiment
*************
A fundamental limitation with machine learning is that the training data must be "similar" to the the test data. This seems reasonable and makes sense but creates problems. As a very simple example, suppose that our volunteers have transcribed documentation that only contain the letters "S" and "T".
We use those transcriptions to build up a really good transcription engine for the letters "S" and "T". We then move on to a new set of documents where there are other letters. For example, the document might have "R" in it. Ideally we'd like the engine to be able to say "there's something here, not a S or T, but I don't know what it is". That would be fine - in that case we would call on our volunteers to transcribe the "R" and we could updated our training set.
But our engine isn't trained on "R" at all - so we really don't know what will happen. We could even encounter a "S", just in a different font, which could cause the same problem.

With Active Weather, we face two competing problems

* connected characters - especially when text is typed (as opposed to hand written), the characters are usually distinct. But sometimes the characters are touching - for example in the below image:

.. image:: images/touching_chars.jpeg
    :width: 200px
    :align: center
    :height: 100px

There has been a lot of work done on character segmentation (word segmentation at least for typed text is much easier) [links to be inserted] and it remains a major source of error for OCR engines.

* incomplete characters - this is the opposite of the above problem. Due to scanning or how the text was original typed, some parts of the characters can be missing (see the "B" in the above example). This could be due to low ink in the typewriter.
Also could happen if a character crosses a grid line - then when the grid line is removed, we are left with a gap in the character.