Optical Character Recognition (OCR) software can be used to automatically transcribe printed text. (OCR is not designed to work with hand written text - whether printed or cursive) Probably the most common open source OCR engine is Tesseract.
Installing Tesseract on Ubuntu is pretty easy (https://help.ubuntu.com/community/OCR ) and is probably petty easy in other operating systems. Running Tesseract is also pretty easy - just give it an input file and an output file (stdout to print to screen). For example ::

    tesseract greg.jpg stdout

Tesseract has already been taught a number of fonts. The challenge is teaching tesseract new fonts. A couple of good websites exist which outline the process (http://michaeljaylissner.com/posts/2012/02/11/adding-new-fonts-to-tesseract-3-ocr-engine/ ) -
the process is pretty tedious but in theory automatable - you just copy each line of the process into a script. However, in practice, things don't go so well. Strange errors turn up - the one I am currently dealing with is ::

    Bad properties for index 3, char A: 0,255 0,255 0,0 0,0 0,0

Even when my unicharset file and my box file look completely correct. I don't remember getting this error when I first started working with Tesseract but I'm not sure. I've seen several posts about this issue but no one has given an answer. So the time being, I would not recommend using Tesseract if you need to teach it new fonts.

But let's look at how one would use Tesseract. One possibility is to use the command line version and the other possibility is through the API. There are a couple of Python wrappers for Tesseract - the best one is Tesserpy (I found at least one other which actually just makes a call to the command line version of Tesseract.)
The problem with the command line version is that you cannot get easy access to the confidence values (a measure of how good Tesseract thinks its transcription is). In order to do so you need to get the ouput in HOCR format - which is XML and horrible to deal with. Also the confidence values are just on a per word level - not character by character. Tesserpy allows you to get the confidence values easily and a character by character level.
Also if you are doing image manipulation you don't have to save the image to file before passing it to Tesserpy.

Once you've got Tesserpy installed (not a completely easy process), follow the instructions on the github page to do the transcription. Use ".symbols()" instead of ".words()" to iterate over the results on a character level. So