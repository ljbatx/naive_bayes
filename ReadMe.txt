ReadMe.txt

Laura Battle
Advanced Data Mining - Homework 3 - Spring 2022

*************************************************
DESCRIPTION
*************************************************

Two programs are made with the "make" command. One creates an executable from
naive.cpp and one from matchingnaive.cpp. They share a header file and makefile.

The program NaiveBayes is my implementation of the Naive Bayes classifier. The
program NaiveBayes2 is another implementation that creates outputs that match
what was given to us as example outputs. The differences in code between these
two files is described in the "Testing" section of the accompanying report.

The outputs for breastcancer.train and breastcancer.test are identical for both
programs. The outputs for led.train and led.test are different.

*************************************************
COMPILE
*************************************************

To compile the programs type "make"

You can also type "make" but this will run a clean command which you might not want.

*************************************************
EXECUTE
*************************************************

To run the program type:

./NaiveBayes <trainingfile> <testfile>

or

./NaiveBayes2 <trainingfile> <testfile>

It will output <true_positive> <false_negative> <false_positive> <true_negative>

***********************************************
OPTIONS
***********************************************

If you would like to see the contents of the label data structures, you can
uncomment lines 60 and 61 in naive.cpp or matchingnaive.cpp
and recompile and rerun.
