#Makefile
#
#Laura Battle
#Advanced Data Mining - HW 3 Spring 2022
#Naive Bayes Classifier

CXX= g++
CXXFLAGS= -std=c++11 -Wall -pedantic

all: clean NaiveBayes NaiveBayes2

NaiveBayes: naive.cpp naive.h
	$(CXX) $(CXXFLAGS) naive.cpp -o NaiveBayes -w

NaiveBayes2: matchingnaive.cpp naive.h
	$(CXX) $(CXXFLAGS) matchingnaive.cpp -o NaiveBayes2 -w

.PHONY: clean

clean:
	rm -rf *.o core.* a.out *~ \#*\# NaiveBayes NaiveBayes2
