#Makefile
#
#Laura Battle
#Advanced Data Mining - HW 3 Spring 2022
#Naive Bayes Classifier

CXX= g++
CXXFLAGS= -std=c++11 -Wall -pedantic -g

all: clean NaiveBayes

NaiveBayes: naive.cpp naive.h
	$(CXX) $(CXXFLAGS) naive.cpp -o NaiveBayes -w

.PHONY: clean

clean:
	rm -rf *.o core.* a.out *~ \#*\# *.dSYM
