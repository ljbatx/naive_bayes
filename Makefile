#Makefile


CXX= g++
CXXFLAGS= -std=c++11 -Wall -pedantic -g

all: clean NaiveBayes

NaiveBayes: naive.cpp naive.h
	$(CXX) $(CXXFLAGS) naive.cpp -o NaiveBayes -w

.PHONY: clean

clean:
	rm -rf *.o core.* a.out *~ \#*\# *.dSYM NaiveBayes
