/*
naive.cpp

Laura Battle
Advanced Data Mining - HW 3 - Spring 2022
Naive Bayes Classifier
*/

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <string>
#include "naive.h"

//#define MAXATTR 1000

int main(int argc, char** argv)
{
  
  if(argc != 3)
  {
    PrintInstructions();
    exit(EXIT_SUCCESS);
  }

  std::string trainfile = argv[1];
  std::string testfile = argv[2];
  
  std::ifstream fin;
  fin.open(trainfile);
  if(!fin)
  {
    std::cerr << "\n\n***Error opening file " << trainfile << "\n\n";
    exit(EXIT_FAILURE);
  }

  Label PlusOne, MinusOne;

  GetLabels(PlusOne, MinusOne, fin);
  ReadTrainingData(PlusOne, MinusOne, fin);
  fin.close();
  
  PlusOne.Dump();
  MinusOne.Dump();
  
  
  return 0;
}


void PrintInstructions()
{
  std::cout << "\n\nWelcome to naive bayes.\nTo run the program "
            << "please provide a training file and a test file as shown below.\n\n"
            << "NaiveBayes <trainingfile> <testfile>\n\n";
}

void ReadTrainingData(Label& PlusOne, Label& MinusOne, std::ifstream& fin)
{
  fin.seekg(0, std::ios::beg);
  std::string line, label;

  while(std::getline(fin, line))
  {
    std::istringstream iss(line);
    iss >> label;
    if(label == PlusOne.GetLabel())
    {
      PlusOne.AddInstance();
      ReadData(PlusOne, iss);
    }
    else if(label == MinusOne.GetLabel())
    {
      MinusOne.AddInstance();
      ReadData(MinusOne, iss);
    }
    else
    {
      std::cout << "\n\n***Error, problem with label matching. Label is: "
		<< label << "\n\n";
      fin.close();
      exit(EXIT_FAILURE);
    }
  }
  PlusOne.AddZerosMakeFractions();
  MinusOne.AddZerosMakeFractions();
}


void ReadData(Label& Label, std::istringstream& iss)
{
  int len, i;
  std::string pair;
  int attribute;

  int category;
  while(iss >> pair)
  {
    sscanf(pair.c_str(), "%d:%d", &attribute, &category);
    Label.AddTrainingPoint(attribute, category);
  }
}

void GetLabels(Label& One, Label& Two, std::ifstream& fin)
{
  bool incomplete = true;
  std::string label1, label2, test;
  fin >> label1;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  while(incomplete && (fin >> test))
  {
    if(test != label1)
    {
      label2 = test;
      incomplete = false;
    }
    else
    {
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
  }
  One.AddLabel(label1);
  Two.AddLabel(label2);
}


/*

LABEL OBJECT IMPLEMENTATIONS

 */

int Label::GetTotal () const
{
  return total_instances;
}

std::string Label::GetLabel () const
{
  return label;
}

void Label::AddLabel (std::string name)
{
  label = name;
}

void Label::AddInstance ()
{
  ++total_instances;
}

void Label::PrintLabel()
{
  std::cout << "\nLabel: " << label << '\n';
}

void Label::AddTrainingPoint(int attribute, int category)
{
  map_itr data_itr;
  attribute_itr attr_itr;
  
  data_itr = data.find(attribute);
  if(data_itr == data.end())
  {
    std::unordered_map<int, double> new_attribute;
    new_attribute.insert(std::make_pair(category, 1));
    data.insert(std::make_pair(attribute, new_attribute));
  }
  else
  {
    attr_itr = (data_itr->second).find(category);
    if(attr_itr == (data_itr->second.end()))
    {
      (data_itr->second).insert(std::make_pair(category, 1));
    }
    else
    {
      attr_itr->second += 1;
    }
  }
  
}
void Label::Dump()
{
  std::cout << "\n\nData dump for label " << label << "\n\n";
  map_itr data_itr;
  attribute_itr attr_itr;
  for(data_itr = data.begin(); data_itr != data.end(); ++data_itr)
  {
    std::cout << "\n\nattribute index: " << data_itr->first
	      << "\n\n\tcategory\tcount";
    for(attr_itr = (data_itr->second).begin();
	attr_itr != (data_itr->second).end();
	++attr_itr)
    {
      std::cout << "\n\t" << attr_itr->first << "\t\t" << attr_itr->second;
    }
  }
  std::cout << "\n\n\n";
}

void Label::AddZerosMakeFractions()
{
  int difference;
  int acc;
  double diff;
  map_itr data_itr;
  attribute_itr attr_itr;
  for(data_itr = data.begin(); data_itr != data.end(); ++data_itr)
  {
    acc = 0;
    for(attr_itr = (data_itr->second).begin();
        attr_itr != (data_itr->second).end();
        ++attr_itr)
    {
      acc += attr_itr->second;
      attr_itr->second /= total_instances;
    }
    if((total_instances - acc) != 0)
    {
      (data_itr->second).insert(std::make_pair(0, (total_instances - acc)));
      attr_itr = (data_itr->second).find(0);
      attr_itr->second /= total_instances;
    }
  }
}
