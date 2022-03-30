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
#include <cmath>
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

  Classify Training, Testing;

  Classification(Training, fin, PlusOne, MinusOne);
  fin.close();
  
  fin.open(testfile);
  if(!fin)
  {
    std::cerr << "\n\n***Error opening file " << testfile << "\n\n";
    exit(EXIT_FAILURE);
  }

  Classification(Testing, fin, PlusOne, MinusOne);
  fin.close();

  // PlusOne.Dump();                      
  // MinusOne.Dump();
  
  Training.PrintResults();
  Testing.PrintResults();
  std::cout << "\n\n";
  
  return 0;
}

void Classification(Classify& Dataset, std::ifstream& fin, Label& Pos, Label& Neg)
{
  fin.clear();
  fin.seekg(0, std::ios::beg);
  
  std::string line, label;
  
  while(std::getline(fin, line))
  {
    std::istringstream iss(line);
    iss >> label;
    Predict(Dataset, iss, label, Pos, Neg);
  }
}

void Predict(Classify& Dataset, std::istringstream& iss, std::string tru_label, Label& Pos, Label& Neg)
{
  std::vector<int> zero_attributes;
  std::unordered_map<int, int> attributes_accounted_for;
  std::string data_point;
  int attribute, category, max, item;
  long double likelihood_pos, likelihood_neg, test;
  likelihood_pos = likelihood_neg = 0;
  
  while(iss >> data_point)
  {
    sscanf(data_point.c_str(), "%d:%d", &attribute, &category);
    attributes_accounted_for.insert(std::make_pair(attribute, category));
    
    likelihood_pos += Pos.GetLikelihood(attribute, category);
    likelihood_neg += Neg.GetLikelihood(attribute, category);
  }
  
  if(Pos.GetMaxAttributes() > attributes_accounted_for.size())
  {
    Pos.GetZeroAttributes(zero_attributes, attributes_accounted_for);
    max = zero_attributes.size();  
    for(int i = 0; i < max; ++i)
    {
      likelihood_pos += Pos.GetLikelihood(zero_attributes[i], 0);
      likelihood_neg += Neg.GetLikelihood(zero_attributes[i], 0);
    }
  }
  
  if(likelihood_pos > likelihood_neg)
  {
    if(tru_label == Pos.GetLabel())
    {
      Dataset.TP();
    }
    else
    {
      Dataset.FP();
    }
  }
  else if(likelihood_neg > likelihood_pos)
  {
    if(tru_label == Neg.GetLabel())
    {
      Dataset.TN();
    }
    else
    {
      Dataset.FN();
    }
  }
  else
  {
    std::cout << "\nError: likelihoods equal\n";
    std::cout << "\npositive: " << likelihood_pos
              << "\nnegative: "	<< likelihood_neg;
  }	      
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
      ReadData(PlusOne, MinusOne, iss);
    }
    else if(label == MinusOne.GetLabel())
    {
      MinusOne.AddInstance();
      ReadData(MinusOne, PlusOne, iss);
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


void ReadData(Label& TheLabel, Label& Unlabel, std::istringstream& iss)
{
  int len, i;
  std::string pair;
  int attribute;

  int category;
  while(iss >> pair)
  {
    sscanf(pair.c_str(), "%d:%d", &attribute, &category);
    if(TheLabel.AddTrainingPoint(attribute, category))
    {
      Unlabel.AddCategory(attribute, category);
    }
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
  if(label1 == "+1" && label2 == "-1")
  {
    One.AddLabel(label1);
    Two.AddLabel(label2);
  }
  else if(label1 == "-1" && label2 == "+1")
  {
    One.AddLabel(label2);
    Two.AddLabel(label1);
  }
  else
  {
    std::cout << "\n\nExpecting labels of +1 and -1\n\n";
  }
	  
}


/*
CLASSIFY OBJECT IMPLEMENTATIONS
*/

void Classify::PrintResults()
{
  std::cout << '\n'
	    << true_positive << ' '
	    << false_negative << ' '
	    << false_positive << ' '
	    << true_negative;
}

void Classify::TP()
{
  ++true_positive;
}

void Classify::FN()
{
  ++false_negative;
}

void Classify::FP()
{
  ++false_positive;
}

void Classify::TN()
{
  ++true_negative;
}

/*

LABEL OBJECT IMPLEMENTATIONS

 */

double Label::GetLikelihood (int attribute, int category)
{
  double return_val = 0;
  map_itr data_itr;
  attribute_itr attr_itr;
  data_itr = data.find(attribute);
  if(data_itr == data.end())
  {
    std::cout << "\nError! GetLikelihood for nonexistent attribute\n";
  }
  else
  {
    attr_itr = (data_itr->second).find(category);
    if(attr_itr == (data_itr->second).end())
    {
      /*
      std::cout << "\nIgnoring novel category " << category
		<< " for attribute: " << attribute << "\n";
      */
    }
    else
    {
      return_val = attr_itr->second;
    }
  }

  //  std::cout << "\nreturning " << return_val;
  return return_val;
}

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

int Label::GetMaxAttributes ()
{
  return max_attributes;
}

void Label::AddInstance ()
{
  ++total_instances;
}

void Label::PrintLabel()
{
  std::cout << "\nLabel: " << label << '\n';
}

bool Label::AddTrainingPoint(int attribute, int category)
{
  bool added = false;
  map_itr data_itr;
  attribute_itr attr_itr;
  
  data_itr = data.find(attribute);
  if(data_itr == data.end())
  {
    std::unordered_map<int, double> new_attribute;
    new_attribute.insert(std::make_pair(category, 1));
    data.insert(std::make_pair(attribute, new_attribute));
    added = true;
  }
  else
  {
    attr_itr = (data_itr->second).find(category);
    if(attr_itr == (data_itr->second.end()))
    {
      (data_itr->second).insert(std::make_pair(category, 1));
      added = true;
    }
    else
    {
      attr_itr->second += 1;
    }
  }
  return added;
}

void Label::AddCategory(int attribute, int category)
{
  map_itr data_itr;
  attribute_itr attr_itr;

  data_itr = data.find(attribute);
  if(data_itr == data.end())
  {
    std::unordered_map<int, double> new_attribute;
    new_attribute.insert(std::make_pair(category, 0));
    data.insert(std::make_pair(attribute, new_attribute));
  }
  else
  {
    attr_itr = (data_itr->second).find(category);
    if(attr_itr == (data_itr->second.end()))
    {
      (data_itr->second).insert(std::make_pair(category, 0));
    }
    else
    {
      std::cout << "\n\nError - reached else statement shouldn't have in Label::AddCategory\n";
      exit(EXIT_FAILURE);
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
	      << "\n\n\tcategory\tvalue";
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
      if(attr_itr->second != 0)
        attr_itr->second = std::log(attr_itr->second);
    }
    if((total_instances - acc) != 0)
    {
      (data_itr->second).insert(std::make_pair(0, (total_instances - acc)));
      attr_itr = (data_itr->second).find(0);
      attr_itr->second /= total_instances;
      if(attr_itr->second != 0)
        attr_itr->second = std::log(attr_itr->second);
    }
  }
  max_attributes = data.size();
}

void Label::GetZeroAttributes(std::vector<int>& zeros, std::unordered_map<int, int>& used)
{
  map_itr data_itr;
  attribute_itr attr_itr;
  std::unordered_map<int, int>::iterator used_itr;
  
  for(data_itr = data.begin(); data_itr != data.end(); ++data_itr)
  {
    used_itr = used.find(data_itr->first);
    if(used_itr == used.end())
    {
      zeros.push_back(data_itr->first);
    }
  }
}

