/*
naive.h

Laura Battle
Advanced Data Mining - HW 3 - Spring 2022
Naive Bayes Classifier
*/

#pragma once
#include <vector>
#include <unordered_map>
#include <sstream>

typedef std::unordered_map<int, double> attribute;
typedef std::unordered_map<int, attribute> map_attributes;
typedef std::unordered_map<int, double>::iterator attribute_itr;
typedef std::unordered_map<int, attribute>::iterator map_itr;

class Label
{
 public:
  int GetTotal() const;
  void AddLabel(std::string);
  void PrintLabel();
  std::string GetLabel() const;
  void AddTrainingPoint(int, int);
  void Dump();
  void AddInstance();
  void AddZerosMakeFractions();
  
 private:
  int total_instances = 0;
  std::string label;
  // in constructor  memset(attribute_count, 0, MAXATTR);
  map_attributes data;
};

void PrintInstructions();
void ReadTrainingData(Label&, Label&, std::ifstream&);
void GetLabels(Label&, Label&, std::ifstream&);
void ReadData(Label&, std::istringstream&);

