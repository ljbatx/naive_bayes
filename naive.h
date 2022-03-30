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
  bool AddTrainingPoint(int, int);
  void Dump();
  void AddInstance();
  void AddZerosMakeFractions();
  double GetLikelihood(int, int);
  void AddCategory(int, int);
  void GetZeroAttributes(std::vector<int>&, std::unordered_map<int, int>&);
  int GetMaxAttributes();

  
 private:
  int max_attributes = 0;
  int total_instances = 0;
  std::string label;
  // in constructor  memset(attribute_count, 0, MAXATTR);
  map_attributes data;
};

class Classify
{
public:
  void PrintResults();
  void TP();
  void FN();
  void FP();
  void TN();
  
 private:
  
  int true_positive = 0;
  int false_negative = 0;
  int false_positive = 0;
  int true_negative = 0;
};

void PrintInstructions();
void ReadTrainingData(Label&, Label&, std::ifstream&);
void GetLabels(Label&, Label&, std::ifstream&);
void ReadData(Label&, Label&, std::istringstream&);
void Classification(Classify&, std::ifstream&, Label&, Label&);
void Predict(Classify&, std::istringstream&, std::string, Label&, Label&);
