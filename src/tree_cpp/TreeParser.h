#pragma once
#include "Leaf.h"
#include "Node.h"
#include "Split.h"
#include "Leaf.h"
// #include "tree_helpers.cpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <ostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;

struct Split_params {
  int id;
  string feature;
  float value;
  int yes;
  int no;
  int missing;
};

struct Leaf_params {
  int id;
  float value;
};

class TreeParser {
  vector<Node *> trees;
  float bias;
  void parse_tree(string filenam);
  Node *parse_subtree(ifstream &file);
  Leaf_params get_leaf_params(string line);
  Split_params get_split_params(string line);

public:
  TreeParser(string _filename, float _bias);
  float eval(DataPoint &x);
  float eval_numpy(const py::array_t<float> input1, list<string> &input2);
  DataPoint convert_to_data_point(list<string> &names, float *values);
};
