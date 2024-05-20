#pragma once
#include "Model.h"
#include <cstdlib>
#include <functional>
#include <map>
#include <stdlib.h>
#include <string>
#include <vector>
using namespace std;
#include "Distribution.h"
typedef std::map<std::string, vector<float>> PathDict;

class CurrentPath {
  PathDict lb, ub;

public:
  float last_lb(string feature);
  float last_ub(string feature);
  void descend_left(string feature, float t);
  void descend_right(string feature, float t);
  void revert_left(string feature);
  void revert_right(string feature);
  tuple<float, float> current_interval(string feature);
  float prob(CdfDict &cdf_dict);
};
