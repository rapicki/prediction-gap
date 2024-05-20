#include "CurrentPath.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <ostream>
#include <set>
#include <tuple>
#include <vector>

float CurrentPath::last_lb(string feature) {
  if (lb.find(feature) != lb.end()) {
    return lb.at(feature).back();
  } else {
    // return -1 * INFINITY;
    return -1 * numeric_limits<float>::max();
  };
};

float CurrentPath::last_ub(string feature) {
  if (ub.find(feature) != ub.end()) {
    return ub.at(feature).back();
  } else {
    // return 1 * INFINITY;
    return numeric_limits<float>::max();
  };
};

void CurrentPath::descend_left(string feature, float t) {
  float new_ = min(last_ub(feature), t);
  if (ub.find(feature) == ub.end()) {
    ub.insert({feature, vector<float>()});
  };
  ub.at(feature).push_back(new_);
};

void CurrentPath::descend_right(string feature, float t) {
  float new_ = max(last_lb(feature), t);
  if (lb.find(feature) == lb.end()) {
    lb.insert({feature, vector<float>()});
  };
  lb.at(feature).push_back(new_);
};

void CurrentPath::revert_left(string feature) {
  ub.at(feature).pop_back();
  if (ub.at(feature).size() == 0) {
    ub.erase(feature);
  }
};

void CurrentPath::revert_right(string feature) {
  lb.at(feature).pop_back();
  if (lb.at(feature).size() == 0) {
    lb.erase(feature);
  }
};

tuple<float, float> CurrentPath::current_interval(string feature) {
  return make_tuple(last_lb(feature), last_ub(feature));
};
float CurrentPath::prob(CdfDict &cdf_dict) {
  vector<string> keys;
  set<string> unique_keys;
  for (auto const &element : ub) {
    keys.push_back(element.first);
  }
  for (auto const &element : lb) {
    keys.push_back(element.first);
  }
  for (auto name : keys) {
    unique_keys.insert(name);
  };
  float prob = 1.0;
  for (auto f : unique_keys) {
    if (cdf_dict.find(f) != cdf_dict.end()) {
      prob *= max(0.0f, cdf_dict.at(f)->get_value(last_ub(f)) -
                            cdf_dict.at(f)->get_value(last_lb(f)));
    };
  }
  return prob;
};
