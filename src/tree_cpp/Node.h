#pragma once
#include "CurrentPath.h"
#include <iostream>
#include <map>
#include <string>
typedef std::map<std::string, float> DataPoint;

class Node {
public:
  int id;
  float value;

  virtual ~Node() {}
  virtual bool is_leaf() = 0;
  virtual float descend(CdfDict cdf_dict, CurrentPath *prob_anc,
                        float (*func)(CdfDict cdf_dict, CurrentPath *prob_anc,
                                      float val, float baseline,
                                      vector<Node *> trees, float cum_prob),
                        float baseline, vector<Node *> trees, float cum_prob,
                        bool outer_loop) = 0;
  virtual void collect_thresholds(DataPoint &data_point,
                                  std::string perturbed_feature,
                                  float current_ub, float result) = 0;
  virtual float eval(DataPoint &x) {
    std::cout << "called_virtual_Node" << std::endl;
    return 0.0f;
  };
};
