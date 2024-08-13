#pragma once
#include "Node.h"

class Leaf : public Node {
public:
  Leaf(int _id, float _value);
  bool is_leaf() override;
  float descend(CdfDict cdf_dict, CurrentPath *prob_anc,
                float (*func)(CdfDict cdf_dict, CurrentPath *prob_anc,
                              float val, float baseline, vector<Node *> trees,
                              float cum_prob),
                float baseline, vector<Node *> trees, float cum_prob,
                bool outer_loop) override;
  void collect_thresholds(DataPoint &data_point, std::string perturbed_feature,
                          float current_ub, float result) override;
  float eval(DataPoint &x) override;
};
