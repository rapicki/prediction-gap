#pragma once
#include "Node.h"

class Leaf : public Node {
public:
  Leaf(int _id, float _value);
  bool is_leaf() override;
  float descend(float cdf_dict, Node &prob_anc) override;
  void collect_thresholds(DataPoint &data_point, std::string perturbed_feature,
                          float current_ub, float result) override;
  float eval(DataPoint &x) override;
};
