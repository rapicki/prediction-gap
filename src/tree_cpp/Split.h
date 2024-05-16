#pragma once
#include "Node.h"

class Split : public Node {
public:
  Split(int _id, float _value, std::string _feature, Node &_yes, Node &_no,
        Node &_missing);
  bool is_leaf() override;
  float descend(float cdf_dict, Node &prob_anc) override;
  void collect_thresholds(DataPoint &data_point, std::string perturbed_feature,
                          float current_ub, float result) override;
  float eval(DataPoint &x) override;

private:
  std::string feature;
  Node *yes;
  Node *no;
  Node *missing;
};
