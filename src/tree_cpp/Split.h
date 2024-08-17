#pragma once
#include "Node.h"

class Split : public Node {
public:
  Split(int _id, float _value, std::string _feature, Node &_yes, Node &_no,
        Node &_missing);
  bool is_leaf() override;
  float descend(CdfDict cdf_dict, CurrentPath *prob_anc,
                float (*func)(CdfDict cdf_dict, CurrentPath *prob_anc,
                              float val, float baseline, vector<Node *> trees,
                              float cum_prob),
                float baseline, vector<Node *> trees, float cum_prob,
                bool outer_loop) override;
  float interval_prob(Distribution *d, tuple<float, float> intervals);
  void collect_thresholds(DataPoint &data_point, std::string perturbed_feature,
                          float current_ub, float result) override;
  float eval(DataPoint &x) override;

private:
  std::string feature;
  Node *yes;
  Node *no;
  Node *missing;
};
