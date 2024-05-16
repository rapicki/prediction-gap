#include "Leaf.h"

Leaf::Leaf(int _id, float _value) {
  id = _id;
  value = _value;
}

bool Leaf::is_leaf() { return true; }

float Leaf::descend(float cdf_dict, Node &prob_anc) { return 1.0f; }

void Leaf::collect_thresholds(DataPoint &data_point,
                              std::string perturbed_feature, float current_ub,
                              float result) {
  // Implement collect_thresholds if needed
}

float Leaf::eval(DataPoint &x) { return value; }
