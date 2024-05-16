#include "Split.h"

Split::Split(int _id, float _value, std::string _feature, Node &_yes, Node &_no,
             Node &_missing) {
  id = _id;
  value = _value;
  feature = _feature;
  yes = &_yes;
  no = &_no;
  missing = &_missing;
}

bool Split::is_leaf() { return false; }

float Split::descend(float cdf_dict, Node &prob_anc) {
  // Implement descend if needed
  return 0.0f;
}

void Split::collect_thresholds(DataPoint &data_point,
                               std::string perturbed_feature, float current_ub,
                               float result) {
  // Implement collect_thresholds if needed
}

float Split::eval(DataPoint &x) {
  if (x.find(feature) != x.end()) {
    if (x.at(feature) < value) {
      return yes->eval(x);
    } else {
      return no->eval(x);
    };
  } else {
    return missing->eval(x);
  };
}
