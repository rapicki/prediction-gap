#include "Split.h"
#include <cmath>
#include <iomanip>
#include <stdlib.h>
using namespace std;
#include <limits>

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
};
float get_next(float x) {
  return nexttoward(x, -std::numeric_limits<float>::infinity());
};
float Split::interval_prob(Distribution *d, tuple<float, float> intervals) {
  return max((d->get_value(get_next(get<1>(intervals))) -
              d->get_value(get_next(get<0>(intervals)))),
             0.0f);
};

float Split::descend(CdfDict cdf_dict, CurrentPath *prob_anc,
                     float (*func)(CdfDict cdf_dict, CurrentPath *prob_anc,
                                   float val, float baseline,
                                   vector<Node *> trees),
                     float baseline, vector<Node *> trees) {

  if (cdf_dict.find(feature) != cdf_dict.end()) {
    float cond_prob = interval_prob(cdf_dict.at(feature),
                                    prob_anc->current_interval(feature));
    if (cond_prob == 0.0) {
      cout << "Point prob 0" << endl;
      return 0.0;
    };
    prob_anc->descend_left(feature, value);
    float left_prob = interval_prob(cdf_dict.at(feature),
                                    prob_anc->current_interval(feature)) /
                      cond_prob;
    float result = 0.0f;
    if (left_prob > 1e-12) {
      result +=
          left_prob * yes->descend(cdf_dict, prob_anc, func, baseline, trees);
    };
    prob_anc->revert_left(feature);
    if (1.0 - left_prob >= 1e-12) {
      prob_anc->descend_right(feature, value);
      result += (1.0f - left_prob) *
                no->descend(cdf_dict, prob_anc, func, baseline, trees);
      prob_anc->revert_right(feature);
    };

    return result;
  } else {
    cout << "missing  " << feature << endl;
    return missing->descend(cdf_dict, prob_anc, func, baseline, trees);
  };
};
