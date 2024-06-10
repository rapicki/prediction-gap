#include "Leaf.h"
#include <iomanip>
#include <stdlib.h>
using namespace std;

Leaf::Leaf(int _id, float _value) {
  id = _id;
  value = _value;
}

bool Leaf::is_leaf() { return true; }

void Leaf::collect_thresholds(DataPoint &data_point,
                              std::string perturbed_feature, float current_ub,
                              float result) {
  // Implement collect_thresholds if needed
}

float Leaf::eval(DataPoint &x) { 
//cout << "Leaf val " << setprecision(25)<< value << endl;
    return value; }

float Leaf::descend(CdfDict cdf_dict, CurrentPath *prob_anc,
                    float (*func)(CdfDict cdf_dict, CurrentPath *prob_anc,
                                  float val, float baseline,
                                  vector<Node *> trees),
                    float baseline, vector<Node *> trees) {
  return func(cdf_dict, prob_anc, value, baseline, trees);
};
