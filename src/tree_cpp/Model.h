#pragma once
#include <functional>
#include <map>
#include <string>
#include "Distribution.h"

typedef std::map<std::string, float> DataPoint;
typedef std::map<std::string, Distribution*> CdfDict;

class Model {
public:
  virtual float expected_diff_squared(float cdf_dict, float baseline) = 0;
  virtual float expected_single_feature(float data_point,
                                        std::string perturbed_feature,
                                        float cdf, float f) = 0;
  virtual float eval(float x) = 0;
};
