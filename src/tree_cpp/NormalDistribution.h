#pragma once
#include "Distribution.h"
#include <cmath>

class NormalDistribution : public Distribution {
public:
  NormalDistribution(float mean_, float var_);
  float get_value(float x) override;

private:
  float mean;
  float var;
  float normalCDF(double value);
};
