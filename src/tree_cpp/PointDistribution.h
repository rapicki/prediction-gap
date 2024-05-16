#pragma once
#include "Distribution.h"

class PointDistribution : public Distribution {
public:
  PointDistribution(float point_value);
  float get_value() override;

private:
  float value;
};
