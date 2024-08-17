#include "NormalDistribution.h"

NormalDistribution::NormalDistribution(float mean_, float var_)
    : mean(mean_), var(var_) {}

float NormalDistribution::normalCDF(double value) {
  value = (value - mean) / var;
  double p = 0.5 * erfc(-value / sqrt(2));
  return static_cast<float>(p);
}

float NormalDistribution::get_value(float x) {
  // Implement get_value() based on your requirements
  return normalCDF(static_cast<double>(x));
}
