#include "PointDistribution.h"
#include "Split.h"
#include <iomanip>
using namespace std;

PointDistribution::PointDistribution(float point_value) : value(point_value) {}

float PointDistribution::get_value(float x) {
  if (x < value) {
    return 0.0f;
  } else {
    return 1.0f;
  };
}
float PointDistribution::get_tresh(){
    return value;
};
