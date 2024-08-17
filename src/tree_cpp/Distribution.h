#pragma once

class Distribution {
public:
  virtual ~Distribution() {}
  virtual float get_value(float x) = 0;
};
