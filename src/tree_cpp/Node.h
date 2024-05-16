#pragma once
#include <string>
#include <map>
#include <iostream>

typedef std::map<std::string, float> DataPoint;

class Node {
public:
    int id;
    float value;

    virtual ~Node() {}
    virtual bool is_leaf() = 0;
    virtual float descend(float cdf_dict, Node& prob_anc) = 0;
    virtual void collect_thresholds(DataPoint& data_point, std::string perturbed_feature, float current_ub, float result) = 0;
    virtual float eval(DataPoint& x) { std::cout << "called_virtual_Node" << std::endl; return 0.0f; };
};

