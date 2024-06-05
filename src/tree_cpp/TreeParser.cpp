#include "TreeParser.h"
#include "Node.h"
#include <ostream>

TreeParser::TreeParser(string _filename, float _bias) {
  bias = _bias;
  parse_tree(_filename);
};

void TreeParser::parse_tree(string filename) {
  ifstream file(filename);
  regex header("booster\\[\\d+]:");
  smatch sm;
  if (file.is_open()) {
    string line;
    while (std::getline(file, line)) {
      if (regex_match(line, sm, header)) {
        Node *tmp = parse_subtree(file);

        // Node *tmp_ptr = &tmp;
        trees.push_back(tmp);
      } else {
        throw invalid_argument("recieved wrong tree file 8");
      };
    }
    file.close();
  };
};
Split_params TreeParser::get_split_params(string line) {
  regex id("\\s*(\\d+):");
  regex feature("\\[(\\w+)<([^\\]]+)]\\s*");
  regex values("yes=(\\d+),no=(\\d+),missing=(\\d+)");
  smatch sm_id, sm_feature, sm_value;
  bool id_match, feature_match, value_match;

  id_match = regex_search(line, sm_id, id);
  feature_match = regex_search(line, sm_feature, feature);
  value_match = regex_search(line, sm_value, values);

  if (id_match && feature_match && value_match) {
    Split_params split;

    if (sm_id.size() == 2) {
      split.id = stoi(sm_id[1].str());
    } else {
      throw invalid_argument("recieved wrong tree file");
    };

    if (sm_feature.size() == 3) {
      split.feature = sm_feature[1].str();
      split.value = stof(sm_feature[2].str());

    } else {
      throw invalid_argument("recieved wrong tree file");
    };

    if (sm_value.size() == 4) {
      split.yes = stoi(sm_value[1].str());
      split.no = stoi(sm_value[2].str());
      split.missing = stoi(sm_value[3].str());

    } else {
      throw invalid_argument("recieved wrong tree file");
    };
    return split;
  } else {
    Split_params split;
    split.id = -1;
    return split;
  };
};

Leaf_params TreeParser::get_leaf_params(string line) {
  regex leaf("\\s*(\\d+):leaf=([^,]+)");
  bool leaf_match;
  smatch sm_leaf;
  Leaf_params l;
  leaf_match = regex_search(line, sm_leaf, leaf);
  if (leaf_match && sm_leaf.size() == 3) {
    l.id = stoi(sm_leaf[1].str());
    l.value = stof(sm_leaf[2].str());
  } else {
    l.id = -1;
  };
  return l;
};

Node *TreeParser::parse_subtree(ifstream &file) {

  if (file.is_open()) {
    string line;
    getline(file, line);
    Split_params s = get_split_params(line);
    if (s.id != -1) {
      Node *s1 = parse_subtree(file);
      Node *s2 = parse_subtree(file);
      Node *missing;
      if (s.missing == s2->id) {
        missing = s2;
      } else if (s.missing == s1->id) {
        missing = s1;
      } else {
        throw invalid_argument("recieved wrong tree file xd");
      };

      if (s.yes != s1->id or s.no != s2->id) {
        throw invalid_argument("recieved wrong tree file");
      }
      return new Split(s.id, s.value, s.feature, *s1, *s2, *missing);
    } else {
      Leaf_params l;
      l = get_leaf_params(line);
      return new Leaf(l.id, l.value);
    };
  } else {
    throw invalid_argument("recieved wrong tree file");
  };
};

float TreeParser::eval_on_array(list<py::array_t<float>> perturbed,
                                const py::array_t<float> input1,
                                list<string> &input2) {
  int count = 0.0f;
  float sum = 0.0f;
  float base = eval_numpy(input1, input2);
  for (auto i : perturbed) {
    sum += pow(eval_numpy(i, input2) - base, 2.0);
    count += 1.0f;
  };
  return sum/count;
};
float TreeParser::eval(DataPoint &x) {
  float results = 0.0;
  for (auto t : trees) {
    results += t->eval(x);
  };
  results += bias;
  return results;
};

float TreeParser::eval_numpy(const pybind11::array_t<float> input1,
                             list<string> &input2) {
  pybind11::buffer_info buf1 = input1.request();

  if (buf1.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be one");
  }
  float *ptr1 = static_cast<float *>(buf1.ptr);
  DataPoint d = convert_to_data_point(input2, ptr1);
  return eval(d);
}

DataPoint TreeParser::convert_to_data_point(list<string> &names,
                                            float *values) {
  DataPoint d;
  int i = 0;
  for (auto n : names) {
    d.insert({n, values[i]});
    i++;
  }
  return d;
};
float value_function(CdfDict cdf_dict, CurrentPath *prob_anc, float val,
                     float baseline, vector<Node *> trees) {
  return val;
}

float contrib_outer(CdfDict cdf_dict, CurrentPath *prob_anc, float val,
                    float baseline, vector<Node *> trees) {
  float inner_sum = -1 * baseline * 2.0;
  for (auto inner_tree : trees) {
    inner_sum += inner_tree->descend(cdf_dict, prob_anc, &value_function,
                                     baseline, trees);
  };
  return val * inner_sum;
};

CdfDict construct_cdf_dict(DataPoint &data_point, list<string> &perturbed_names,
                           float std) {
  CdfDict cdf_dict;
  for (auto feature : data_point) {
    if (find(perturbed_names.begin(), perturbed_names.end(), feature.first) !=
        perturbed_names.end()) {
      NormalDistribution *norm = new NormalDistribution(feature.second, std);
      cdf_dict.insert({feature.first, norm});

    } else {
      PointDistribution *point = new PointDistribution(feature.second);
      cdf_dict.insert({feature.first, point});
    };
  };
  return cdf_dict;
};

float TreeParser::expected_diff_squared(const pybind11::array_t<float> input1,
                                        list<string> &input2,
                                        list<string> &perturbed_names,
                                        float std) {

  pybind11::buffer_info buf1 = input1.request();

  if (buf1.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be one");
  }
  float *ptr1 = static_cast<float *>(buf1.ptr);
  DataPoint d = convert_to_data_point(input2, ptr1);
  float baseline = eval(d);
  baseline -= bias;

  CdfDict cdf_dict = construct_cdf_dict(d, perturbed_names, std);

  float result = baseline * baseline;
  // cout<<"result"<<result<<endl;

  for (auto t : trees) {
    CurrentPath *path_pointer = new CurrentPath();
    float tmp =
        t->descend(cdf_dict, path_pointer, &contrib_outer, baseline, trees);
    result += tmp;

    // if (result < 0){cout<<result<<endl;};
    delete path_pointer;
    // cout<<tmp<<endl;
  }
  return result;
};
