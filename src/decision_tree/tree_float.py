import functools
import math
import operator
import re
from abc import ABC
from collections import defaultdict
from operator import itemgetter
import numpy as np
from pathlib import Path
import json
import pandas as pd


class Model:
    # cdf_dict is of type feature -> cumulative distribution function
    def expected_diff_squared(self, cdf_dict: dict, baseline):
        raise NotImplementedError('')

    def expected_single_feature(self, data_point, perturbed_feature, cdf, f):
        raise NotImplementedError('')

    def eval(self, x):
        raise NotImplementedError('')


class _CurrentPath:
    def __init__(self):
        self.lb = defaultdict(list)
        self.ub = defaultdict(list)

    def last_lb(self, feature):
        return self.lb[feature][-1] if feature in self.lb else float('-inf')

    def last_ub(self, feature):
        return self.ub[feature][-1] if feature in self.ub else float('inf')

    def descend_left(self, feature, t):
        t = min(self.last_ub(feature), t)
        self.ub[feature].append(t)

    def descend_right(self, feature, t):
        t = max(self.last_lb(feature), t)
        self.lb[feature].append(t)

    def revert_left(self, feature):
        self.ub[feature].pop()
        if len(self.ub[feature]) == 0:
            self.ub.pop(feature)

    def revert_right(self, feature):
        self.lb[feature].pop()
        if len(self.lb[feature]) == 0:
            self.lb.pop(feature)

    def current_interval(self, feature):
        return self.last_lb(feature), self.last_ub(feature)

    def prob(self, cdf_dict):
        features = set(self.lb.keys()).union(set(self.ub.keys()))
        prob = 1.0
        for f in features:
            if f in cdf_dict:
                prob *= max(0, cdf_dict[f](self.last_ub(f)) - cdf_dict[f](self.last_lb(f)))
        return prob


class Node(ABC):
    def is_leaf(self):
        raise NotImplementedError('')

    def descend(self, cdf_dict: dict, prob_anc: _CurrentPath, leaf_contrib_fun):
        raise NotImplementedError('')

    def collect_thresholds(self, data_point, perturbed_feature, current_ub, result):
        raise NotImplementedError('')

    def eval(self, x):
        raise NotImplementedError('')


class Leaf(Node):
    def __init__(self, val):
        self.val = val

    def is_leaf(self):
        return True

    def descend(self, cdf_dict: dict, prob_anc: dict, leaf_contrib_fun):
        return leaf_contrib_fun(cdf_dict, prob_anc, self.val)

    def eval(self, x):
        return self.val

    def collect_thresholds(self, data_point, perturbed_feature, current_lb, result):
        result.append((current_lb, self.val))


def _interval_prob(cdf, interval):
    # we mean a closed-open [a,b) interval
    a, b = interval
    return max(cdf(b - 1.0e-12) - cdf(a - 1.0e-12), 0.0)


class Split(Node):
    def __init__(self, feature, threshold: float, yes: Node, no: Node, missing: Node):
        self.feature = feature
        self.threshold = threshold
        self.yes = yes
        self.no = no
        self.missing = missing

    def is_leaf(self):
        return False

    def descend(self, cdf_dict: dict, prob_anc: _CurrentPath, leaf_contrib_fun):
        if self.feature in cdf_dict:
            cond_prob = _interval_prob(cdf_dict[self.feature], prob_anc.current_interval(self.feature))
            if cond_prob == 0.0:
                return 0.0
            prob_anc.descend_left(self.feature, self.threshold)
            prob_left = _interval_prob(cdf_dict[self.feature], prob_anc.current_interval(self.feature)) / cond_prob
            result = 0.0
            if prob_left > 1e-12:
                result += prob_left * self.yes.descend(cdf_dict, prob_anc, leaf_contrib_fun)
            prob_anc.revert_left(self.feature)
            if 1.0 - prob_left > 1e-12:
                prob_anc.descend_right(self.feature, self.threshold)
                result += (1.0 - prob_left) * self.no.descend(cdf_dict, prob_anc, leaf_contrib_fun)
                prob_anc.revert_right(self.feature)
            return result
        else:
            return self.missing.descend(cdf_dict, prob_anc, leaf_contrib_fun)

    def eval(self, x):
        if self.feature in x:
            if float(x[self.feature]) < float(self.threshold):
                return self.yes.eval(x)
            else:
                return self.no.eval(x)
        else:
            return self.missing.eval(x)

    def collect_thresholds(self, data_point, perturbed_feature, current_lb, result):
        if self.feature in data_point:
            if self.feature == perturbed_feature:
                self.yes.collect_thresholds(data_point, perturbed_feature, current_lb, result)
                self.no.collect_thresholds(data_point, perturbed_feature, max(current_lb, self.threshold), result)
            else:
                if data_point[self.feature] < self.threshold:
                    self.yes.collect_thresholds(data_point, perturbed_feature, current_lb, result)
                else:
                    self.no.collect_thresholds(data_point, perturbed_feature, current_lb, result)
        else:
            self.missing.collect_thresholds(data_point, perturbed_feature, current_lb, result)


class TreeEnsemble(Model):
    def __init__(self, trees: list, json_file: Path = None):
        self.trees = trees
        self.json_file = json_file
        if self.json_file is not None:
            self.bias = self.get_bias()
            li = self.get_tree_lists()
            self.parse_correct_numbers(li)

    def get_tree_lists(self):
        tree_list = []
        with open(self.json_file) as f:
            data = json.load(f)
            for i in data['learner']['gradient_booster']['model']['trees']:
                tree_list.append(i['split_conditions'])
        return tree_list

    def get_bias(self):
        with open(self.json_file, "r") as fd:
            model = json.load(fd)
        return float(model['learner']['learner_model_param']['base_score'])

    def expected_diff_squared(self, cdf_dict: dict, baseline):
        baseline -= self.bias
        result = baseline ** 2

        def contrib_outer(cdd, prob_anc, val):
            inner_sum = -baseline * 2.0
            for inner_tree in self.trees:
                inner_sum += inner_tree.descend(cdd, prob_anc, lambda _a, _b, v: v)
            return val * inner_sum

        for tree in self.trees:
            result += tree.descend(cdf_dict, _CurrentPath(), contrib_outer)
        return result

    def expected_single_feature(self, data_point, perturbed_feature, cdf, f):
        deltas = [(float('inf'), 0)]
        for tree in self.trees:
            tree_deltas = []
            tree.collect_thresholds(data_point, perturbed_feature, float('-inf'), tree_deltas)
            for i in range(len(tree_deltas) - 1, 0, -1):
                tree_deltas[i] = tree_deltas[i][0], tree_deltas[i][1] - tree_deltas[i - 1][1]
            deltas.extend(tree_deltas)
        deltas.sort(key=itemgetter(0))
        aggr, result, prev = 0.0, 0.0, float('-inf')
        for x, d in deltas:
            result += f(aggr + self.bias) * (cdf(x) - cdf(prev))
            prev = x
            aggr += d
        return result

    # Python3 program to illustrate the
    # Kahan summation algorithm

    # Function to implement the Kahan
    # summation algorithm
    def kahanSum(self, fa):
        sum = float(0)

        # Variable to store the error
        c = float(0)

        # Loop to iterate over the array
        for f in fa:
            y = f - c
            t = sum + y

            # Algebraically, c is always 0
            # when t is replaced by its
            # value from the above expression.
            # But, when there is a loss,
            # the higher-order y is cancelled
            # out by subtracting y from c and
            # all that remains is the
            # lower-order error in c
            c = (t - sum) - y
            sum = t

        return sum

    def eval(self, x):
        return functools.reduce(operator.add, [tree.eval(x) for tree in self.trees], float(0)) + self.bias
    
    def eval_on_multiple_rows(self, df: pd.DataFrame) -> np.array:
        y = []
        for i in range(len(df)):
            x = df.iloc[i, :-1]
            y.append(self.eval(x))
        return np.array(y)

    def parse_correct_numbers(self, splits_list: list):
        for tree, splits in zip(self.trees, splits_list):
            self.traverse_tree_and_correct_numbers(tree, splits)

    def traverse_tree_and_correct_numbers(self, tree: Node, correct_splits: list):
        nodes_to_visit = [tree]
        while len(nodes_to_visit) > 0:
            current_node = nodes_to_visit[0]
            if isinstance(current_node, Split):
                current_node.threshold = float(correct_splits[0])
                if current_node.yes is not None:
                    nodes_to_visit.append(current_node.yes)
                if current_node.no is not None:
                    nodes_to_visit.append(current_node.no)
            elif isinstance(current_node, Leaf):
                current_node.val = float(correct_splits[0])
            correct_splits.pop(0)
            nodes_to_visit.pop(0)


def parse_xgboost_dump(dump_file):
    f = open(dump_file, "r")

    header = re.compile(r"booster\[\d+]:\n")
    node_pattern = re.compile(
        r"\s*(?P<id>\d+):"
        r"\[(?P<feature>\w+)<(?P<threshold>[^\]]+)]\s*"
        r"yes=(?P<yes>\d+),no=(?P<no>\d+),missing=(?P<missing>\d+)"
    )
    leaf_pattern = re.compile(r"\s*(?P<id>\d+):leaf=(?P<value>[^,]+)")

    trees = []

    while True:
        header_line = f.readline()
        if not header_line:
            break
        if header.fullmatch(header_line) is None:
            raise ValueError("broken tree header")
        nodes = dict()

        def parse_subtree():
            line = f.readline()
            match = node_pattern.match(line)
            if match is not None:
                parse_subtree()
                parse_subtree()
                md = match.groupdict()
                if md['id'] in nodes:
                    raise ValueError("node id error")
                nodes[md['id']] = Split(md['feature'], float(md['threshold']), nodes[md['yes']], nodes[md['no']],
                                        nodes[md['missing']])
                return nodes[md['id']]
            else:
                match = leaf_pattern.match(line)
                if match is None:
                    raise ValueError("invalid node format")
                md = match.groupdict()
                nodes[md['id']] = Leaf(float(md['value']))
                return nodes[md['id']]

        trees.append(parse_subtree())

    f.close()
    return TreeEnsemble(trees)

def load_trees(models_path: Path, model_name: str):
    trees_from_dump = parse_xgboost_dump(models_path / f"{model_name}_dumped.txt")
    return TreeEnsemble(trees_from_dump.trees, models_path / f"{model_name}_saved.json")
