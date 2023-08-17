import re
from abc import ABC
from collections import defaultdict
from operator import itemgetter


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
        print(feature)
        print(self.lb)
        return self.lb[feature][-1] if feature in self.lb else float('-inf')

    def last_ub(self, feature):
        print(feature)
        print(self.ub)
        return self.ub[feature][-1] if feature in self.ub else float('inf')

    def descend_left(self, feature, t):
        self.ub[feature].append(min(self.last_ub(feature), t))

    def descend_right(self, feature, t):
        self.lb[feature].append(max(self.last_lb(feature), t))

    def revert_left(self, feature):
        self.ub[feature].pop()
        if len(self.ub[feature]) == 0:
            self.ub.pop(feature)
            assert(feature not in self.ub)

    def revert_right(self, feature):
        self.lb[feature].pop()
        if len(self.lb[feature]) == 0:
            self.lb.pop(feature)
            assert (feature not in self.lb)

    def current_interval(self, feature):
        return self.last_lb(feature), self.last_ub(feature)


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

    def collect_thresholds(self, data_point, perturbed_feature, current_ub, result):
        prev = result[-1][1] if len(result) > 0 else 0
        result.append((current_ub, self.val - prev))


def _interval_prob(cdf, interval):
    # we mean a closed-open [a,b) interval
    a, b = interval
    return max(cdf(b - 1.0e-9) - cdf(a - 1.0e-9), 0.0)


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
            result = prob_left * self.yes.descend(cdf_dict, prob_anc, leaf_contrib_fun)
            prob_anc.revert_left(self.feature)
            prob_anc.descend_right(self.feature, self.threshold)
            result += (1.0 - prob_left) * self.no.descend(cdf_dict, prob_anc, leaf_contrib_fun)
            prob_anc.revert_right(self.feature)
            return result
        else:
            return self.missing.descend(cdf_dict, prob_anc, leaf_contrib_fun)

    def eval(self, x):
        if self.feature in x:
            if x[self.feature] < self.threshold:
                return self.yes.eval(x)
            else:
                return self.no.eval(x)
        else:
            return self.missing.eval(x)

    def collect_thresholds(self, data_point, perturbed_feature, current_ub, result):
        if self.feature == perturbed_feature:
            self.no.collect_thresholds(data_point, perturbed_feature, min(current_ub, self.threshold), result)
            self.yes.collect_thresholds(data_point, perturbed_feature, current_ub, result)
        elif self.feature in data_point:
            if data_point[self.feature] < self.threshold:
                self.yes.collect_thresholds(data_point, perturbed_feature, current_ub, result)
            else:
                self.no.collect_thresholds(data_point, perturbed_feature, current_ub, result)
        else:
            self.missing.collect_thresholds(data_point, perturbed_feature, current_ub, result)


class TreeEnsemble(Model):
    def __init__(self, trees: list):
        self.trees = trees

    def expected_diff_squared(self, cdf_dict: dict, baseline):
        result = baseline ** 2

        def contrib_outer(cdd, prob_anc, val):
            inner_sum = -baseline
            for inner_tree in self.trees:
                inner_sum += inner_tree.descend(cdd, prob_anc, lambda _a, _b, v: v)
            return val * inner_sum / len(self.trees)

        for tree in self.trees:
            result += tree.descend(cdf_dict, _CurrentPath(), contrib_outer)
        return result

    def expected_single_feature(self, data_point, perturbed_feature, cdf, f):
        deltas = [(float('inf'), 0)]
        for tree in self.trees:
            tree_deltas = []
            tree.collect_thresholds(data_point, perturbed_feature, float('inf'), tree_deltas)
            deltas.extend(tree_deltas)
        deltas.sort(key=itemgetter(0))
        aggr, result, prev = 0.0, 0.0, float('-inf')
        for x, d in deltas:
            aggr += d / len(self.trees)
            result += f(aggr) * (cdf(x) - cdf(prev))
            prev = x
        return result

    def eval(self, x):
        result = 0.0
        for tree in self.trees:
            result += tree.eval(x)
        result /= len(self.trees)
        return result


def parse_xgboost_dump(dump_file):
    f = open(dump_file, "r")

    header = re.compile(r"booster\[\d+]:\n")
    node_pattern = re.compile(
        r"\s*(?P<id>\d+):"
        r"\[(?P<feature>\w+)<(?P<threshold>-?\d+(?:\.\d+)?)]\s*"
        r"yes=(?P<yes>\d+),no=(?P<no>\d+),missing=(?P<missing>\d+)"
    )
    leaf_pattern = re.compile(r"\s*(?P<id>\d+):leaf=(?P<value>-?\d+(?:\.\d+)?)")

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
