import re
from collections import defaultdict


class Model:
    def expected_value(self, cdf_dict: dict, baseline):
        raise NotImplementedError('')

    def eval(self, x):
        raise NotImplementedError('')


class _CurrentPath:
    def __init__(self):
        self.lb = defaultdict(list)
        self.ub = defaultdict(list)

    def descend_left(self, feature, t):
        self.ub[feature].append(t)

    def descend_right(self, feature, t):
        self.lb[feature].append(t)

    def revert_left(self, feature):
        self.ub[feature].pop()

    def revert_right(self, feature):
        self.lb[feature].pop()

    def current_interval(self, feature):
        return (self.lb[feature][-1] if len(self.lb[feature]) > 0 else float('-inf'),
                self.ub[feature][-1] if len(self.ub[feature]) > 0 else float('inf'))


class Node(Model):
    def is_leaf(self):
        raise NotImplementedError('')

    def _expected_value(self, cdf_dict: dict, prob_anc: _CurrentPath, baseline):
        raise NotImplementedError('')

    # cdf_dict is of type feature -> cumulative distribution function
    def expected_value(self, cdf_dict: dict, baseline):
        return self._expected_value(cdf_dict, _CurrentPath(), baseline)


class Leaf(Node):
    def __init__(self, val):
        self.val = val

    def is_leaf(self):
        return True

    def _expected_value(self, cdf_dict: dict, prob_anc: dict, baseline):
        return abs(baseline - self.val)

    def eval(self, x):
        return self.val


class Split(Node):
    def __init__(self, feature, threshold: float, yes: Node, no: Node, missing: Node):
        self.feature = feature
        self.threshold = threshold
        self.yes = yes
        self.no = no
        self.missing = missing

    def is_leaf(self):
        return False

    @staticmethod
    def _interval_prob(cdf, interval):
        # we mean a closed-open [a,b) interval
        a, b = interval
        return cdf(b - 1.0e-9) - cdf(a - 1.0e-9)

    def _expected_value(self, cdf_dict: dict, prob_anc: _CurrentPath, baseline):
        if self.feature in cdf_dict:
            cond_prob = self._interval_prob(cdf_dict[self.feature], prob_anc.current_interval(self.feature))
            if cond_prob == 0.0:
                return 0.0
            prob_anc.descend_left(self.feature, self.threshold)
            prob_left = self._interval_prob(cdf_dict[self.feature], prob_anc.current_interval(self.feature)) / cond_prob
            result = prob_left * self.yes._expected_value(cdf_dict, prob_anc, baseline)
            prob_anc.revert_left(self.feature)
            prob_anc.descend_right(self.feature, self.threshold)
            result += (1.0 - prob_left) * self.no._expected_value(cdf_dict, prob_anc, baseline)
            prob_anc.revert_right(self.feature)
            return result
        else:
            return self.missing._expected_value(cdf_dict, prob_anc, baseline)

    def eval(self, x):
        if self.feature in x:
            if x[self.feature] < self.threshold:
                return self.yes.eval(x)
            else:
                return self.no.eval(x)
        else:
            return self.missing.eval(x)


class TreeEnsemble(Model):
    def __init__(self, trees: list):
        self.trees = trees

    def expected_value(self, cdf_dict: dict, baseline):
        raise NotImplementedError()
        result = 0
        # to jest zle... :(
        for tree in self.trees:
            result += tree.expected_value(cdf_dict, baseline)
        return result / len(self.trees)


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
