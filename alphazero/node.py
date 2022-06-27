from math import sqrt


class Node:
    def __init__(self, prior_prob, c_puct=5, parent=None):
        self.Q = 0
        self.U = 0
        self.N = 0
        self.score = 0
        self.P = prior_prob
        self.c_puct = c_puct
        self.parent = parent
        self.children = {}

    def select(self):
        return max(self.children.items(), key=lambda item: item[1].get_score())

    def expand(self, action_probs):
        for action, prior_prob in action_probs:
            self.children[action] = Node(prior_prob, self.c_puct, self)

    def __update(self, value):
        self.Q = (self.N * self.Q + value) / (self.N + 1)
        self.N += 1

    def backup(self, value):
        if self.parent:
            self.parent.backup(-value)
        self.__update(value)

    def get_score(self):
        self.U = self.c_puct * self.P * sqrt(self.parent.N) / (1 + self.N)
        self.score = self.U + self.Q
        return self.score

    def is_leaf_node(self):
        return len(self.children) == 0
