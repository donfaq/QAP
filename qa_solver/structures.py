import numpy as np
import itertools
from functools import total_ordering


class QAProblem:
    def __init__(self, n, distances, flows):
        self.n = n
        self.distances = distances
        self.flows = flows


@total_ordering
class QASolution:
    def __init__(self, problem: QAProblem, sol):
        self._problem: QAProblem = problem
        assert len(sol) == self._problem.n, "Incorrect sol"
        self._sol = np.array(sol)

    @property
    def objective_function(self):
        res = 0
        for i, j in itertools.permutations(range(self._problem.n), 2):
            res += (
                self._problem.flows[i][j]
                * self._problem.distances[self._sol[i]][self._sol[j]]
            )
        return res

    def __gt__(self, other):
        return self.objective_function > other.objective_function

    def __eq__(self, other):
        return self.objective_function == other.objective_function

    def __repr__(self):
        return (
            str(self._sol.tolist()).replace("[", "").replace("]", "").replace(",", "")
        )
