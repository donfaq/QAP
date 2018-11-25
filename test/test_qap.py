import os

import pytest

from qa_solver import *


@pytest.fixture
def problem():
    path = os.path.abspath("instances/tai20a")
    return QAReader()(path)


def test_objective_function_exact(problem):
    sol = [9, 8, 11, 19, 18, 2, 13, 5, 16, 10, 4, 6, 14, 15, 17, 1, 3, 7, 12, 0]
    exact = QASolution(problem, sol)
    assert exact.objective_function == 703482
