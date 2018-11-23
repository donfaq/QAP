import numpy as np
from qa_solver.structures import QAProblem, QASolution


class QASolver:
    def __init__(
        self, problem: QAProblem, population_size=10, n_children=100, mutation_p=0.2
    ):
        self._problem = problem
        self._pop_size = population_size
        self._n_children = n_children
        self._p_m = mutation_p

    def _generate_init_pop(self):
        pop = [np.arange(self._problem.n) for _ in range(self._pop_size)]
        list(map(np.random.shuffle, pop))
        pop = list(map(lambda x: QASolution(self._problem, x), pop))
        return np.array(pop)

    def _crossover(self, population: np.ndarray):
        def ox(par1: QASolution, par2: QASolution):
            a, b = par1._sol.copy(), par2._sol.copy()
            size = np.random.randint(2, a.shape[0] // 2)
            start_idx = np.random.randint(a.shape[0] - size)
            a_slice = np.arange(start_idx, start_idx + size)
            child = np.zeros(a.shape[0], dtype=np.int)
            child[a_slice] = a[a_slice]
            b = b[np.isin(b, a[a_slice], invert=True)]
            fill_idx = np.arange(child.shape[0])
            fill_idx = fill_idx[np.isin(fill_idx, a_slice, invert=True)]
            child[fill_idx] = b
            assert len(set(child)) == child.shape[0], "child not feasible"
            return QASolution(par1._problem, child)

        res = []
        probs = self._fitness(population)
        for _ in range(self._n_children):
            par_a, par_b = np.random.choice(population, size=2, replace=False, p=probs)
            res.append(ox(par_a, par_b))
        return np.array(res)

    def _mutation(self, solution: QASolution):
        res = solution._sol.copy()
        if np.random.sample() <= self._p_m:
            size = np.random.randint(2, res.shape[0])
            start_idx = np.random.randint(res.shape[0] - size + 1)
            sl = np.arange(start_idx, start_idx + size)
            a = res[sl]
            np.random.shuffle(a)
            res[sl] = a
        return QASolution(solution._problem, res)

    def _fitness(self, population):
        obj = np.array(list(map(lambda x: x.objective_function, population)))
        return obj / np.sum(obj)

    def solve(self):
        population = self._generate_init_pop()
        best = min(population)
        while True:
            offsprings = list(map(self._mutation, self._crossover(population)))
            population = np.concatenate([population, offsprings])
            population = np.random.choice(
                population,
                size=self._pop_size,
                replace=False,
                p=self._fitness(population),
            )
            cur_opt = min(population)
            if cur_opt < best:
                best = cur_opt
                print(best, best.objective_function)
        return best
