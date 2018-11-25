import numpy as np

from qa_solver.structures import QAProblem, QASolution


class QASolver:
    def __init__(
        self, problem: QAProblem, population_size=5, n_children=5, mutation_p=0.2
    ):
        self._problem = problem
        self._pop_size = population_size
        self._n_children = n_children
        self._p_m = mutation_p

    def _gen_random_solution(self):
        sol = np.arange(self._problem.n)
        np.random.shuffle(sol)
        return QASolution(self._problem, sol)

    def _gen_random_pop(self):
        return np.array([self._gen_random_solution() for _ in range(self._pop_size)])

    def _ox(self, par1: QASolution, par2: QASolution):
        a, b = par1._sol.copy(), par2._sol.copy()
        size = np.random.randint(2, a.shape[0] // 2)
        start_idx = np.random.randint(a.shape[0] - size)
        a_slice = np.arange(start_idx, start_idx + size)
        # a_slice = np.random.randint(20, size=size)
        child = np.zeros(a.shape[0], dtype=np.int)
        child[a_slice] = list(reversed(a[a_slice]))
        b = b[np.isin(b, a[a_slice], invert=True)]
        fill_idx = np.arange(child.shape[0])
        fill_idx = fill_idx[np.isin(fill_idx, a_slice, invert=True)]
        child[fill_idx] = b
        return QASolution(par1._problem, child)

    def _ulx(self, first: QASolution, second: QASolution):
        """Crossover of two parents producing one child
        Equal positions are kept, while the rest is randomly chosen.
        """
        first, second = first._sol.copy(), second._sol.copy()
        child = np.zeros(self._problem.n, dtype=np.int)
        equal_positions = np.where(first == second)[0]
        placed = []
        if equal_positions.shape[0] > 0:
            child[equal_positions] = first[equal_positions]
            placed += first[equal_positions].tolist()

        for i in range(self._problem.n):
            if i not in equal_positions:
                if first[i] not in placed and second[i] not in placed:
                    if np.random.sample() < 0.5:
                        child[i] = first[i]
                    else:
                        child[i] = second[i]
                elif first[i] not in placed:
                    child[i] = first[i]
                elif second[i] not in placed:
                    child[i] = second[i]
                else:
                    # If both values already placed choose random
                    child[i] = np.random.choice(
                        [x for x in range(self._problem.n) if x not in placed]
                    )
            placed.append(child[i])
        return QASolution(self._problem, child)

    def _crossover(self, population: np.ndarray):
        children = []
        probs = self._fitness(population)
        while len(children) < self._n_children:
            par_a, par_b = np.random.choice(population, size=2, replace=False, p=probs)
            cross_func = np.random.choice([self._ulx, self._ox])
            child = cross_func(par_a, par_b)
            assert len(set(child._sol)) == child._sol.shape[0], "child not feasible"
            if child not in population and child not in children:
                children.append(child)
        return np.array(children)

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
        ranks = {str(el._sol): rank for rank, el in enumerate(sorted(population))}
        ranks = np.array(list(map(lambda x: ranks[str(x._sol)], population)))
        return ranks / ranks.sum()
        # obj = np.array(list(map(lambda x: x.objective_function, population)))
        # obj = 1 / np.array(list(map(lambda x: x.objective_function, population)))
        # return obj / obj.sum()

    def solve(self):
        population = self._gen_random_pop()
        best = min(population)
        n_converge = 0
        not_cool_count = 0
        while n_converge < 100:
            not_cool = True
            offsprings = list(map(self._mutation, self._crossover(population)))
            population = np.concatenate([population, offsprings])
            # select n best from population
            # population = np.random.choice(population, size=self._pop_size,
            #     replace=False, p=self._fitness(population))
            population = sorted(population, reverse=True)[self._pop_size :]
            cur_opt = min(population)
            if cur_opt < best:
                not_cool = False
                best = cur_opt
                print(best, best.objective_function)
            else:
                not_cool_count += 1
            if not_cool_count > 500:
                # perturbate this garbage
                population = np.concatenate([population, self._gen_random_pop()])
                population = np.random.choice(
                    population, 
                    size=self._pop_size, 
                    replace=False
                )
                n_converge += 1

        return best
