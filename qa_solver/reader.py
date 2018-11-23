from qa_solver.structures import QAProblem


class QAReader:
    def __init__(self):
        pass

    def __call__(self, problem_filepath, *args, **kwargs):
        print("Reading problem from {}".format(problem_filepath))
        with open(problem_filepath, "r") as f:
            n = int(f.readline().strip())
            distances, flows = [], []
            for _ in range(n):
                flows.append(list(map(int, f.readline().split())))
            _ = f.readline()
            for _ in range(n):
                distances.append(list(map(int, f.readline().split())))
        return QAProblem(n, distances, flows)
