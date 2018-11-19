import os
import argparse


from qa_solver import QAReader, QASolver


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solving QA with Genetic algorithm")
    parser.add_argument("problem_file", type=str, help="Problem file")
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    PROBLEM_FILE = os.path.abspath(args.problem_file)
    reader = QAReader()
    problem = reader(PROBLEM_FILE)
    solver = QASolver(problem)
    solver.solve()

