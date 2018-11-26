import argparse
import os
from glob import glob

from qa_solver import QAReader, QASolver


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solving QA with Genetic algorithm")
    parser.add_argument("-p", type=str, required=False, help="Problem file")
    parser.add_argument("-o", type=str, required=False, help="Output file")
    parser.add_argument("--test", action="store_true", help="Test on instances folder")
    return parser.parse_args()


def solve_and_save(prob_file, out_file):
    solution = QASolver(QAReader()(prob_file)).solve()
    with open(OUTPUT_FILE, "w") as f:
        f.write(str(solution))


if __name__ == "__main__":
    args = arguments()
    IS_TEST = args.test
    if not IS_TEST:
        assert args.p, "Please specify problem file path"
        assert args.o, "Please specify output file path"
        PROBLEM_FILE = os.path.abspath(args.p)
        OUTPUT_FILE = os.path.abspath(args.o)
        solve_and_save(PROBLEM_FILE, OUTPUT_FILE)
    else:
        FILES = glob("instances/*")
        if not os.path.exists("results"):
            os.mkdir("results")
        for PROBLEM_FILE in FILES:
            OUTPUT_FILE = os.path.join(
                "results", "{}.sol".format(os.path.basename(PROBLEM_FILE))
            )
            solve_and_save(PROBLEM_FILE, OUTPUT_FILE)
