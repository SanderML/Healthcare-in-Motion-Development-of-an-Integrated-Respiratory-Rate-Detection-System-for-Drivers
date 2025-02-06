import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_jobs(jobs_file: str, train_one_subject_only, train_subset):
    with open(jobs_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row

        for i, row in enumerate(reader, 1):
            lr, wd = row
            cmd = [
                sys.executable,
                "main.py",
                "--lr",
                lr,
                "--wd",
                wd,
            ]
            if args.train_one_subject_only:
                cmd.append("--train_one_subject_only")
            if args.train_subset:
                cmd.append("--train_subset")

            try:
                subprocess.run(
                    cmd,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error running job {i}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run jobs from a csv file: first column is learning rate, second column is weight decay, first row is header"
    )
    parser.add_argument("jobs_file", type=str, help="Path to the jobs file", default="jobs.csv")
    parser.add_argument("--train_one_subject_only", action="store_true", help="Train one subject only")
    parser.add_argument("--train_subset", action="store_true", help="Use only 3 batches per subject")
    args = parser.parse_args()

    jobs_file = Path(args.jobs_file)
    if not jobs_file.exists():
        print(args.jobs_file, "not found!")
        sys.exit(1)

    run_jobs(jobs_file, args.train_one_subject_only, args.train_subset)
