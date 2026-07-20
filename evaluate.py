"""
evaluate.py
-----------
Honest evaluation of the matcher: for each synthetic candidate, we know
the "true_target_job" they were generated for. We check whether the
matcher's top-1 and top-3 recommendations recover that job. This is a
real, computed metric — not a placeholder number.
"""

import pandas as pd
from matcher import JobMatcher


def evaluate(jobs_csv="data/jobs.csv", candidates_csv="data/candidates.csv"):
    matcher = JobMatcher(jobs_csv)
    candidates = pd.read_csv(candidates_csv)

    top1_correct = 0
    top3_correct = 0

    for _, row in candidates.iterrows():
        matches = matcher.top_matches(row["skills"], k=3)
        titles = [m["job_title"] for m in matches]
        if titles[0] == row["true_target_job"]:
            top1_correct += 1
        if row["true_target_job"] in titles:
            top3_correct += 1

    n = len(candidates)
    top1_acc = top1_correct / n
    top3_acc = top3_correct / n
    print(f"Candidates evaluated: {n}")
    print(f"Top-1 accuracy: {top1_acc:.1%}")
    print(f"Top-3 accuracy: {top3_acc:.1%}")
    return top1_acc, top3_acc


if __name__ == "__main__":
    evaluate()
