"""
main.py
-------
End-to-end demo: given a candidate's skills, find the best-matching jobs,
compute the skill gap for the top match, and recommend learning resources
to close that gap.

Run: python3 main.py
"""

import sys
sys.path.append("src")

from matcher import JobMatcher
from learning_recommender import recommend_resources


def demo(candidate_skills):
    matcher = JobMatcher("data/jobs.csv")

    print(f"Candidate skills: {candidate_skills}\n")

    matches = matcher.top_matches(candidate_skills, k=3)
    print("Top 3 job matches:")
    for i, m in enumerate(matches, start=1):
        print(f"  {i}. {m['job_title']}  (similarity: {m['similarity']})")

    best_job = matches[0]["job_title"]
    gap = matcher.skill_gap(candidate_skills, best_job)

    print(f"\nSkill gap for top match ({best_job}):")
    if gap:
        for skill in gap:
            print(f"  - {skill}")
        print("\nRecommended learning resources:")
        for skill, resource in recommend_resources(gap).items():
            print(f"  - {skill}: {resource}")
    else:
        print("  None — candidate already meets all required skills!")


if __name__ == "__main__":
    # Example candidate: knows some data skills but is missing a few for Data Scientist
    example_skills = "python; pandas; sql; statistics"
    demo(example_skills)
