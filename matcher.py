"""
matcher.py
----------
Core matching engine: represents each job's required skills and each
candidate's skills as text, vectorizes with TF-IDF, and ranks jobs for a
candidate by cosine similarity.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobMatcher:
    def __init__(self, jobs_csv):
        self.jobs = pd.read_csv(jobs_csv)
        # TF-IDF over skill lists — treats each skill as a "word" (semicolon-split, not comma)
        self.jobs["skills_text"] = self.jobs["required_skills"].apply(
            lambda s: " ".join(skill.strip().replace(" ", "_") for skill in s.split(";"))
        )
        self.vectorizer = TfidfVectorizer()
        self.job_vectors = self.vectorizer.fit_transform(self.jobs["skills_text"])

    def _vectorize_candidate(self, skills_str):
        text = " ".join(skill.strip().replace(" ", "_") for skill in skills_str.split(";"))
        return self.vectorizer.transform([text])

    def top_matches(self, candidate_skills, k=3):
        """Return the top-k job matches for a candidate's skill string, ranked by similarity."""
        vec = self._vectorize_candidate(candidate_skills)
        sims = cosine_similarity(vec, self.job_vectors).flatten()
        ranked = sims.argsort()[::-1][:k]
        results = []
        for idx in ranked:
            results.append({
                "job_title": self.jobs.iloc[idx]["job_title"],
                "similarity": round(float(sims[idx]), 3),
                "required_skills": self.jobs.iloc[idx]["required_skills"],
            })
        return results

    def skill_gap(self, candidate_skills, job_title):
        """Return the skills required by job_title that the candidate does not have."""
        candidate_set = {s.strip().lower() for s in candidate_skills.split(";")}
        job_row = self.jobs[self.jobs["job_title"] == job_title].iloc[0]
        required_set = {s.strip().lower() for s in job_row["required_skills"].split(";")}
        return sorted(required_set - candidate_set)
