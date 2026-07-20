"""
learning_recommender.py
------------------------
Maps a missing skill to a suggested learning resource. In a production
system this would query a real course catalog / API; here it's a curated
static lookup so the pipeline is runnable end-to-end offline.
"""

RESOURCE_MAP = {
    "python": "Python for Everybody (Coursera)",
    "sql": "SQL for Data Science (Coursera)",
    "machine learning": "Supervised Machine Learning (Stanford / DeepLearning.AI)",
    "tensorflow": "TensorFlow Developer Certificate materials",
    "pytorch": "PyTorch official tutorials",
    "docker": "Docker for Beginners (Docker docs)",
    "kubernetes": "Kubernetes Basics (kubernetes.io)",
    "statistics": "Statistics with Python (Coursera)",
    "data visualization": "Data Visualization with Python (freeCodeCamp)",
    "aws": "AWS Cloud Practitioner Essentials",
    "azure": "Microsoft Azure Fundamentals (AZ-900)",
    "nlp": "NLP Specialization (DeepLearning.AI)",
    "cnn": "CNNs in TensorFlow (DeepLearning.AI)",
    "react": "React official documentation + tutorial",
    "javascript": "The Modern JavaScript Tutorial",
    "git": "Git & GitHub for Beginners",
    "linux": "Linux Command Line Basics",
    "mlops": "MLOps Specialization (DeepLearning.AI)",
}

DEFAULT_RESOURCE = "Search for a beginner-level course on this specific skill"


def recommend_resources(missing_skills):
    return {skill: RESOURCE_MAP.get(skill, DEFAULT_RESOURCE) for skill in missing_skills}
