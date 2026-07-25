"""
Skill vocabulary for the resume matcher.

Kept in its own file so you can grow it without touching app logic.

`SKILLS` maps a canonical skill name -> list of surface forms to search for.
Matching is case-insensitive and whole-word (so "R" doesn't match "React",
and "go" doesn't match "google").

This is deliberately a hand-curated list, not a model. For a project this size
that is the right call: it's explainable, debuggable, and you can defend every
entry. Swapping it for a learned skill extractor is a good v2.
"""

SKILLS = {
    # --- languages ---
    "Python":            ["python"],
    "Java":              ["java"],
    "C":                 ["c"],
    "C++":               ["c++", "cpp"],
    "C#":                ["c#", "csharp"],
    "JavaScript":        ["javascript", "js"],
    "TypeScript":        ["typescript"],
    "R":                 ["r"],
    "SQL":               ["sql"],
    "Scala":             ["scala"],
    "Go":                ["golang"],
    "Rust":              ["rust"],
    "MATLAB":            ["matlab"],
    "Bash":              ["bash", "shell scripting"],

    # --- ml / ds core ---
    "Machine Learning":      ["machine learning", "ml"],
    "Deep Learning":         ["deep learning"],
    "NLP":                   ["nlp", "natural language processing"],
    "Computer Vision":       ["computer vision", "cv"],
    "Reinforcement Learning":["reinforcement learning"],
    "Time Series":           ["time series", "forecasting"],
    "Statistics":            ["statistics", "statistical analysis"],
    "A/B Testing":           ["a/b testing", "ab testing", "hypothesis testing"],
    "Feature Engineering":   ["feature engineering"],
    "Model Deployment":      ["model deployment", "mlops"],
    "Recommender Systems":   ["recommender", "recommendation system", "collaborative filtering"],

    # --- frameworks ---
    "PyTorch":           ["pytorch", "torch"],
    "TensorFlow":        ["tensorflow"],
    "Keras":             ["keras"],
    "Scikit-learn":      ["scikit-learn", "sklearn", "scikit learn"],
    "HuggingFace":       ["huggingface", "hugging face", "transformers"],
    "XGBoost":           ["xgboost"],
    "LightGBM":          ["lightgbm"],
    "spaCy":             ["spacy"],
    "NLTK":              ["nltk"],
    "OpenCV":            ["opencv"],
    "LangChain":         ["langchain"],

    # --- data libs ---
    "Pandas":            ["pandas"],
    "NumPy":             ["numpy"],
    "Matplotlib":        ["matplotlib"],
    "Seaborn":           ["seaborn"],
    "Plotly":            ["plotly"],
    "SciPy":             ["scipy"],
    "Spark":             ["spark", "pyspark", "apache spark"],
    "Hadoop":            ["hadoop"],
    "Airflow":           ["airflow"],
    "dbt":               ["dbt"],

    # --- databases ---
    "PostgreSQL":        ["postgresql", "postgres"],
    "MySQL":             ["mysql"],
    "MongoDB":           ["mongodb"],
    "Redis":             ["redis"],
    "Snowflake":         ["snowflake"],
    "BigQuery":          ["bigquery"],
    "Redshift":          ["redshift"],
    "Elasticsearch":     ["elasticsearch"],

    # --- cloud / infra ---
    "AWS":               ["aws", "amazon web services"],
    "Azure":             ["azure"],
    "GCP":               ["gcp", "google cloud"],
    "Docker":            ["docker"],
    "Kubernetes":        ["kubernetes", "k8s"],
    "CI/CD":             ["ci/cd", "cicd", "continuous integration"],
    "Linux":             ["linux", "unix"],
    "Git":               ["git", "github", "version control"],

    # --- web / app ---
    "React":             ["react", "reactjs"],
    "Node.js":           ["node.js", "nodejs"],
    "Django":            ["django"],
    "Flask":             ["flask"],
    "FastAPI":           ["fastapi"],
    "Streamlit":         ["streamlit"],
    "REST API":          ["rest api", "restful", "api development"],
    "HTML/CSS":          ["html", "css"],

    # --- bi / analytics ---
    "Power BI":          ["power bi", "powerbi"],
    "Tableau":           ["tableau"],
    "Excel":             ["excel", "advanced excel"],
    "Looker":            ["looker"],
    "Google Analytics":  ["google analytics"],
    "Data Visualization":["data visualization", "data visualisation"],
    "ETL":               ["etl", "elt", "data pipeline"],
    "Data Modeling":     ["data modeling", "data modelling", "dimensional modeling"],
    "Data Warehousing":  ["data warehouse", "data warehousing"],

    # --- ways of working ---
    "Agile":             ["agile", "scrum"],
    "Jira":              ["jira"],
    "Stakeholder Management": ["stakeholder"],
    "Technical Writing": ["technical writing", "documentation"],
    "Communication":     ["communication skills", "written communication"],
    "Leadership":        ["leadership", "team lead", "mentoring"],
    "Problem Solving":   ["problem solving", "problem-solving"],
}


def skill_count():
    return len(SKILLS), sum(len(v) for v in SKILLS.values())


if __name__ == "__main__":
    n_skills, n_forms = skill_count()
    print(f"{n_skills} canonical skills, {n_forms} surface forms")
