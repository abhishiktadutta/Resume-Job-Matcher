"""
Core matching logic — kept separate from the Streamlit UI.

Why separate? Because it means you can import these functions into a notebook
and evaluate them, which you cannot do if the logic lives inside button
callbacks. Being able to test your logic without launching the app is the
difference between a project you can measure and a project you can only demo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from skills_vocab import SKILLS


# ----------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, collapse whitespace. Deliberately gentle — we keep +, #, /
    because they carry meaning in skill names (c++, c#, ci/cd)."""
    if not text:
        return ""
    text = text.replace(" ", " ")
    text = re.sub(r"[^\w\s+#/.\-]", " ", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------------------------------------------------------
# Skill extraction
# ----------------------------------------------------------------------------

def _form_pattern(form: str) -> re.Pattern:
    """Whole-'word' match that tolerates the punctuation inside skill names.

    Plain \\b fails on things like 'c++' because '+' is already a non-word
    character, so we use explicit lookarounds instead.

    The boundary class includes '+' and '#' as well as alphanumerics. Without
    them, the skill "C" matches the "c" inside "c++" and every C++ developer
    gets credited with C. It does NOT include '/' — otherwise "R/Python" would
    fail to match R, which is a common way people write it.

    This is the kind of bug that silently inflates your results, so there are
    tests for it in the notebook. Run them if you edit this.
    """
    escaped = re.escape(form)
    return re.compile(rf"(?<![a-z0-9+#]){escaped}(?![a-z0-9+#])")


_COMPILED = {
    canonical: [_form_pattern(f) for f in forms]
    for canonical, forms in SKILLS.items()
}


def extract_skills(text: str) -> set[str]:
    """Return the set of canonical skill names found in `text`."""
    t = clean_text(text)
    found = set()
    for canonical, patterns in _COMPILED.items():
        if any(p.search(t) for p in patterns):
            found.add(canonical)
    return found


# ----------------------------------------------------------------------------
# Similarity
# ----------------------------------------------------------------------------

def tfidf_similarity(resume: str, jds: list[str]) -> np.ndarray:
    """TF-IDF cosine similarity between one resume and N job descriptions.

    Note the vectorizer is fitted on resume + all JDs together. That is correct
    here — we need one shared vocabulary — but be aware it means scores shift
    slightly when you add or remove a JD. Mention this if anyone asks.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [clean_text(resume)] + [clean_text(j) for j in jds]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(docs)
    return cosine_similarity(X[0:1], X[1:]).ravel()


_MODEL_CACHE: dict[str, object] = {}


def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load and cache the embedding model (first call downloads it)."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embedding_similarity(resume: str, jds: list[str],
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Cosine similarity in embedding space. Catches paraphrase that TF-IDF misses
    ('built ML pipelines' vs 'experience with machine learning workflows')."""
    model = get_model(model_name)
    emb = model.encode([resume] + list(jds), normalize_embeddings=True,
                       convert_to_numpy=True)
    return (emb[0:1] @ emb[1:].T).ravel()


# ----------------------------------------------------------------------------
# Putting it together
# ----------------------------------------------------------------------------

@dataclass
class MatchResult:
    jd_index: int
    jd_label: str
    tfidf_score: float
    embedding_score: float
    skill_coverage: float          # fraction of JD skills present in resume
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)


def match(resume: str, jds: list[str], jd_labels: list[str] | None = None,
          model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
          use_embeddings: bool = True) -> list[MatchResult]:
    """Score one resume against N job descriptions and return results sorted
    best-first by embedding score (falling back to TF-IDF if embeddings are off)."""
    if not jds:
        return []
    labels = jd_labels or [f"Job {i+1}" for i in range(len(jds))]

    tfidf = tfidf_similarity(resume, jds)
    emb = embedding_similarity(resume, jds, model_name) if use_embeddings else np.zeros(len(jds))

    resume_skills = extract_skills(resume)
    results = []
    for i, jd in enumerate(jds):
        jd_skills = extract_skills(jd)
        matched = sorted(jd_skills & resume_skills)
        missing = sorted(jd_skills - resume_skills)
        extra   = sorted(resume_skills - jd_skills)
        coverage = len(matched) / len(jd_skills) if jd_skills else 0.0
        results.append(MatchResult(
            jd_index=i, jd_label=labels[i],
            tfidf_score=float(tfidf[i]),
            embedding_score=float(emb[i]),
            skill_coverage=coverage,
            matched_skills=matched, missing_skills=missing, extra_skills=extra,
        ))

    key = (lambda r: r.embedding_score) if use_embeddings else (lambda r: r.tfidf_score)
    return sorted(results, key=key, reverse=True)


# ----------------------------------------------------------------------------
# PDF text extraction
# ----------------------------------------------------------------------------

def extract_pdf_text(file_obj) -> str:
    """Pull text out of a PDF file object.

    Heads up: this works well on text-based PDFs (anything exported from Word,
    LaTeX, Google Docs) and produces garbage on scanned images, because there is
    no text layer to extract. If a user uploads a scan you get an empty string,
    which the app surfaces as a warning rather than silently scoring zero.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # older name, same API for our purposes

    reader = PdfReader(file_obj)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()


def split_jds(blob: str, delimiter: str = "---") -> list[str]:
    """Split a pasted blob into separate job descriptions on a delimiter line."""
    chunks = [c.strip() for c in blob.split(delimiter)]
    return [c for c in chunks if c]
