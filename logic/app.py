"""
Resume ↔ Job Description Matcher
================================

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

A design note you should be able to defend:

This app ranks ONE resume against SEVERAL job descriptions, rather than showing
a single "you are a 78% match" number. That is deliberate.

Cosine similarity between sentence embeddings is not a calibrated probability.
For most embedding models, unrelated text still scores around 0.5-0.7, so
"78%" would be meaningless on its own and actively misleading to a job seeker.
The similarity IS meaningful comparatively — it can tell you that JD A suits
this resume better than JD B. So the app reports a ranking, plus a skill
coverage number that genuinely is interpretable as a fraction.

Most resume-matcher projects show the fake percentage. Not doing so is a small
thing that says you understood what your metric actually measures.
"""

import streamlit as st
import pandas as pd

from matcher import match, extract_pdf_text, split_jds, extract_skills
from skills_vocab import SKILLS

st.set_page_config(page_title="Resume ↔ JD Matcher", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Settings")
    use_embeddings = st.checkbox(
        "Use embedding similarity", value=True,
        help="Off = TF-IDF only (instant). On = also loads a small transformer; "
             "first run downloads ~90MB.")
    model_name = st.selectbox(
        "Embedding model",
        ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
        help="MiniLM is the fast default. BGE usually scores better on retrieval "
             "but note it expects an instruction prefix on queries — see the "
             "model card if you want to do this properly.")
    st.divider()
    st.caption(f"Skill vocabulary: **{len(SKILLS)}** skills. "
               "Edit `skills_vocab.py` to add your own.")

st.title("Resume ↔ Job Description Matcher")
st.caption("Rank job descriptions against your resume, and see which skills you're missing.")

# ------------------------------------------------------------------ inputs ---
left, right = st.columns(2)

with left:
    st.subheader("1. Your resume")
    mode = st.radio("Input method", ["Paste text", "Upload PDF"],
                    horizontal=True, label_visibility="collapsed")

    resume_text = ""
    if mode == "Paste text":
        resume_text = st.text_area("Resume text", height=320,
                                   placeholder="Paste your resume here...",
                                   label_visibility="collapsed")
    else:
        up = st.file_uploader("Upload resume PDF", type=["pdf"],
                              label_visibility="collapsed")
        if up is not None:
            with st.spinner("Extracting text..."):
                resume_text = extract_pdf_text(up)
            if not resume_text:
                st.error(
                    "No text found in that PDF. It's most likely a scan — an image "
                    "of a document rather than a document. You'd need OCR to read it. "
                    "Try pasting the text instead.")
            else:
                st.success(f"Extracted {len(resume_text.split()):,} words")
                with st.expander("Check the extracted text"):
                    st.text(resume_text[:3000])

with right:
    st.subheader("2. Job descriptions")
    st.caption("Separate multiple jobs with a line containing `---`")
    jd_blob = st.text_area("Job descriptions", height=320,
                           placeholder="Paste one or more job descriptions...\n\n---\n\nSecond job here...",
                           label_visibility="collapsed")
    jd_pdfs = st.file_uploader("...or upload JD PDFs", type=["pdf"],
                               accept_multiple_files=True)

# Assemble JD list from both sources
jds, jd_labels = [], []
for i, chunk in enumerate(split_jds(jd_blob)):
    jds.append(chunk)
    first_line = chunk.strip().splitlines()[0][:60] if chunk.strip() else f"Job {i+1}"
    jd_labels.append(first_line)
for f in (jd_pdfs or []):
    text = extract_pdf_text(f)
    if text:
        jds.append(text)
        jd_labels.append(f.name)

# ------------------------------------------------------------------ action ---
st.divider()
go = st.button("Match", type="primary", use_container_width=True)

if go:
    if not resume_text.strip():
        st.warning("Add your resume first.")
        st.stop()
    if not jds:
        st.warning("Add at least one job description.")
        st.stop()

    with st.spinner("Scoring..."):
        results = match(resume_text, jds, jd_labels,
                        model_name=model_name, use_embeddings=use_embeddings)

    # ---- ranking table ----
    st.subheader("Ranking")
    if len(jds) == 1:
        st.info(
            "Only one job description, so there's nothing to rank against. The "
            "similarity number below is **not** interpretable on its own — add a "
            "second job to see a meaningful comparison. Skill coverage, however, "
            "is a real fraction and does mean something by itself.")

    table = pd.DataFrame([{
        "Job": r.jd_label,
        "Skill coverage": f"{r.skill_coverage:.0%}",
        "Embedding sim.": f"{r.embedding_score:.3f}" if use_embeddings else "—",
        "TF-IDF sim.": f"{r.tfidf_score:.3f}",
        "Matched": len(r.matched_skills),
        "Missing": len(r.missing_skills),
    } for r in results])
    st.dataframe(table, use_container_width=True, hide_index=True)

    if use_embeddings and len(results) > 1:
        spread = results[0].embedding_score - results[-1].embedding_score
        if spread < 0.05:
            st.caption(
                f"⚠️ The top and bottom jobs differ by only {spread:.3f} in embedding "
                "similarity. That's within noise — treat this ranking as weak evidence. "
                "Skill coverage is the more trustworthy signal here.")

    # ---- per-job detail ----
    st.subheader("Skill gap")
    for r in results:
        with st.expander(f"**{r.jd_label}** — {r.skill_coverage:.0%} of required skills present",
                         expanded=(r is results[0])):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**✅ You have**")
                st.write(", ".join(r.matched_skills) if r.matched_skills else "_none detected_")
            with c2:
                st.markdown("**❌ Missing**")
                st.write(", ".join(r.missing_skills) if r.missing_skills else "_none — good coverage_")
            with c3:
                st.markdown("**➕ Extra (not asked for)**")
                st.write(", ".join(r.extra_skills[:15]) if r.extra_skills else "_none_")

            if r.missing_skills:
                st.caption(
                    "These are skills named in the job description that don't appear "
                    "in your resume. Some may be genuine gaps; others you may have but "
                    "phrased differently — which is itself worth fixing, since keyword "
                    "screening is real.")

    # ---- honesty footer ----
    with st.expander("What this tool does and doesn't do"):
        st.markdown("""
- **Skill matching is keyword-based.** It looks for a fixed list of ~%d skills
  from `skills_vocab.py`. It cannot spot a skill that isn't in that list, and it
  cannot tell "I have used PyTorch" apart from "familiarity with PyTorch preferred".
- **Similarity scores are relative, not absolute.** Use them to compare jobs
  against each other, not as a "you are an X%% match" verdict.
- **It doesn't know about seniority, location, or salary** — a perfect skill
  match on a job requiring 10 years' experience is still not a match.
- **PDF extraction fails on scanned resumes.** No text layer, nothing to read.
        """ % len(SKILLS))

else:
    st.info("Add a resume and at least two job descriptions, then hit **Match**.")
    with st.expander("Try it with sample data"):
        st.markdown("""
Paste this as your resume:

```
Abhishikta Dutta — CSE student. Python, Java, C. PyTorch, HuggingFace
Transformers, Scikit-learn. Pandas, NumPy, Matplotlib, Seaborn. Machine
learning, deep learning, NLP. Fine-tuned BERT and DistilBERT for misinformation
classification. Built end-to-end ML pipelines. Git, Linux, Jupyter.
```

And this as your job descriptions:

```
ML Engineer Intern
Looking for Python, PyTorch, deep learning, NLP experience. Docker and AWS a plus.

---

Data Analyst
Requires SQL, Excel, Power BI, Tableau. Strong communication skills.
Experience with data visualization and stakeholder management.

---

Backend Developer
Node.js, React, PostgreSQL, REST API design, Docker, Kubernetes.
```

The ML role should rank first, and the skill gap for the analyst role should
show SQL and Power BI as missing — which is correct, and useful.
        """)
