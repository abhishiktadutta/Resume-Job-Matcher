# 📄 Resume ↔ Job Description Matching with Sentence Embeddings

### An Evaluation of Zero-Shot Semantic Similarity for Candidate–Job Fit

**Author:** Abhishikta Dutta
**Institution:** Siliguri Institute of Technology (MAKAUT)
**Project Type:** Applied NLP / Information Retrieval

---

## 📌 Abstract

This project evaluates whether off-the-shelf sentence embeddings can predict the fit between a resume and a job description, and ships the result as an interactive application. Using 8,000 labelled resume–job pairs, a frozen `all-MiniLM-L6-v2` encoder was evaluated zero-shot against two metrics: pooled AUC, and a within-job pairwise ranking accuracy that controls for job identity. The headline result is a gap between them. Pooled AUC reaches 0.631 (95% CI 0.594–0.669), suggesting usable performance, but within-job accuracy — the setting in which the tool is actually applied — falls to 0.556 (95% CI 0.510–0.600), barely above chance. The difference indicates that most apparent performance derives from between-job variation rather than genuine candidate discrimination. A parallel keyword-based skill-matching baseline was found to be indistinguishable from chance (AUC 0.502) and is reported as a negative result. All confidence intervals are cluster-bootstrapped over job descriptions rather than rows, because the dataset contains substantial pair non-independence that naive intervals would conceal.

---

## 📊 Results

Evaluated zero-shot on [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit), train and test pooled — 8,000 pairs, reduced to 6,000 after dropping the ambiguous *Potential Fit* class, yielding 2,000 good-fit and 4,000 no-fit pairs across **350 unique job descriptions**.

| Metric | Score | 95% CI | Interpretation |
|---|---|---|---|
| Embedding similarity — pooled AUC | **0.631** | 0.594 – 0.669 | Real but modest |
| Embedding similarity — within-job accuracy | **0.556** | 0.510 – 0.600 | Barely above chance |
| Keyword skill coverage — pooled AUC | 0.502 | — | Indistinguishable from chance |

**Model:** `all-MiniLM-L6-v2`, no fine-tuning.
**Within-job comparisons:** 37,526.
**Bootstrap:** 1,000 resamples, clustered by job description.

### Why the two embedding numbers differ

Pooled AUC allows the model to earn credit from **between-job variation** — some postings attract higher similarity from every resume, owing to their length, vocabulary, or generality. This inflates the pooled figure while saying nothing about whether the model can distinguish the better of two candidates.

Within-job pairwise accuracy removes this by only comparing resumes competing for the *same* posting. Performance falls most of the way back to chance.

> **A headline metric that appears acceptable while collapsing on the subgroup that matters is the central finding of this project.** Any deployment decision should be made on 0.556, not 0.631.

---

## 🔍 Error Analysis

Four problems identified by tracing anomalous results backwards. Documented in full, because the diagnostic process is a substantive part of the work.

### 1. Single-class evaluation sample

The initial evaluation drew the first 300 rows of an unshuffled dataset and obtained only one label. All metrics were undefined; the sole visible symptom was a histogram legend listing a single class. Resolved by shuffling prior to sampling.

### 2. Keyword skill matching does not work

**65.3%** of pairs had *zero* detected skill overlap, and the median job description yielded only **2** recognised skills. With a denominator of 2, coverage can take essentially only the values 0, 0.5, or 1.0 — the spikes at exactly 0.0 and 1.0 in the score distribution are a rounding artifact, not genuine perfect matches. AUC 0.502 confirms the metric carries no information. Retained and reported as a negative result rather than removed.

### 3. Frequency mining cannot construct the vocabulary

Mining frequent n-grams from job descriptions to expand the skill list returned generic English — *related, using, best, new, opportunity*. Skills are rare terms almost by definition, so raw frequency surfaces filler instead. The data-driven shortcut fails; this task requires hand-curation or an established taxonomy such as ESCO or O\*NET.

### 4. Non-independent pairs

The initial 300-row sample contained only **61 unique job descriptions**, each recurring roughly five times against different resumes. Treating 225 such pairs as independent substantially understates uncertainty. At that scale AUC was 0.587 with a 95% CI of 0.489–0.698 — straddling chance. Scaling to 350 clusters produced the tighter interval reported above. All intervals in this project resample job descriptions rather than rows.

### A hypothesis that was tested and rejected

Job descriptions were suspected of sharing large blocks of boilerplate — equal-opportunity statements, benefits listings — that would dominate the embedding and compress all similarity scores into a narrow band.

**Rejected.** Similarity between two random job descriptions averaged **0.381** (5th–95th percentile 0.212–0.555), and EEO language appeared in only **4.7%** of postings. No shared-text effect exists; the weak separation has another cause. Recorded here because a rejected hypothesis is part of the evidence.

---

## 🧪 Methodology

**Similarity.** Two methods, both reported:

- *TF-IDF cosine* — fast, model-free, but restricted to literally shared words. Scores exactly 0 against job descriptions using different vocabulary.
- *Embedding cosine* (`all-MiniLM-L6-v2`) — captures paraphrase.

**Skill extraction.** Regex matching against a hand-curated vocabulary mapping canonical skills to surface forms (`sklearn` → Scikit-learn). Word-boundary matching uses a custom class including `+` and `#`, so the skill "C" does not match inside "C++" — plain `\b` fails here because `+` is already a non-word character, silently crediting every C++ developer with C. The class excludes `/` so that "R/Python" still matches R. Both cases are unit-tested in the notebook.

**Evaluation.** Pooled AUC plus within-job pairwise ranking accuracy, with 1,000-iteration bootstrap confidence intervals clustered by job description.

> **Note on pooling splits.** Train and test are pooled for evaluation. This would ordinarily be a serious error; it is valid here only because no parameters are fitted — the encoder is frozen and never observes a label.

---

## 💡 Application Design

- Resume input by pasted text or PDF upload
- **Ranks** multiple job descriptions rather than scoring one in isolation
- Per-job skill breakdown: matched, missing, extra, with coverage percentage

### Why there is no "78% match" score

Raw cosine similarity between sentence embeddings is not calibrated. Unrelated job descriptions in this dataset score ~0.38 against one another, and resume–JD pairs occupy a similar band. Presenting that as a percentage would imply a precision the number does not possess, to someone making consequential decisions about their job search.

The similarity is meaningful *comparatively*, so the application reports a ranking. The within-job result is the direct justification: comparison within a job is the only setting where the score carries signal — and even there it is weak.

---

## 📁 Repository Structure

| File | Purpose |
|---|---|
| `resume_matcher.ipynb` | Development and full evaluation. **Start here.** |
| `matcher.py` | Skill extraction, TF-IDF and embedding similarity |
| `skills_vocab.py` | Skill vocabulary |
| `app.py` | Streamlit application |
| `requirements.txt` | Dependencies |

Core logic sits outside the application deliberately, so that it can be imported into the notebook and measured. Logic embedded in UI callbacks can only be demonstrated, never evaluated.

---

## ⚙️ Installation & Usage

```bash
git clone https://github.com/abhishiktadutta/Resume-Job-Matcher.git
cd Resume-Job-Matcher
pip install -r requirements.txt
streamlit run app.py
```

The notebook runs in Google Colab. The full evaluation requires a GPU runtime and completes in approximately two minutes on a T4.

---

## ⚠️ Limitations

- **Keyword skill matching cannot detect paraphrase.** A resume stating `ML/DL` receives no credit for a posting requesting `deep learning`.
- **No representation of depth.** "Used PyTorch in a tutorial" and "three years of PyTorch" score identically.
- **512-token truncation** means the latter half of a two-page resume is ignored entirely.
- **PDF extraction fails on scanned documents** — no text layer exists. The application warns rather than silently scoring zero.
- **No awareness of seniority, location, or compensation.**
- **Labels are dataset-provided.** "Good fit" is itself a judgement whose inter-rater reliability and provenance were not independently verified — this caps the interpretability of every number above.

---

## 🚀 Future Work

1. **Chunk long resumes** rather than truncating, taking maximum similarity per requirement
2. **Evaluate a retrieval-tuned encoder** (`BAAI/bge-small-en-v1.5`) — re-measuring the *within-job* metric, since pooled AUC will mislead
3. **Fine-tune on the labelled pairs**; zero-shot performance is a floor, not a ceiling
4. **Replace the hand-written skill list** with an established taxonomy (ESCO, O\*NET)
5. **Model the between-job variance explicitly** rather than only controlling for it

---

## 📚 Data

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) — resume–job description pairs labelled *No Fit* / *Potential Fit* / *Good Fit*.
