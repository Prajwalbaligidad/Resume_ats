# Resume Analyzer & Job Matching System

> **Advanced Python micro-project** — 100% from scratch. No scikit-learn. No spaCy. Pure NumPy TF-IDF + cosine similarity.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![NumPy](https://img.shields.io/badge/TF--IDF-NumPy-orange)
![pytest](https://img.shields.io/badge/tests-pytest-green)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Running the CLI](#running-the-cli)
7. [Running the Streamlit App](#running-the-streamlit-app)
8. [Running the Tests](#running-the-tests)
9. [How It Works](#how-it-works)
10. [Configuration](#configuration)
11. [Design Decisions](#design-decisions)
12. [Sample Resumes](#sample-resumes)

---

## Overview

This project builds an end-to-end **resume-to-job matching system** entirely from first principles. It:

- **Scrapes** real job postings from a public job board (with an offline fallback of 60+ hand-crafted entries).
- **Parses** resumes in PDF, DOCX, or TXT format.
- **Pre-processes** text using custom regex cleaning, alias normalisation, and optional Porter stemming — no NLTK vectorisers allowed.
- **Vectorises** documents using a **manually implemented TF-IDF** on top of NumPy.
- **Ranks** jobs by a weighted combination of **cosine similarity** and **skill overlap ratio**.
- Presents results in a **Streamlit web app** and a fully-featured **CLI**.

---

## Features

| Feature | Detail |
|---|---|
| 🔍 TF-IDF (NumPy) | Smooth IDF formula `log((1+N)/(1+df))+1`, L2 row normalisation |
| 🧠 Cosine Similarity | Pure matrix-vector dot product on normalised vectors |
| 🎯 Skill Matching | 200+ skill taxonomy across 13 categories, multi-word phrases first |
| 📄 Resume Parser | PDF (pdfplumber), DOCX (python-docx), TXT; extracts name/email/phone/sections |
| 📊 Resume Score | 0-100 quality score with grade and improvement tips |
| 🌐 Web Scraper | HTTP retries, exponential back-off, robots.txt respect |
| 📦 60+ Sample Jobs | Offline dataset spanning 13 domains/career paths |
| 🖥️ Streamlit UI | Charts, skill pills, skill gap, CSV download |
| ⌨️ CLI | argparse, tabulate output, CSV export |
| ✅ Tests | 7 test files, 70+ test cases (unit + integration + E2E) |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `numpy` | TF-IDF vectors, cosine similarity, all maths |
| `pandas` | Jobs DataFrame pipeline |
| `requests` + `beautifulsoup4` + `lxml` | Web scraping |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `nltk` | Stopwords + Porter stemmer (with built-in fallback) |
| `streamlit` | Web interface |
| `matplotlib` | Score breakdown & skill gap charts |
| `pytest` | Test suite |

> ❌ **Not used:** scikit-learn, spaCy, gensim, or any library providing TF-IDF or cosine similarity.

---

## Project Structure

```
resume_job_matcher/
├── app.py                       # Streamlit web app
├── main.py                      # CLI entrypoint
├── generate_sample_resumes.py   # Generate PDF/DOCX sample resumes
├── requirements.txt
├── pytest.ini
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Central config: paths, weights, alias map
│   ├── utils.py                 # Logger, custom exceptions, helpers
│   ├── scraper.py               # Web scraper + 60+ sample jobs
│   ├── preprocess.py            # Text cleaning, tokenisation, stemming
│   ├── skills.py                # Skills extraction & gap analysis
│   ├── resume_parser.py         # PDF/DOCX/TXT parser → ParsedResume
│   ├── vectorizer.py            # TfidfVectorizerNP (pure NumPy)
│   └── matcher.py               # Cosine sim, skill ratio, Resume Score
│
├── data/
│   ├── skills.json              # 200+ skills in 13 categories
│   ├── raw/                     # Scraped/generated job CSVs
│   ├── processed/               # Cleaned job CSVs
│   └── sample_resumes/          # 4 TXT + optional PDF/DOCX samples
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/fake_jobs.html
│   ├── test_preprocess.py
│   ├── test_vectorizer.py
│   ├── test_matcher.py
│   ├── test_skills.py
│   ├── test_resume_parser.py
│   ├── test_scraper.py
│   └── test_e2e.py
│
├── docs/
│   ├── report.md
│   └── viva-notes.md
└── logs/
```

---

## Quick Start

### 1. Clone / unzip the project

```bash
cd resume_job_matcher
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (optional — built-in fallback exists)

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## Running the CLI

```bash
python main.py path/to/resume.pdf
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `resume` | (required) | Path to your resume (PDF/DOCX/TXT) |
| `--top-k` | 10 | Number of top job matches to show |
| `--source` | `offline` | `offline`, `scrape`, or `both` |
| `--location` | `` | Location preference (bonus scoring) |
| `--cosine-weight` | `0.6` | Weight for cosine similarity (0–1) |
| `--skill-weight` | `0.4` | Weight for skill overlap (0–1) |
| `--output-csv` | `` | Path to export results as CSV |

### Examples

```bash
# Use sample resume, top 5 results
python main.py data/sample_resumes/alice_chen_data_scientist.txt --top-k 5

# Scrape live jobs, prefer New York, export CSV
python main.py resume.pdf --source scrape --location "New York" --output-csv results.csv

# Adjust weights
python main.py resume.docx --cosine-weight 0.7 --skill-weight 0.3
```

---

## Running the Streamlit App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### App Features

- **Upload** a PDF/DOCX/TXT resume, or select from the 4 built-in samples.
- **Resume Summary** — detected name, email, phone, years of experience, skills by category.
- **Resume Score** — 0-100 grade with breakdown bars and improvement tips.
- **Job Matches** — ranked table with cosine score, skill ratio, and final score.
- **Score Breakdown Chart** — horizontal bar chart comparing cosine vs skill scores.
- **Skill Gap Chart** — top missing skills across all matched jobs.
- **Detailed Expanders** — per-job matched/missing skill pills and apply link.
- **CSV Download** — export all results with one click.
- **Sidebar** — adjust data source, top-K, score weights, location filter, skill filter.

---

## Running the Tests

```bash
# All tests
pytest

# With verbose output
pytest -v

# Specific module
pytest tests/test_vectorizer.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

### Test Coverage

| Module | Tests |
|---|---|
| `test_preprocess.py` | 20 tests — cleaning, tokenise, stopwords, alias, pipeline |
| `test_vectorizer.py` | 17 tests — vocab, IDF formula, unit norms, save/load, bigrams |
| `test_matcher.py` | 15 tests — cosine similarity, skill ratio, ranking, resume score |
| `test_skills.py` | 14 tests — extraction, multi-word, aliases, gap, categories |
| `test_resume_parser.py` | 14 tests — all 4 sample resumes, edge cases |
| `test_scraper.py` | 11 tests — parsing, 60+ samples, CSV, merge, mocked HTTP |
| `test_e2e.py` | 5 tests — end-to-end ranking correctness |

---

## How It Works

### 1. Text Preprocessing

```
raw text → lowercase → remove URLs/emails/phones/digits/punctuation
        → tokenise → remove stopwords → apply alias map → (optional) stem
```

Key alias examples: `js → javascript`, `k8s → kubernetes`, `ml → machine learning`.

### 2. TF-IDF (Manual NumPy)

**Term Frequency:**
```
tf(t, d) = count(t in d) / len(d)
```

**Inverse Document Frequency (smooth):**
```
idf(t) = log( (1 + N) / (1 + df(t)) ) + 1
```

**TF-IDF + L2 normalisation:**
```
tfidf(t, d) = tf(t,d) × idf(t)
row_i ← row_i / ||row_i||_2
```

### 3. Cosine Similarity

Since all vectors are L2-normalised, cosine similarity reduces to a dot product:

```
cosine(resume, job_i) = resume_vec · job_vec_i
```

Implemented as: `job_matrix @ resume_vec` — a single NumPy matrix-vector multiply.

### 4. Final Score

```
final_score = w_cosine × cosine_sim
            + w_skill  × skill_ratio
            + location_bonus  (if location matched)
```

Default weights: `w_cosine = 0.6`, `w_skill = 0.4`.

Skill ratio = `|resume_skills ∩ job_skills| / max(|job_skills|, 1)`.

---

## Configuration

All constants live in [`src/config.py`](src/config.py):

| Constant | Default | Description |
|---|---|---|
| `COSINE_WEIGHT` | `0.6` | Weight for cosine similarity |
| `SKILL_WEIGHT` | `0.4` | Weight for skill overlap |
| `DEFAULT_TOP_K` | `10` | Default number of results |
| `LOCATION_BONUS` | `0.05` | Bonus score for location match |
| `SCRAPE_URL` | fake-jobs URL | Target job board |
| `MAX_RETRIES` | `3` | HTTP retry count |
| `BACKOFF_FACTOR` | `2` | Exponential back-off multiplier |
| `ALIAS_MAP` | dict | Token alias normalisations |
| `RESUME_SCORE_WEIGHTS` | dict | Weights for each resume dimension |

---

## Design Decisions

1. **No scikit-learn TF-IDF** — implemented from scratch with NumPy broadcasting for performance.
2. **No spaCy** — NLTK stopwords only; custom suffix-stripping fallback if NLTK unavailable.
3. **Multi-word skills first** — phrases like "machine learning" matched before individual tokens to avoid double-counting.
4. **Smooth IDF** — avoids division-by-zero; rare terms always get higher IDF than frequent ones.
5. **L2 normalisation** — enables cosine = dot product; avoids recomputing norms each time.
6. **Offline fallback** — 60+ hand-crafted jobs across 13 domains ensure the app works with no internet.
7. **Alias normalisation** — unifies `js`/`javascript`, `k8s`/`kubernetes` etc. at preprocessing time.
8. **Streamlit caching** — `@st.cache_data` for jobs loading, `@st.cache_resource` for vectoriser to avoid re-fitting on every rerun.

---

## Sample Resumes

| File | Profile |
|---|---|
| `alice_chen_data_scientist.txt` | Senior data scientist, ML/Python/SQL |
| `marcus_johnson_web_developer.txt` | Full-stack JS developer, React/Node.js |
| `priya_sharma_devops.txt` | DevOps engineer, K8s/AWS/Terraform |
| `jordan_rivera_marketing.txt` | Digital marketing manager, SEO/SEM |

---

## Licence

MIT — free to use, modify, and distribute.
