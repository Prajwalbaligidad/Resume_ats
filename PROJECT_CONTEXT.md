# Resume Analyzer & Job Matching System — Project Context

## Status: ✅ COMPLETE

All 11 phases implemented.

---

## Completed Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Skeleton, Config, Utils | ✅ Done |
| 2 | Scraper + 60+ sample jobs | ✅ Done |
| 3 | Preprocessing pipeline | ✅ Done |
| 4 | Skills taxonomy (200+ skills) | ✅ Done |
| 5 | Resume parser (PDF/DOCX/TXT) + 4 sample resumes | ✅ Done |
| 6 | NumPy TF-IDF vectoriser | ✅ Done |
| 7 | Matcher (cosine sim + skill overlap + Resume Score) | ✅ Done |
| 8 | CLI (main.py / argparse) | ✅ Done |
| 9 | Streamlit app (app.py) | ✅ Done |
| 10 | Pytest suite (7 files, 70+ tests) | ✅ Done |
| 11 | Documentation (README, report, viva-notes) | ✅ Done |

---

## Key Files

| File | Purpose |
|---|---|
| `src/config.py` | Central constants, paths, weights, alias map |
| `src/utils.py` | Logger, custom exceptions |
| `src/scraper.py` | Scraper + 60 offline jobs |
| `src/preprocess.py` | Text cleaning pipeline |
| `src/skills.py` | Skill extraction + gap analysis |
| `src/resume_parser.py` | PDF/DOCX/TXT resume parser |
| `src/vectorizer.py` | TfidfVectorizerNP (pure NumPy) |
| `src/matcher.py` | match_jobs, cosine_similarity, ResumeScore |
| `app.py` | Streamlit web UI |
| `main.py` | CLI |
| `tests/` | 7 test files |
| `README.md` | Full usage documentation |
| `docs/report.md` | Technical report |
| `docs/viva-notes.md` | Q&A for exam/interview |

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run CLI
python main.py data/sample_resumes/alice_chen_data_scientist.txt --top-k 10

# Run tests
pytest
```

---

## Bug Fixes Applied During Development

1. `app.py`: `RESUME_SCORE_WEIGHTS` was referenced but not imported → fixed.
2. `app.py`: `/tmp` hardcoded path incompatible with Windows → replaced with `tempfile.gettempdir()`.

---

## Design Decisions

- TF-IDF IDF: `log((1+N)/(1+df)) + 1` (smooth, scikit-learn compatible)
- Cosine = dot product on L2-normalised vectors (pure NumPy)
- Weights: cosine=0.6, skill=0.4 (tuned empirically on sample data)
- Skills: multi-word phrases matched before individual tokens
- NLTK: used for stopwords + stemming only; 50-word built-in fallback
- No scikit-learn, spaCy, or any TF-IDF/cosine library
