"""
app.py — Streamlit web interface for the Resume Job Matching System.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    COSINE_WEIGHT,
    DEFAULT_TOP_K,
    RESUME_SCORE_WEIGHTS,
    SAMPLE_RESUMES_DIR,
    SKILL_WEIGHT,
)
from src.matcher import (
    JobMatch,
    compute_resume_score,
    match_jobs,
    results_to_dataframe,
)
from src.preprocess import clean_jobs_dataframe
from src.resume_parser import parse_resume, ParsedResume, validate_is_resume
from src.utils import NotAResumeError
from src.scraper import load_jobs
from src.skills import extract_skills, skills_by_category
from src.vectorizer import TfidfVectorizerNP

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3347;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Score bar */
    .score-bar-container { margin: 4px 0; }
    .score-bar-label { font-size: 0.85rem; color: #aab4c4; }
    .score-bar-outer {
        background: #1e2130;
        border-radius: 6px;
        height: 10px;
        width: 100%;
    }
    .score-bar-inner {
        background: linear-gradient(90deg, #4f8ef7, #7c5cbf);
        border-radius: 6px;
        height: 10px;
    }

    /* Skill pills */
    .skill-matched {
        display: inline-block;
        background: #1a3a2a;
        color: #4caf84;
        border: 1px solid #4caf84;
        border-radius: 20px;
        padding: 2px 10px;
        margin: 3px;
        font-size: 0.8rem;
    }
    .skill-missing {
        display: inline-block;
        background: #3a1a1a;
        color: #f47b7b;
        border: 1px solid #f47b7b;
        border-radius: 20px;
        padding: 2px 10px;
        margin: 3px;
        font-size: 0.8rem;
    }

    /* Section headers */
    h1 { color: #e8eaf6; }
    h2 { color: #c5cae9; }
    h3 { color: #9fa8da; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading job data …")
def _load_and_clean(source: str) -> pd.DataFrame:
    raw_df = load_jobs(source=source)
    return clean_jobs_dataframe(raw_df)


@st.cache_resource(show_spinner="Fitting TF-IDF vectoriser …")
def _fit_vectorizer(corpus_key: str, corpus: tuple[str, ...]) -> TfidfVectorizerNP:
    """Fit vectoriser on corpus (corpus passed as tuple for hashability)."""
    vec = TfidfVectorizerNP()
    vec.fit(list(corpus))
    return vec


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skill_pills(skills: frozenset[str], css_class: str) -> str:
    pills = "".join(
        f'<span class="{css_class}">{s}</span>'
        for s in sorted(skills)
    )
    return pills or "<em>None</em>"


def _score_bar(label: str, value: float, max_val: float) -> str:
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div class="score-bar-container">'
        f'<span class="score-bar-label">{label}: {value:.1f}</span>'
        f'<div class="score-bar-outer"><div class="score-bar-inner" style="width:{pct:.0f}%"></div></div>'
        f'</div>'
    )


def _bar_chart(matches: list[JobMatch]) -> plt.Figure:
    """Horizontal bar chart of final scores for top matches."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(matches) * 0.45)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1e2130")

    labels = [f"[{m.rank}] {m.title[:28]}" for m in reversed(matches)]
    cosine_vals = [m.cosine_score for m in reversed(matches)]
    skill_vals = [m.skill_score for m in reversed(matches)]
    y = np.arange(len(labels))
    bar_h = 0.35

    ax.barh(y + bar_h / 2, cosine_vals, bar_h, label="Cosine", color="#4f8ef7", alpha=0.85)
    ax.barh(y - bar_h / 2, skill_vals, bar_h, label="Skill Ratio", color="#7c5cbf", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color="#c5cae9")
    ax.set_xlabel("Score", color="#aab4c4")
    ax.set_xlim(0, 1.05)
    ax.tick_params(colors="#aab4c4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3347")
    ax.legend(fontsize=8, facecolor="#1e2130", labelcolor="#c5cae9")
    ax.set_title("Score Breakdown — Top Matches", color="#e8eaf6", fontsize=10)
    plt.tight_layout()
    return fig


def _skill_gap_chart(matches: list[JobMatch]) -> plt.Figure:
    """Aggregated skill gap chart: top missing skills across all matches."""
    from collections import Counter
    counter: Counter = Counter()
    for m in matches:
        counter.update(m.missing_skills)

    if not counter:
        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1e2130")
        ax.text(0.5, 0.5, "No skill gaps found! 🎉", ha="center", va="center",
                color="#4caf84", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        return fig

    top_missing = counter.most_common(12)
    labels = [s for s, _ in reversed(top_missing)]
    counts = [c for _, c in reversed(top_missing)]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.4)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1e2130")

    bars = ax.barh(labels, counts, color="#f47b7b", alpha=0.8)
    ax.set_xlabel("# of job postings requiring this skill", color="#aab4c4")
    ax.tick_params(colors="#aab4c4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3347")
    ax.set_title("Skill Gap — Top Missing Skills Across Matches", color="#e8eaf6", fontsize=10)
    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _build_sidebar() -> dict:
    st.sidebar.image(
        "https://via.placeholder.com/280x60/1a1a2e/4f8ef7?text=Resume+Job+Matcher",
        use_container_width=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.header("⚙️ Settings")

    source = st.sidebar.selectbox(
        "📡 Data source",
        ["offline", "scrape", "both"],
        index=0,
        help="'offline' uses the bundled 60+ sample jobs. 'scrape' tries the live site.",
    )

    top_k = st.sidebar.slider("🏆 Top-K results", min_value=3, max_value=30, value=DEFAULT_TOP_K)

    st.sidebar.markdown("**Score weights** (must sum to 1.0)")
    cosine_w = st.sidebar.slider(
        "Cosine similarity weight", 0.1, 0.9, COSINE_WEIGHT, 0.05
    )
    skill_w = round(1.0 - cosine_w, 2)
    st.sidebar.caption(f"Skill overlap weight: **{skill_w}**")

    location_filter = st.sidebar.text_input(
        "📍 Location preference (bonus)",
        placeholder="e.g. New York",
        help="Jobs matching this location receive a small score bonus.",
    )

    skill_filter = st.sidebar.text_input(
        "🔧 Filter by required skill",
        placeholder="e.g. python",
        help="Only show results that contain this skill in job description.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**ℹ️ About**\n\n"
        "Ranks jobs using **TF-IDF** (pure NumPy) + **skill overlap**.\n\n"
        "No scikit-learn. No spaCy. 100% from scratch. 🚀"
    )

    return {
        "source": source,
        "top_k": top_k,
        "cosine_weight": cosine_w,
        "skill_weight": skill_w,
        "location_filter": location_filter,
        "skill_filter": skill_filter,
    }


# ── Resume upload section ─────────────────────────────────────────────────────

def _show_not_a_resume_error(exc: NotAResumeError, filename: str) -> None:
    """Render a detailed, user-friendly error when the document is not a resume."""
    confidence_pct = int(exc.confidence * 100)
    st.markdown(
        f"""
        <div style="background:#2a0a0a; border:1px solid #f47b7b; border-radius:12px; padding:1.2rem; margin:0.5rem 0">
          <h3 style="color:#f47b7b; margin:0 0 0.5rem 0">❌ Not a Resume Detected</h3>
          <p style="color:#e0c0c0; margin:0 0 0.3rem 0">
            <strong>{filename}</strong> does not appear to be a resume or CV.
            Resume confidence score: <strong>{confidence_pct}%</strong>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if exc.hints:
        st.markdown("**Why was it rejected?**")
        for hint in exc.hints:
            st.markdown(f"&nbsp;&nbsp;⚠️ {hint}")
    st.markdown("**What should a resume contain?**")
    st.markdown(
        "- Your **name** and **contact info** (email, phone)\n"
        "- An **Experience** or **Work History** section\n"
        "- An **Education** section\n"
        "- A **Skills** section\n"
        "- Employment date ranges (e.g. 2020 – 2023)"
    )


def _resume_section() -> ParsedResume | None:
    st.header("📄 Resume Input")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT",
        )

    with col2:
        sample_resumes = sorted(SAMPLE_RESUMES_DIR.glob("*"))
        sample_names = ["— None —"] + [f.name for f in sample_resumes]
        selected_sample = st.selectbox("Or use a sample resume", sample_names)

    resume: ParsedResume | None = None

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        tmp_path = Path(tempfile.gettempdir()) / f"uploaded_resume{suffix}"
        tmp_path.write_bytes(uploaded_file.getvalue())
        try:
            with st.spinner("Parsing and validating resume …"):
                resume = parse_resume(tmp_path)
            confidence_pct = int((resume.validation.confidence if resume.validation else 1.0) * 100)
            st.success(f"✅ Parsed: **{uploaded_file.name}**  ·  Resume confidence: **{confidence_pct}%**")
        except NotAResumeError as exc:
            _show_not_a_resume_error(exc, uploaded_file.name)
            return None
        except Exception as exc:
            st.error(f"❌ Failed to parse resume: {exc}")
            return None

    elif selected_sample != "— None —":
        sample_path = SAMPLE_RESUMES_DIR / selected_sample
        try:
            with st.spinner(f"Parsing {selected_sample} …"):
                resume = parse_resume(sample_path)
            st.info(f"📁 Using sample resume: **{selected_sample}**")
        except NotAResumeError as exc:
            _show_not_a_resume_error(exc, selected_sample)
            return None
        except Exception as exc:
            st.error(f"❌ Failed to parse sample resume: {exc}")
            return None

    return resume


# ── Resume summary display ────────────────────────────────────────────────────

def _show_resume_summary(resume: ParsedResume) -> None:
    st.subheader("👤 Parsed Resume Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Name", resume.name or "Not detected")
    col2.metric("Email", resume.email or "Not detected")
    col3.metric("Phone", resume.phone or "Not detected")
    yoe = f"{resume.years_of_experience:.0f} yrs" if resume.years_of_experience else "N/A"
    col4.metric("Experience", yoe)
    if resume.validation:
        conf_pct = int(resume.validation.confidence * 100)
        col5.metric("Resume Confidence", f"{conf_pct}%")

    st.markdown(f"**Sections detected:** {', '.join(resume.sections.keys()) or 'None'}")

    # Skills detected
    resume_skills = extract_skills(resume.full_text)
    by_cat = skills_by_category(resume_skills)
    if resume_skills:
        with st.expander(f"🔧 Detected Skills ({len(resume_skills)})"):
            for cat, skills in by_cat.items():
                cat_name = cat.replace("_", " ").title()
                pills = " ".join(f'<span class="skill-matched">{s}</span>' for s in skills)
                st.markdown(f"**{cat_name}:** {pills}", unsafe_allow_html=True)
    else:
        st.warning("No skills detected. Try a resume with clearer skill listings.")


def _show_resume_score(resume: ParsedResume) -> None:
    score = compute_resume_score(resume)
    st.subheader(f"📊 Resume Score: **{score.total:.0f} / 100**")

    # Gauge-like metric
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall", f"{score.total:.0f}%")
        grade = "A" if score.total >= 80 else "B" if score.total >= 65 else "C" if score.total >= 50 else "D"
        st.markdown(f"**Grade: {grade}**")

    with col2:
        bars_html = "".join(
            _score_bar(dim.replace("_", " ").title(), val, RESUME_SCORE_WEIGHTS[dim])
            for dim, val in score.breakdown.items()
        )
        st.markdown(bars_html, unsafe_allow_html=True)

    with st.expander("💡 Improvement Tips"):
        for tip in score.tips:
            st.markdown(f"• {tip}")


# ── Results display ───────────────────────────────────────────────────────────

def _show_results(
    matches: list[JobMatch],
    skill_filter: str,
    jobs_df: pd.DataFrame,
) -> None:
    # Apply skill filter
    if skill_filter.strip():
        sf = skill_filter.strip().lower()
        filtered = [
            m for m in matches
            if sf in m.description.lower() or sf in m.title.lower()
            or any(sf in s for s in m.all_job_skills)
        ]
        if not filtered:
            st.warning(f"No results contain skill: '{skill_filter}'. Showing all.")
        else:
            matches = filtered
            st.info(f"Filtered to {len(matches)} results containing '{skill_filter}'.")

    if not matches:
        st.warning("No matches found. Try adjusting the settings.")
        return

    st.subheader(f"🏆 Top {len(matches)} Job Matches")

    # Summary table
    df_results = results_to_dataframe(matches)
    st.dataframe(
        df_results[["Rank", "Title", "Company", "Location", "Final Score", "Cosine Score", "Skill Score"]],
        use_container_width=True,
        hide_index=True,
    )

    # CSV download
    csv_bytes = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results CSV",
        csv_bytes,
        file_name="job_matches.csv",
        mime="text/csv",
    )

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Score Breakdown")
        fig = _bar_chart(matches)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("❌ Skill Gap Analysis")
        fig2 = _skill_gap_chart(matches)
        st.pyplot(fig2)
        plt.close(fig2)

    # Per-job expanders
    st.subheader("📋 Detailed Results")
    for m in matches:
        score_badge = f"{m.final_score:.3f}"
        with st.expander(f"#{m.rank} {m.title} @ {m.company}  —  Score: {score_badge}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Score", f"{m.final_score:.3f}")
            c2.metric("Cosine", f"{m.cosine_score:.3f}")
            c3.metric("Skill Ratio", f"{m.skill_score:.3f}")

            st.markdown(f"**📍 Location:** {m.location}")

            if m.apply_url:
                st.markdown(f"**🔗 Apply:** [{m.apply_url}]({m.apply_url})")

            st.markdown("**📝 Job Description:**")
            st.caption(m.description[:500] + ("…" if len(m.description) > 500 else ""))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**✅ Matched Skills:**")
                st.markdown(_skill_pills(m.matched_skills, "skill-matched"), unsafe_allow_html=True)
            with col_b:
                st.markdown("**❌ Missing Skills (Skill Gap):**")
                st.markdown(_skill_pills(m.missing_skills, "skill-missing"), unsafe_allow_html=True)


# ── Main app ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🔍 Resume Job Matcher")
    st.markdown(
        "Powered by **TF-IDF** (pure NumPy) + **skill overlap scoring**. "
        "Upload your resume and discover your best-fit jobs instantly."
    )

    settings = _build_sidebar()

    # ── Load jobs ──────────────────────────────────────────────────────────
    try:
        jobs_df = _load_and_clean(settings["source"])
    except Exception as exc:
        st.error(f"Failed to load job data: {exc}")
        return

    st.sidebar.caption(f"📦 {len(jobs_df)} jobs loaded")

    # ── Resume section ─────────────────────────────────────────────────────
    resume = _resume_section()

    if resume is None:
        st.info("👆 Upload a resume or select a sample resume to get started.")
        st.stop()

    st.divider()
    _show_resume_summary(resume)

    st.divider()
    _show_resume_score(resume)

    st.divider()

    # ── Run matching ───────────────────────────────────────────────────────
    with st.spinner("🧠 Running TF-IDF matching and skill analysis …"):
        text_col = "cleaned_full_text" if "cleaned_full_text" in jobs_df.columns else "description"
        corpus_tuple = tuple(jobs_df[text_col].fillna("").tolist())
        vectorizer = _fit_vectorizer(settings["source"], corpus_tuple)

        matches, _ = match_jobs(
            resume=resume,
            jobs_df=jobs_df,
            top_k=settings["top_k"],
            cosine_weight=settings["cosine_weight"],
            skill_weight=settings["skill_weight"],
            location_preference=settings["location_filter"],
            vectorizer=vectorizer,
        )

    _show_results(matches, settings["skill_filter"], jobs_df)


if __name__ == "__main__":
    main()
