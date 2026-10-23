"""
main.py — CLI for the Resume Job Matching System.

Usage
-----
  python main.py --resume path/to/resume.pdf --top 10 --source offline
  python main.py --resume path/to/resume.pdf --top 5 --source scrape --export results.csv
  python main.py --resume path/to/resume.pdf --source both --location "New York"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.config import DEFAULT_TOP_K, COSINE_WEIGHT, SKILL_WEIGHT
from src.matcher import match_jobs, compute_resume_score, results_to_dataframe
from src.preprocess import load_and_clean_jobs
from src.resume_parser import parse_resume
from src.scraper import load_jobs
from src.utils import get_logger

logger = get_logger("main")


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _separator(char: str = "─", width: int = 90) -> str:
    return char * width


def _print_header() -> None:
    print(_separator("═"))
    print(" 🔍  RESUME JOB MATCHER — AI-Powered Job Ranking System")
    print(_separator("═"))


def _print_resume_summary(resume) -> None:
    print(f"\n📄  Resume: {Path(resume.file_path).name}")
    print(f"   Name   : {resume.name or 'Not detected'}")
    print(f"   Email  : {resume.email or 'Not detected'}")
    print(f"   Phone  : {resume.phone or 'Not detected'}")
    print(f"   Sections detected : {', '.join(resume.sections.keys()) or 'None'}")
    if resume.years_of_experience:
        print(f"   Est. experience   : {resume.years_of_experience} years")


def _print_resume_score(score) -> None:
    print(f"\n📊  Resume Score: {score.total:.0f} / 100")
    for dim, val in score.breakdown.items():
        bar = "█" * int(val) + "░" * int(10 - val / 10)
        print(f"   {dim:<20} {val:5.1f}  {bar}")
    print("\n💡  Improvement tips:")
    for tip in score.tips:
        print(f"   • {tip}")


def _print_results(matches: list, top_k: int) -> None:
    print(f"\n🏆  Top {len(matches)} Job Matches")
    print(_separator())
    header = (
        f"{'#':>3}  {'Score':>6}  {'Cosine':>6}  {'Skill':>5}  "
        f"{'Title':<35}  {'Company':<25}  {'Location':<20}"
    )
    print(header)
    print(_separator("-"))

    for m in matches:
        title = m.title[:34]
        company = m.company[:24]
        location = m.location[:19]
        print(
            f"{m.rank:>3}  {m.final_score:>6.3f}  {m.cosine_score:>6.3f}  "
            f"{m.skill_score:>5.3f}  {title:<35}  {company:<25}  {location:<20}"
        )

    print(_separator())
    print("\n📌  Detailed breakdown (top 5):")
    for m in matches[:5]:
        print(f"\n  [{m.rank}] {m.title} @ {m.company}")
        print(f"      Final: {m.final_score:.3f}  |  Cosine: {m.cosine_score:.3f}  |  Skill: {m.skill_score:.3f}")
        if m.matched_skills:
            print(f"      ✅ Matched skills : {', '.join(sorted(m.matched_skills)[:8])}")
        if m.missing_skills:
            print(f"      ❌ Missing skills : {', '.join(sorted(m.missing_skills)[:8])}")
        if m.apply_url:
            print(f"      🔗 Apply : {m.apply_url}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Resume Job Matching System — rank job postings against a resume.",
    )
    p.add_argument(
        "--resume", "-r",
        required=True,
        help="Path to the resume file (.pdf, .docx, or .txt).",
    )
    p.add_argument(
        "--top", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top results to show (default: {DEFAULT_TOP_K}).",
    )
    p.add_argument(
        "--source", "-s",
        choices=["scrape", "offline", "both"],
        default="offline",
        help="Job data source: scrape live / offline sample / both (default: offline).",
    )
    p.add_argument(
        "--export", "-e",
        default=None,
        help="Optional path to export results as CSV (e.g. results.csv).",
    )
    p.add_argument(
        "--location", "-l",
        default="",
        help="Location preference string for bonus scoring (e.g. 'New York').",
    )
    p.add_argument(
        "--cosine-weight",
        type=float,
        default=COSINE_WEIGHT,
        help=f"Weight for cosine similarity (default: {COSINE_WEIGHT}).",
    )
    p.add_argument(
        "--skill-weight",
        type=float,
        default=SKILL_WEIGHT,
        help=f"Weight for skill overlap (default: {SKILL_WEIGHT}).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _print_header()

    # ── Load and parse resume ─────────────────────────────────────────────
    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"\n❌  Error: Resume file not found: {resume_path}")
        sys.exit(1)

    print(f"\n⏳  Parsing resume …")
    try:
        resume = parse_resume(resume_path)
    except Exception as exc:
        print(f"\n❌  Failed to parse resume: {exc}")
        sys.exit(1)

    _print_resume_summary(resume)

    # ── Resume score ──────────────────────────────────────────────────────
    score = compute_resume_score(resume)
    _print_resume_score(score)

    # ── Load jobs ─────────────────────────────────────────────────────────
    print(f"\n⏳  Loading job data (source={args.source}) …")
    try:
        raw_df = load_jobs(source=args.source)
    except Exception as exc:
        print(f"\n❌  Failed to load jobs: {exc}")
        sys.exit(1)

    print(f"   Loaded {len(raw_df)} job postings.")

    # ── Preprocess ────────────────────────────────────────────────────────
    print("⏳  Preprocessing job descriptions …")
    from src.preprocess import clean_jobs_dataframe
    clean_df = clean_jobs_dataframe(raw_df)

    # ── Match ─────────────────────────────────────────────────────────────
    print(f"⏳  Ranking top {args.top} matches …")
    matches, _ = match_jobs(
        resume=resume,
        jobs_df=clean_df,
        top_k=args.top,
        cosine_weight=args.cosine_weight,
        skill_weight=args.skill_weight,
        location_preference=args.location,
    )

    # ── Display results ───────────────────────────────────────────────────
    _print_results(matches, args.top)

    # ── Export ────────────────────────────────────────────────────────────
    if args.export:
        export_path = Path(args.export)
        df_results = results_to_dataframe(matches)
        df_results.to_csv(export_path, index=False, encoding="utf-8")
        print(f"\n💾  Results exported to: {export_path}")

    print(f"\n{_separator('═')}")
    print("  Done! Good luck with your job search. 🚀")
    print(_separator("═"))


if __name__ == "__main__":
    main()
