"""
generate_sample_resumes.py — One-time script to create PDF and DOCX sample resumes.
Run: python generate_sample_resumes.py
"""
from pathlib import Path

RESUMES_DIR = Path(__file__).parent / "data" / "sample_resumes"
RESUMES_DIR.mkdir(parents=True, exist_ok=True)

# ── DOCX resume: Data Engineer ────────────────────────────────────────────────
DOCX_CONTENT = {
    "name": "Samuel Okonkwo",
    "contact": "samuel.okonkwo@dataeng.io | +1 (312) 555-0241 | Chicago, IL",
    "summary": (
        "Data Engineer with 4 years of experience building robust data pipelines "
        "and warehousing solutions. Proficient in Apache Spark, Airflow, dbt and "
        "cloud data platforms (AWS, GCP). Processed 5TB+ data daily and reduced "
        "pipeline failure rate by 60%."
    ),
    "experience": [
        {
            "title": "Senior Data Engineer | PipelineHub | 2022 - Present",
            "bullets": [
                "Designed and maintained ETL pipelines with Apache Spark and Airflow processing 5TB+ data daily",
                "Built dbt models for a Snowflake data warehouse serving 50+ BI analysts",
                "Implemented data quality checks reducing downstream reporting errors by 60%",
                "Migrated on-premise Hadoop cluster to AWS EMR saving $300K in infrastructure annually",
                "Created real-time streaming pipelines with Apache Kafka and Flink for fraud detection",
            ],
        },
        {
            "title": "Data Engineer | AnalyticsPro | 2020 - 2022",
            "bullets": [
                "Built Python ETL scripts ingesting data from 20+ REST APIs into PostgreSQL and BigQuery",
                "Optimised SQL queries reducing average report generation time from 45 seconds to 3 seconds",
                "Set up automated data monitoring and alerting with Great Expectations and PagerDuty",
                "Collaborated with data scientists to deploy ML features to the feature store",
            ],
        },
    ],
    "education": "B.S. Computer Science | University of Illinois Urbana-Champaign | 2020",
    "skills": (
        "Languages: Python, SQL, Scala, Bash\n"
        "Big Data: Apache Spark, Apache Kafka, Apache Airflow, Apache Flink, Hadoop\n"
        "Cloud: AWS (S3, EMR, Glue, Redshift, Lambda), GCP (BigQuery, Dataflow, Composer)\n"
        "Databases: PostgreSQL, MySQL, MongoDB, Snowflake, Cassandra, Redis\n"
        "Tools: dbt, Git, Docker, Kubernetes, Terraform, Tableau, Great Expectations\n"
        "Soft Skills: problem solving, communication, teamwork, attention to detail"
    ),
    "certifications": [
        "AWS Certified Data Analytics Specialty | 2023",
        "Google Professional Data Engineer | 2022",
        "dbt Certified Developer | 2023",
    ],
}


def create_docx_resume() -> None:
    """Create a DOCX resume using python-docx."""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        print("python-docx not installed. Run: pip install python-docx")
        return

    doc = Document()

    # Title / Name
    title = doc.add_heading(DOCX_CONTENT["name"], level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Contact
    contact_para = doc.add_paragraph(DOCX_CONTENT["contact"])
    contact_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Summary", level=2)
    doc.add_paragraph(DOCX_CONTENT["summary"])

    doc.add_heading("Experience", level=2)
    for exp in DOCX_CONTENT["experience"]:
        p = doc.add_paragraph()
        run = p.add_run(exp["title"])
        run.bold = True
        for bullet in exp["bullets"]:
            doc.add_paragraph(bullet, style="List Bullet")

    doc.add_heading("Education", level=2)
    doc.add_paragraph(DOCX_CONTENT["education"])

    doc.add_heading("Skills", level=2)
    doc.add_paragraph(DOCX_CONTENT["skills"])

    doc.add_heading("Certifications", level=2)
    for cert in DOCX_CONTENT["certifications"]:
        doc.add_paragraph(cert, style="List Bullet")

    path = RESUMES_DIR / "samuel_okonkwo_data_engineer.docx"
    doc.save(str(path))
    print(f"Created DOCX resume: {path}")


# ── PDF resume: ML Engineer ───────────────────────────────────────────────────
PDF_TEXT = """\
Elena Vasquez
elena.vasquez@mlengineer.ai | +1 (408) 555-0378 | San Jose, CA

Summary
Machine Learning Engineer with 3 years of experience building and deploying production ML systems.
Specialised in NLP and recommender systems. Deployed models serving 10M+ daily predictions with
sub-100ms latency. Reduced model training time by 45% through distributed computing optimisations.

Experience
ML Engineer | NeuralPath Inc. | 2022 - Present
- Built and deployed NLP models (BERT, RoBERTa) for document classification achieving 94% F1 score
- Designed MLOps pipelines with MLflow, Kubeflow and Docker reducing model release cycle from 2 weeks to 2 days
- Implemented feature store using Feast reducing feature engineering duplication by 70%
- Optimised model serving with TorchServe and ONNX achieving 60ms average inference latency
- Built A/B testing framework for model experimentation used by 5 ML teams

Junior Data Scientist | TechInsights | 2021 - 2022
- Developed recommender system using collaborative filtering improving click-through rate by 22%
- Built data preprocessing pipelines with pandas and NumPy processing 500K records daily
- Created data visualisation dashboards with Plotly and Streamlit for model performance monitoring

Education
M.S. Artificial Intelligence | Carnegie Mellon University | 2021
B.S. Mathematics | University of Texas Austin | 2019

Skills
ML Frameworks: PyTorch, TensorFlow, Keras, scikit-learn, HuggingFace Transformers, XGBoost
MLOps: MLflow, Kubeflow, Feast, Docker, Kubernetes, GitHub Actions, CI/CD
Languages: Python, SQL, R, Bash
Cloud: AWS (SageMaker, S3, EC2, Lambda), GCP (Vertex AI, BigQuery)
Data: pandas, NumPy, Apache Spark, Airflow, Kafka
NLP: natural language processing, BERT, transformers, text classification, named entity recognition
Databases: PostgreSQL, MongoDB, Redis, Pinecone (vector database)
Soft Skills: research, communication, problem solving, collaboration, time management

Projects
Semantic Search Engine - Built dense retrieval system with bi-encoder models; 35% improvement over BM25
Fraud Detection System - Real-time fraud scoring with XGBoost; $2M+ fraud prevented in first year

Certifications
AWS Certified Machine Learning Specialty | 2023
Deep Learning Specialisation (Coursera/DeepLearning.AI) | 2022
"""


def create_pdf_resume() -> None:
    """Create a PDF resume using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib import colors
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab")
        # Fallback: write as .txt renamed to .pdf won't work, so write a .txt instead
        path = RESUMES_DIR / "elena_vasquez_ml_engineer.txt"
        path.write_text(PDF_TEXT, encoding="utf-8")
        print(f"reportlab not available; created TXT instead: {path}")
        return

    path = RESUMES_DIR / "elena_vasquez_ml_engineer.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "Name", parent=styles["Heading1"], alignment=TA_CENTER, fontSize=16, spaceAfter=4
    )
    contact_style = ParagraphStyle(
        "Contact", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, spaceAfter=8
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=12,
        textColor=colors.HexColor("#2c3e50"), spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=styles["Normal"], fontSize=10, leading=14, leftIndent=16,
        bulletIndent=8
    )

    story = []
    lines = PDF_TEXT.strip().splitlines()

    story.append(Paragraph(lines[0], name_style))
    story.append(Paragraph(lines[1], contact_style))

    current_section = None
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue

        lower = stripped.lower()
        if lower in ("summary", "experience", "education", "skills", "projects", "certifications"):
            current_section = lower
            story.append(Paragraph(stripped, heading_style))
        elif stripped.startswith("- "):
            story.append(Paragraph("• " + stripped[2:], bullet_style))
        else:
            story.append(Paragraph(stripped, body_style))

    doc.build(story)
    print(f"Created PDF resume: {path}")


if __name__ == "__main__":
    create_docx_resume()
    create_pdf_resume()
    print("Done! Sample resumes created in data/sample_resumes/")
