"""AI Resume Assistant - ATS score checker built with Streamlit + Gemini."""

import io
import json
import os
import re

import streamlit as st
from docx import Document
from pypdf import PdfReader

DEFAULT_MODEL = "gemini-3.6-flash"  # change in the sidebar or via GEMINI_MODEL
MAX_RESUME_CHARS = 15000
MAX_JD_CHARS = 6000
MIN_RESUME_CHARS = 200

SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and \
professional resume coach. Evaluate the resume strictly and honestly. Do not \
inflate scores. Only reference content that is actually in the resume. \
Respond with valid JSON only."""

JSON_SCHEMA_HINT = """{
  "overall_score": <integer 0-100>,
  "summary": "<2-3 sentence overall assessment>",
  "section_scores": {
    "formatting": <integer 0-100>,
    "keywords": <integer 0-100>,
    "content_quality": <integer 0-100>,
    "structure": <integer 0-100>,
    "impact": <integer 0-100>
  },
  "strengths": ["<string>", "..."],
  "missing_keywords": ["<string>", "..."],
  "improvements": [
    {"priority": "High" | "Medium" | "Low", "issue": "<string>", "suggestion": "<string>"}
  ],
  "rewrite_examples": [
    {"original": "<line from the resume>", "improved": "<stronger version>"}
  ]
}"""


# ---------------------------------------------------------------- extraction
def extract_text(file_name: str, data: bytes) -> str:
    """Extract plain text from PDF, DOCX or TXT bytes."""
    name = file_name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password protected.")
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts).strip()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()
    raise ValueError("Unsupported file type. Upload a PDF, DOCX or TXT file.")


# ------------------------------------------------------------------- prompt
def build_prompt(resume_text: str, job_description: str = "") -> str:
    jd = job_description.strip()
    if jd:
        jd_block = (
            "A target job description is provided. Score keyword match and "
            "relevance against it, and list keywords from it that are missing "
            f"from the resume.\n\nJOB DESCRIPTION:\n{jd[:MAX_JD_CHARS]}\n"
        )
    else:
        jd_block = (
            "No job description was provided. Judge keywords against general "
            "ATS best practices for the role the resume appears to target.\n"
        )
    return (
        f"{jd_block}\nRESUME:\n{resume_text[:MAX_RESUME_CHARS]}\n\n"
        "Return ONLY a JSON object in exactly this shape:\n"
        f"{JSON_SCHEMA_HINT}\n"
        "Give 3-6 strengths, up to 12 missing keywords, 4-8 improvements "
        "ordered by priority, and 2-4 rewrite examples."
    )


# ------------------------------------------------------------------ parsing
def _clamp(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def parse_response(raw: str) -> dict:
    """Turn the model output into a validated dict. Raises ValueError if unusable."""
    if not raw or not raw.strip():
        raise ValueError("The model returned an empty response.")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("The model response did not contain JSON.")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse the model response: {exc}")
    if not isinstance(data, dict):
        raise ValueError("Unexpected response format.")

    sections = data.get("section_scores")
    sections = sections if isinstance(sections, dict) else {}
    result = {
        "overall_score": _clamp(data.get("overall_score")),
        "summary": str(data.get("summary", "")).strip(),
        "section_scores": {str(k): _clamp(v) for k, v in sections.items()},
        "strengths": [str(s) for s in (data.get("strengths") or []) if s],
        "missing_keywords": [str(s) for s in (data.get("missing_keywords") or []) if s],
        "improvements": [],
        "rewrite_examples": [],
    }
    for item in data.get("improvements") or []:
        if isinstance(item, dict):
            priority = str(item.get("priority", "Medium")).capitalize()
            if priority not in ("High", "Medium", "Low"):
                priority = "Medium"
            result["improvements"].append(
                {
                    "priority": priority,
                    "issue": str(item.get("issue", "")).strip(),
                    "suggestion": str(item.get("suggestion", "")).strip(),
                }
            )
        elif item:
            result["improvements"].append(
                {"priority": "Medium", "issue": "", "suggestion": str(item)}
            )
    for item in data.get("rewrite_examples") or []:
        if isinstance(item, dict) and (item.get("original") or item.get("improved")):
            result["rewrite_examples"].append(
                {
                    "original": str(item.get("original", "")).strip(),
                    "improved": str(item.get("improved", "")).strip(),
                }
            )
    order = {"High": 0, "Medium": 1, "Low": 2}
    result["improvements"].sort(key=lambda i: order[i["priority"]])
    return result


# ------------------------------------------------------------------- gemini
def analyze_resume(api_key: str, model: str, resume_text: str, job_description: str = "") -> dict:
    # Imported here so the rest of the module stays importable without the SDK.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(resume_text, job_description),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    return parse_response(response.text)


# ----------------------------------------------------------------------- ui
def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:  # no secrets file locally
        value = ""
    return value or os.environ.get(name, "")


def score_label(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Fair - needs work"
    return "Weak - major improvements needed"


def render_results(result: dict) -> None:
    score = result["overall_score"]
    st.subheader("Your ATS score")
    col1, col2 = st.columns([1, 3])
    col1.metric("Overall", f"{score}/100")
    with col2:
        st.write(score_label(score))
        st.progress(score / 100)
    if result["summary"]:
        st.write(result["summary"])

    if result["section_scores"]:
        st.subheader("Score breakdown")
        for name, value in result["section_scores"].items():
            st.write(f"**{name.replace('_', ' ').title()}** - {value}/100")
            st.progress(value / 100)

    if result["strengths"]:
        st.subheader("Strengths")
        for s in result["strengths"]:
            st.markdown(f"- {s}")

    if result["missing_keywords"]:
        st.subheader("Missing keywords")
        st.write(", ".join(f"`{k}`" for k in result["missing_keywords"]))

    if result["improvements"]:
        st.subheader("Suggested improvements")
        icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
        for item in result["improvements"]:
            title = item["issue"] or "Suggestion"
            with st.expander(f"{icons[item['priority']]} {item['priority']}: {title}"):
                st.write(item["suggestion"])

    if result["rewrite_examples"]:
        st.subheader("Example rewrites")
        for ex in result["rewrite_examples"]:
            st.markdown(f"**Before:** {ex['original']}")
            st.markdown(f"**After:** {ex['improved']}")
            st.divider()

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(result, indent=2),
        file_name="ats_report.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="AI Resume Assistant", page_icon="📄", layout="centered")
    st.title("📄 AI Resume Assistant")
    st.caption("Upload your resume to get an ATS score and concrete ways to improve it.")

    with st.sidebar:
        st.header("Settings")
        secret_key = get_secret("GEMINI_API_KEY")
        api_key = secret_key
        if not secret_key:
            api_key = st.text_input("Gemini API key", type="password",
                                    help="Get a free key at aistudio.google.com")
        model = st.text_input("Gemini model", value=get_secret("GEMINI_MODEL") or DEFAULT_MODEL)
        st.caption("Your resume is sent to Google's Gemini API for analysis.")

    uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    job_description = st.text_area(
        "Job description (optional)",
        height=150,
        placeholder="Paste the job posting to get a role-specific keyword match...",
    )

    if st.button("Analyze resume", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            return
        if uploaded is None:
            st.error("Please upload a resume first.")
            return
        try:
            text = extract_text(uploaded.name, uploaded.getvalue())
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
        if len(text) < MIN_RESUME_CHARS:
            st.error(
                "Very little text could be extracted. If this is a scanned or "
                "image-based PDF, ATS systems can't read it either - export a "
                "text-based PDF or DOCX instead."
            )
            return
        with st.spinner("Analyzing your resume..."):
            try:
                result = analyze_resume(api_key, model.strip() or DEFAULT_MODEL, text, job_description)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return
        render_results(result)


if __name__ == "__main__":
    main()# AI-Resume-Assistant1235234
